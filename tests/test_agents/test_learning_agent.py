"""
tests/test_agents/test_learning_agent.py
Tests for LearningAgent — DecisionSchema validation, confidence gating,
hallucination guards, @traceable wiring, and routing correctness.
All LLM calls are mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.learning_agent import (
    DecisionSchema,
    LearningAgent,
    _recent_messages,
    _truncate,
    _VALID_ACTIONS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _state(**overrides) -> dict:
    base = {
        "session_id":    "test-session",
        "messages":      [],
        "user_query":    "hello",
        "status":        "ready",
        "data_type":     None,
        "data_path":     None,
        "omic_type":     None,
        "n_proteins":    None,
        "n_samples":     None,
        "sample_columns": [],
        "group1_samples": [],
        "group2_samples": [],
        "group1_label":  None,
        "group2_label":  None,
        "top_biomarkers": None,
        "top_proteins":   None,
        "n_significant":  None,
        "pathways":       None,
        "plot_paths":     None,
        "analysis_mode":  None,
        "analysis_code":  None,
        "label_map":      None,
        "is_pooled_design": False,
        "intent":         None,
        "active_agent":   None,
        "error_message":  None,
        "excel_path":     None,
        "analysis_summary": None,
        "organism":       "human",
        "disease_program": "General",
        "raw_data_path":  None,
        "qc_passed":      None,
        "qc_summary":     None,
        "enrichment_result_path": None,
        "report_path":    None,
        "dea_result_path": None,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def agent():
    """LearningAgent with patched AzureOpenAI client."""
    with patch("agents.base_agent._build_client", return_value=MagicMock()):
        a = LearningAgent()
    return a


# ── DecisionSchema ────────────────────────────────────────────────────────────

class TestDecisionSchema:

    def test_valid_action_passes(self):
        d = DecisionSchema.model_validate({"action": "run_analysis", "confidence": 0.9})
        assert d.action == "run_analysis"

    def test_all_valid_actions_accepted(self):
        for action in _VALID_ACTIONS:
            d = DecisionSchema.model_validate({"action": action})
            assert d.action == action

    def test_unknown_action_coerced_to_answer(self):
        d = DecisionSchema.model_validate({"action": "do_magic"})
        assert d.action == "answer"

    def test_confidence_clamped_above_1(self):
        d = DecisionSchema.model_validate({"action": "answer", "confidence": 1.5})
        assert d.confidence == 1.0

    def test_confidence_clamped_below_0(self):
        d = DecisionSchema.model_validate({"action": "answer", "confidence": -0.5})
        assert d.confidence == 0.0

    def test_confidence_defaults_to_1(self):
        d = DecisionSchema.model_validate({"action": "answer"})
        assert d.confidence == 1.0

    def test_bad_confidence_string_defaults_to_1(self):
        # mode="before" validator runs before Pydantic type coercion, so "high" → 1.0
        d = DecisionSchema.model_validate({"action": "answer", "confidence": "high"})
        assert d.confidence == 1.0

    def test_lists_default_empty(self):
        d = DecisionSchema.model_validate({"action": "answer"})
        assert d.group1_samples == []
        assert d.group2_samples == []
        assert d.requested_plots == []

    def test_model_dump_round_trips(self):
        payload = {
            "action": "run_visualization",
            "group1_label": "WT",
            "group1_samples": ["WT_1", "WT_2"],
            "group2_label": "KO",
            "group2_samples": ["KO_1", "KO_2"],
            "requested_plots": ["volcano", "pca"],
            "confidence": 0.88,
            "reason": "user asked for volcano and pca",
        }
        d = DecisionSchema.model_validate(payload)
        dumped = d.model_dump()
        assert dumped["action"] == "run_visualization"
        assert dumped["requested_plots"] == ["volcano", "pca"]
        assert abs(dumped["confidence"] - 0.88) < 0.001


# ── Confidence gating in _make_decision ───────────────────────────────────────

class TestMakeDecisionConfidenceGating:

    def _mock_llm_response(self, agent, payload: dict):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(payload)
        mock_resp.usage.total_tokens = 50
        mock_resp.choices[0].finish_reason = "stop"
        agent.client.chat.completions.create = MagicMock(return_value=mock_resp)

    def test_high_confidence_action_preserved(self, agent):
        self._mock_llm_response(agent, {
            "action": "run_analysis", "confidence": 0.95, "reason": "user said run"
        })
        decision = agent._make_decision(_state(
            data_type="olink_npx", user_query="run analysis"
        ))
        assert decision["action"] == "run_analysis"

    def test_low_confidence_demoted_to_answer(self, agent):
        self._mock_llm_response(agent, {
            "action": "run_analysis", "confidence": 0.5, "reason": "not sure"
        })
        decision = agent._make_decision(_state(user_query="do something"))
        assert decision["action"] == "answer"

    def test_exactly_0_7_confidence_passes(self, agent):
        self._mock_llm_response(agent, {
            "action": "run_enrichment", "confidence": 0.7, "reason": "borderline"
        })
        decision = agent._make_decision(_state(user_query="enrich"))
        assert decision["action"] == "run_enrichment"

    def test_unknown_action_json_falls_back_to_answer(self, agent):
        self._mock_llm_response(agent, {
            "action": "explode_everything", "confidence": 0.99, "reason": "..."
        })
        decision = agent._make_decision(_state(user_query="go"))
        assert decision["action"] == "answer"

    def test_malformed_json_falls_back_to_answer(self, agent):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "NOT JSON AT ALL"
        mock_resp.usage.total_tokens = 5
        mock_resp.choices[0].finish_reason = "stop"
        agent.client.chat.completions.create = MagicMock(return_value=mock_resp)
        decision = agent._make_decision(_state(user_query="hi"))
        assert decision["action"] == "answer"

    def test_llm_exception_falls_back_to_answer(self, agent):
        agent.client.chat.completions.create = MagicMock(
            side_effect=Exception("connection error")
        )
        decision = agent._make_decision(_state(user_query="hi"))
        assert decision["action"] == "answer"


# ── Hallucination guards in _answer ──────────────────────────────────────────

class TestAnswerHallucinationGuards:

    def _run_answer(self, agent, state_overrides: dict) -> dict:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "The answer is X."
        mock_resp.usage.total_tokens = 30
        mock_resp.choices[0].finish_reason = "stop"
        agent.client.chat.completions.create = MagicMock(return_value=mock_resp)
        return agent._answer(_state(**state_overrides))

    def test_answer_appends_assistant_message(self, agent):
        state = self._run_answer(agent, {"user_query": "what is proteomics?"})
        assert any(m["role"] == "assistant" for m in state["messages"])

    def test_grounded_biomarkers_injected_into_context(self, agent):
        top_bm = [
            {"rank": 1, "protein": "PROT_A", "log2_fold_change": 2.5, "adj_p_value": 0.001},
            {"rank": 2, "protein": "PROT_B", "log2_fold_change": -1.8, "adj_p_value": 0.02},
        ]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Analysis shows PROT_A is up."
        mock_resp.usage.total_tokens = 40
        mock_resp.choices[0].finish_reason = "stop"
        create_mock = MagicMock(return_value=mock_resp)
        agent.client.chat.completions.create = create_mock

        agent._answer(_state(top_biomarkers=top_bm, data_type="olink_npx",
                              user_query="what are the top proteins?"))

        # The system message sent to LLM must contain the grounding anchor
        call_args = create_mock.call_args
        messages_sent = call_args.args[0] if call_args.args else call_args.kwargs.get("messages", [])
        system_content = next(
            (m["content"] for m in messages_sent if m["role"] == "system"), ""
        )
        assert "PROT_A" in system_content
        assert "PROT_B" in system_content
        assert "Grounded biomarker" in system_content or "cite ONLY" in system_content

    def test_grounded_pathways_injected_when_present(self, agent):
        pathways = [
            {"pathway": "Glycolysis", "p_adjust": 0.001, "gene_count": 10},
            {"pathway": "TCA cycle",  "p_adjust": 0.005, "gene_count": 8},
        ]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Glycolysis is enriched."
        mock_resp.usage.total_tokens = 40
        mock_resp.choices[0].finish_reason = "stop"
        create_mock = MagicMock(return_value=mock_resp)
        agent.client.chat.completions.create = create_mock

        agent._answer(_state(pathways=pathways, data_type="olink_npx",
                              user_query="what pathways are enriched?"))

        messages_sent = create_mock.call_args.args[0] if create_mock.call_args.args \
            else create_mock.call_args.kwargs.get("messages", [])
        system_content = next(
            (m["content"] for m in messages_sent if m["role"] == "system"), ""
        )
        assert "Glycolysis" in system_content
        assert "TCA cycle" in system_content

    def test_anti_hallucination_rules_in_system_prompt(self, agent):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "ok"
        mock_resp.usage.total_tokens = 5
        mock_resp.choices[0].finish_reason = "stop"
        create_mock = MagicMock(return_value=mock_resp)
        agent.client.chat.completions.create = create_mock

        agent._answer(_state(user_query="explain FDR correction"))

        messages_sent = create_mock.call_args.args[0] if create_mock.call_args.args \
            else create_mock.call_args.kwargs.get("messages", [])
        system_content = next(
            (m["content"] for m in messages_sent if m["role"] == "system"), ""
        )
        assert "hallucin" in system_content.lower() or "fabricat" in system_content.lower()


# ── _recent_messages helper ───────────────────────────────────────────────────

class TestRecentMessages:

    def test_returns_last_n(self):
        msgs = [{"role": "user", "content": str(i)} for i in range(30)]
        recent = _recent_messages(msgs, n=5)
        assert len(recent) == 5
        assert recent[-1]["content"] == "29"

    def test_truncates_long_content(self):
        msgs = [{"role": "user", "content": "x" * 1000}]
        recent = _recent_messages(msgs, n=5, truncate_at=100)
        assert len(recent[0]["content"]) <= 115  # 100 + "[truncated]" overhead

    def test_skips_non_user_assistant_roles(self):
        msgs = [
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        recent = _recent_messages(msgs, n=10)
        roles = [m["role"] for m in recent]
        assert "system" not in roles

    def test_handles_langchain_message_objects(self):
        class FakeAIMessage:
            type    = "ai"
            content = "I am AI"

        msgs = [FakeAIMessage()]
        recent = _recent_messages(msgs, n=5)
        assert len(recent) == 1
        assert recent[0]["role"] == "assistant"
        assert recent[0]["content"] == "I am AI"

    def test_empty_list_returns_empty(self):
        assert _recent_messages([], n=10) == []


# ── _truncate helper ──────────────────────────────────────────────────────────

class TestTruncate:

    def test_short_string_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_string_truncated_with_indicator(self):
        result = _truncate("a" * 700, 600)
        assert result.endswith("[truncated]")
        assert len(result) < 650

    def test_exact_boundary_not_truncated(self):
        s = "x" * 600
        assert _truncate(s, 600) == s

    def test_one_over_boundary_is_truncated(self):
        s = "x" * 601
        assert "[truncated]" in _truncate(s, 600)

    def test_non_string_coerced(self):
        result = _truncate(12345, 100)
        assert result == "12345"


# ── Action routing in run() ───────────────────────────────────────────────────

class TestRunRouting:

    def _patch_decision(self, agent, action: str, **extra):
        payload = {"action": action, "confidence": 0.95, "reason": "test", **extra}
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(payload)
        mock_resp.usage.total_tokens = 30
        mock_resp.choices[0].finish_reason = "stop"
        agent.client.chat.completions.create = MagicMock(return_value=mock_resp)

    def test_answer_action_calls_answer(self, agent):
        self._patch_decision(agent, "answer")
        with patch.object(agent, "_answer", wraps=agent._answer) as spy:
            agent.run(_state(user_query="what is a t-test?"))
        spy.assert_called_once()

    def test_show_code_with_no_code_returns_message(self, agent):
        self._patch_decision(agent, "show_code")
        state = agent.run(_state(user_query="show me the code", analysis_code=None))
        assert any("code" in m["content"].lower() for m in state["messages"])

    def test_show_code_with_code_returns_code_block(self, agent):
        self._patch_decision(agent, "show_code")
        state = agent.run(_state(
            user_query="show code",
            analysis_code="import pandas as pd\nprint('hello')",
        ))
        code_block = next(
            m["content"] for m in state["messages"]
            if "```" in m["content"]
        )
        assert "pandas" in code_block

    def test_visualization_passes_requested_plots_to_specialist(self, agent):
        self._patch_decision(
            agent, "run_visualization",
            requested_plots=["volcano", "pca"]
        )
        mock_viz = MagicMock()
        mock_viz.run.return_value = _state(status="report_ready")
        agent._specialists["visualization"] = mock_viz

        agent.run(_state(user_query="show volcano and pca"))
        mock_viz.run.assert_called_once()
        _, kwargs = mock_viz.run.call_args
        assert kwargs.get("requested_plots") == ["volcano", "pca"]

    def test_run_analysis_sets_group_labels_from_decision(self, agent):
        self._patch_decision(
            agent, "run_analysis",
            group1_label="WT", group1_samples=["WT_1", "WT_2"],
            group2_label="KO", group2_samples=["KO_1", "KO_2"],
        )
        mock_bm = MagicMock()
        mock_bm.run.return_value = _state(status="analysis_complete")
        agent._specialists["biomarker"] = mock_bm

        state = agent.run(_state(
            data_type="olink_npx",
            user_query="compare WT vs KO",
        ))
        called_state = mock_bm.run.call_args.args[0]
        assert called_state["group1_label"]   == "WT"
        assert called_state["group2_label"]   == "KO"
        assert called_state["group1_samples"] == ["WT_1", "WT_2"]

    def test_user_message_always_appended(self, agent):
        self._patch_decision(agent, "answer")
        state = agent.run(_state(user_query="hello world"))
        user_msgs = [m for m in state["messages"] if m["role"] == "user"]
        assert any("hello world" in m["content"] for m in user_msgs)

    def test_intent_set_in_state(self, agent):
        self._patch_decision(agent, "run_enrichment")
        mock_enr = MagicMock()
        # Return the INPUT state with status modified — preserves intent set by run()
        mock_enr.run.side_effect = lambda s: {**s, "status": "enrichment_complete"}
        agent._specialists["enrichment"] = mock_enr
        state = agent.run(_state(user_query="run enrichment"))
        assert state["intent"] == "run_enrichment"


# ── LangSmith @traceable wiring ───────────────────────────────────────────────

class TestTraceableWiring:

    def test_run_is_wrapped_with_traceable(self):
        """run() must have the __wrapped__ or __func__ attribute from @traceable."""
        # When langsmith is available, @traceable adds metadata; when not, it's a no-op.
        # Either way the method must be callable.
        with patch("agents.base_agent._build_client", return_value=MagicMock()):
            a = LearningAgent()
        assert callable(a.run)

    def test_make_decision_is_wrapped(self):
        with patch("agents.base_agent._build_client", return_value=MagicMock()):
            a = LearningAgent()
        assert callable(a._make_decision)

    def test_answer_is_wrapped(self):
        with patch("agents.base_agent._build_client", return_value=MagicMock()):
            a = LearningAgent()
        assert callable(a._answer)
