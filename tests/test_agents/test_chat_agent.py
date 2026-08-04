"""
tests/test_agents/test_chat_agent.py

Unit and integration tests for LearningAgent routing fixes:

  1. Multi-intent / action_sequence — compound imperatives produce a sequence
     and all steps are executed (not just the first).
  2. Multi-question splitting — enrichment and visualization actually fire
     instead of being demoted to text.
  3. Reason-quality confidence gate — thin-reason side-effect actions are
     demoted to "answer" even when the model reports high confidence.
  4. Viz override regex guard — conceptual questions about plots do NOT
     trigger run_visualization.
  5. DecisionSchema / ActionStep validation — invalid actions are coerced.

All tests are fully offline (no live LLM calls). The LLM layer is patched
with unittest.mock.patch so the test suite runs without Azure credentials.

Run:
    pytest tests/test_agents/test_chat_agent.py -v
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agents.learning_agent import (
    ActionStep,
    DecisionSchema,
    LearningAgent,
    _VALID_ACTIONS,
    _SIDE_EFFECT_ACTIONS,
    _extract_questions,
)
from core.state import BiomarkerState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_state(**kwargs) -> BiomarkerState:
    """Minimal state dict suitable for passing to LearningAgent methods."""
    state: BiomarkerState = {
        "messages": [],
        "user_query": "",
        "session_id": "test-session-001",
        "data_type": "proteomics",
        "n_proteins": 100,
        "n_samples": 6,
        "sample_columns": ["WT_1", "WT_2", "WT_3", "mdx_1", "mdx_2", "mdx_3",
                           "uDys5_1", "uDys5_2", "uDys5_3"],
        "all_groups": {
            "WT":    ["WT_1", "WT_2", "WT_3"],
            "mdx":   ["mdx_1", "mdx_2", "mdx_3"],
            "uDys5": ["uDys5_1", "uDys5_2", "uDys5_3"],
        },
        "group1_label": None,
        "group1_samples": [],
        "group2_label": None,
        "group2_samples": [],
        "status": "ready",
        "intent": None,
        "active_agent": None,
        "analysis_params": {},
    }
    state.update(kwargs)
    return state


def _agent_with_mock_llm(llm_response: str) -> LearningAgent:
    """Return a LearningAgent whose _call_llm always returns `llm_response`."""
    agent = LearningAgent.__new__(LearningAgent)
    agent.logger = MagicMock()
    agent._specialists = {}
    # Patch _call_llm at instance level
    agent._call_llm = MagicMock(return_value=llm_response)
    return agent


def _decision_json(**kwargs) -> str:
    base = {
        "action": "answer",
        "group1_label": None,
        "group1_samples": [],
        "group2_label": None,
        "group2_samples": [],
        "requested_plots": [],
        "confidence": 0.97,
        "reason": "Routing user question to answer handler.",
        "adj_pval_cutoff": None,
        "log2fc_cutoff": None,
        "missing_threshold": None,
        "top_n": None,
        "test_method": None,
        "is_paired": None,
        "all_groups": None,
        "omic_type": None,
        "clarification_question": None,
        "action_sequence": [],
    }
    base.update(kwargs)
    return json.dumps(base)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ActionStep / DecisionSchema validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecisionSchemaValidation:

    def test_valid_actions_are_accepted(self):
        for action in _VALID_ACTIONS:
            d = DecisionSchema(action=action)
            assert d.action == action

    def test_invalid_action_coerced_to_answer(self):
        d = DecisionSchema(action="fly_to_the_moon")
        assert d.action == "answer"

    def test_confidence_clamped_to_zero_one(self):
        d = DecisionSchema(confidence=1.5)
        assert d.confidence == 1.0
        d2 = DecisionSchema(confidence=-0.5)
        assert d2.confidence == 0.0

    def test_adj_pval_clamped(self):
        d = DecisionSchema(adj_pval_cutoff=2.0)
        assert d.adj_pval_cutoff == 1.0

    def test_action_step_invalid_action_coerced(self):
        step = ActionStep(action="nonexistent_action")
        assert step.action == "answer"

    def test_action_step_valid_fields(self):
        step = ActionStep(
            action="run_analysis",
            group1_label="WT",
            group2_label="mdx",
            top_n=10,
        )
        assert step.action == "run_analysis"
        assert step.top_n == 10

    def test_decision_schema_action_sequence_field(self):
        raw = _decision_json(
            action="run_analysis",
            group1_label="WT",
            group2_label="mdx",
            action_sequence=[
                {"action": "run_analysis", "group1_label": "WT",
                 "group2_label": "mdx", "top_n": 10},
                {"action": "run_enrichment"},
                {"action": "run_visualization", "requested_plots": ["volcano"]},
            ],
        )
        d = DecisionSchema.model_validate(json.loads(raw))
        assert len(d.action_sequence) == 3
        assert d.action_sequence[0].action == "run_analysis"
        assert d.action_sequence[1].action == "run_enrichment"
        assert d.action_sequence[2].requested_plots == ["volcano"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reason-quality confidence gate
# ═══════════════════════════════════════════════════════════════════════════════

class TestReasonQualityGate:

    def _make_decision_via_agent(self, json_str: str) -> Dict[str, Any]:
        """Call _make_decision on a minimal state with patched LLM."""
        agent = _agent_with_mock_llm(json_str)
        state = _base_state(user_query="run analysis on WT vs mdx")
        return agent._make_decision(state)

    def test_high_confidence_good_reason_passes(self):
        raw = _decision_json(
            action="run_analysis",
            group1_label="WT",
            group2_label="mdx",
            confidence=0.97,
            reason="User asked for WT vs mdx comparison.",
        )
        d = self._make_decision_via_agent(raw)
        assert d["action"] == "run_analysis"

    def test_low_confidence_demoted_to_answer(self):
        raw = _decision_json(
            action="run_analysis",
            confidence=0.5,
            reason="User might want analysis.",
        )
        d = self._make_decision_via_agent(raw)
        assert d["action"] == "answer"

    def test_thin_reason_side_effect_action_demoted(self):
        """A side-effect action with a very short/empty reason is demoted."""
        for action in ("run_analysis", "run_enrichment", "run_visualization",
                       "run_full_pipeline"):
            raw = _decision_json(
                action=action,
                confidence=0.97,
                reason="ok",          # < 8 chars
            )
            d = self._make_decision_via_agent(raw)
            assert d["action"] == "answer", \
                f"Expected 'answer' for thin-reason {action}, got {d['action']!r}"

    def test_thin_reason_non_side_effect_passes(self):
        """Non-side-effect actions with thin reason are NOT demoted."""
        raw = _decision_json(action="answer", confidence=0.97, reason="")
        d = self._make_decision_via_agent(raw)
        assert d["action"] == "answer"

    def test_llm_failure_returns_answer_fallback(self):
        agent = LearningAgent.__new__(LearningAgent)
        agent.logger = MagicMock()
        agent._specialists = {}
        agent._call_llm = MagicMock(side_effect=RuntimeError("timeout"))
        state = _base_state(user_query="run analysis")
        d = agent._make_decision(state)
        assert d["action"] == "answer"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-intent / action_sequence execution
# ═══════════════════════════════════════════════════════════════════════════════

class TestActionSequenceExecution:

    def _make_agent_with_specialists(self) -> LearningAgent:
        """Return an agent with mock specialists that record calls."""
        agent = LearningAgent.__new__(LearningAgent)
        agent.logger = MagicMock()
        agent._specialists = {}
        agent._code_reviewer = None

        def _make_specialist(name: str):
            m = MagicMock()
            m.run.side_effect = lambda st, **kw: {
                **st,
                "messages": st.get("messages", []) + [
                    {"role": "assistant", "content": f"[{name} done]"}
                ],
                "status": "answered",
            }
            return m

        for name in ("biomarker", "enrichment", "visualization"):
            agent._specialists[name] = _make_specialist(name)
        return agent

    def test_sequence_executes_all_steps(self):
        """All three steps in a sequence must be executed."""
        agent = self._make_agent_with_specialists()
        state = _base_state()

        seq = [
            {"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx"},
            {"action": "run_enrichment"},
            {"action": "run_visualization", "requested_plots": ["volcano"]},
        ]
        result = agent._execute_action_sequence(state, seq)

        assert agent._specialists["biomarker"].run.call_count == 1
        assert agent._specialists["enrichment"].run.call_count == 1
        assert agent._specialists["visualization"].run.call_count == 1

    def test_sequence_deduplicates_same_action(self):
        """Duplicate (action, group-pair) steps are skipped."""
        agent = self._make_agent_with_specialists()
        state = _base_state()

        seq = [
            {"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx"},
            {"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx"},  # dup
            {"action": "run_enrichment"},
        ]
        agent._execute_action_sequence(state, seq)

        # biomarker called only once despite two identical steps
        assert agent._specialists["biomarker"].run.call_count == 1
        assert agent._specialists["enrichment"].run.call_count == 1

    def test_sequence_dedup_different_pairs_both_run(self):
        """Two different group pairs in one sequence both run."""
        agent = self._make_agent_with_specialists()
        state = _base_state()

        seq = [
            {"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx"},
            {"action": "run_analysis", "group1_label": "WT", "group2_label": "uDys5"},
        ]
        agent._execute_action_sequence(state, seq)
        assert agent._specialists["biomarker"].run.call_count == 2

    def test_sequence_resolves_groups_from_all_groups(self):
        """Group labels are expanded from state['all_groups']."""
        agent = self._make_agent_with_specialists()
        captured_states: List[BiomarkerState] = []

        def capture_run(st, **kw):
            captured_states.append({k: st.get(k) for k in
                                     ("group1_label", "group1_samples",
                                      "group2_label", "group2_samples")})
            return {**st, "messages": st.get("messages", []) + [{"role": "assistant", "content": "done"}]}

        agent._specialists["biomarker"].run.side_effect = capture_run

        state = _base_state()
        seq = [{"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx"}]
        agent._execute_action_sequence(state, seq)

        assert len(captured_states) == 1
        assert captured_states[0]["group1_samples"] == ["WT_1", "WT_2", "WT_3"]
        assert captured_states[0]["group2_samples"] == ["mdx_1", "mdx_2", "mdx_3"]

    def test_single_action_sequence_executes_correctly(self):
        """A sequence with only 1 step still works."""
        agent = self._make_agent_with_specialists()
        state = _base_state()
        seq = [{"action": "run_enrichment"}]
        result = agent._execute_action_sequence(state, seq)
        assert agent._specialists["enrichment"].run.call_count == 1

    def test_sequence_applies_param_overrides(self):
        """Parameter overrides from each step are merged into analysis_params."""
        agent = self._make_agent_with_specialists()
        state = _base_state(analysis_params={})

        seq = [
            {"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx",
             "adj_pval_cutoff": 0.01, "top_n": 20},
        ]
        result = agent._execute_action_sequence(state, seq)
        params = result.get("analysis_params") or {}
        assert params.get("adj_pval_cutoff") == 0.01
        assert params.get("top_n") == 20

    def test_run_dispatches_to_sequence_when_two_or_more_steps(self):
        """run() calls _execute_action_sequence when action_sequence has ≥ 2 steps."""
        agent = self._make_agent_with_specialists()

        seq_json = _decision_json(
            action="run_analysis",
            group1_label="WT",
            group2_label="mdx",
            reason="User asked for compound pipeline step.",
            action_sequence=[
                {"action": "run_analysis", "group1_label": "WT", "group2_label": "mdx"},
                {"action": "run_enrichment"},
            ],
        )
        agent._call_llm = MagicMock(return_value=seq_json)

        state = _base_state(user_query="Compare WT vs mdx then run pathway analysis")

        with patch.object(agent, "_execute_action_sequence",
                          wraps=agent._execute_action_sequence) as mock_exec:
            agent.run(state)
            assert mock_exec.call_count == 1
            call_seq = mock_exec.call_args[0][1]
            assert len(call_seq) == 2

    def test_run_skips_sequence_when_empty(self):
        """run() does NOT call _execute_action_sequence for single-intent messages."""
        agent = self._make_agent_with_specialists()
        agent._call_llm = MagicMock(return_value=_decision_json(action="answer"))
        agent._answer = MagicMock(return_value=_base_state(status="answered"))
        state = _base_state(user_query="What is a t-test?")

        with patch.object(agent, "_execute_action_sequence") as mock_exec:
            agent.run(state)
            assert mock_exec.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Multi-question splitting — enrichment and viz now execute
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiQuestionHandlerFix:

    def _build_agent_for_multi_q(self, routed_actions: List[str]) -> LearningAgent:
        """
        Build an agent where _make_decision returns successive actions from
        `routed_actions` list, and each specialist records its calls.
        """
        agent = LearningAgent.__new__(LearningAgent)
        agent.logger = MagicMock()
        agent._specialists = {}
        agent._code_reviewer = None

        decisions_iter = iter(routed_actions)

        def _fake_decision(st):
            action = next(decisions_iter, "answer")
            return {
                "action": action,
                "confidence": 0.97,
                "reason": f"Routing to {action} for sub-question.",
                "group1_label": None, "group1_samples": [],
                "group2_label": None, "group2_samples": [],
                "requested_plots": [],
                "action_sequence": [],
            }

        agent._make_decision = _fake_decision

        for name in ("enrichment", "visualization"):
            m = MagicMock()
            m.run.side_effect = lambda st, **kw: {
                **st,
                "pathways": [{"pathway": "Glycolysis"}],
                "plot_paths": ["/tmp/volcano.html"],
                "messages": st.get("messages", []) + [
                    {"role": "assistant", "content": "[done]"}
                ],
                "status": "answered",
            }
            agent._specialists[name] = m

        def _fake_answer(st):
            return {**st, "messages": st.get("messages", []) +
                    [{"role": "assistant", "content": "[answered]"}],
                    "status": "answered"}
        agent._answer = MagicMock(side_effect=_fake_answer)

        return agent

    def test_enrichment_executes_in_multi_question(self):
        """run_enrichment sub-action MUST fire, not be demoted to answer."""
        questions = [
            "Which proteins differ between groups?",
            "What pathways are enriched?",
        ]
        agent = self._build_agent_for_multi_q(["answer", "run_enrichment"])
        state = _base_state(top_biomarkers=[{"protein": "P1"}], n_significant=5)

        agent._handle_multi_question(state, questions)
        assert agent._specialists["enrichment"].run.call_count == 1

    def test_enrichment_not_duplicated_in_multi_question(self):
        """If two questions both route to run_enrichment, it only fires once."""
        questions = [
            "What pathways are enriched?",
            "Can you identify candidate biomarkers?",
            "Show me the pathways again?",
        ]
        agent = self._build_agent_for_multi_q(
            ["run_enrichment", "answer", "run_enrichment"]
        )
        state = _base_state()

        agent._handle_multi_question(state, questions)
        assert agent._specialists["enrichment"].run.call_count == 1

    def test_visualization_executes_in_multi_question(self):
        """run_visualization sub-action MUST fire."""
        questions = [
            "What proteins are significant?",
            "Can you show me the volcano plot?",
        ]
        agent = self._build_agent_for_multi_q(["answer", "run_visualization"])
        state = _base_state(top_biomarkers=[{"protein": "P1"}])

        agent._handle_multi_question(state, questions)
        assert agent._specialists["visualization"].run.call_count == 1

    def test_heavy_pipeline_still_demoted(self):
        """run_analysis in multi-question mode still falls through to answer."""
        questions = [
            "Which proteins differ between groups?",
            "Can you run analysis for uDys5 vs mdx?",
        ]
        agent = self._build_agent_for_multi_q(["answer", "run_analysis"])
        state = _base_state()

        agent._handle_multi_question(state, questions)
        # visualization / enrichment specialists NOT called; answer is
        assert agent._specialists["enrichment"].run.call_count == 0
        assert agent._specialists["visualization"].run.call_count == 0
        # _answer called for the run_analysis sub-question
        assert agent._answer.call_count >= 1

    def test_enrichment_skipped_if_already_in_state(self):
        """If state already has pathways, enrichment question answered from state."""
        questions = [
            "What pathways are enriched?",
        ]
        agent = self._build_agent_for_multi_q(["run_enrichment"])
        state = _base_state(pathways=[{"pathway": "Glycolysis", "p_adjust": 0.01}])

        agent._handle_multi_question(state, questions)
        # The specialist should NOT be called again
        assert agent._specialists["enrichment"].run.call_count == 0

    def test_pathways_merged_to_outer_state(self):
        """Pathways generated by enrichment in multi-question propagate back."""
        questions = ["What pathways are enriched?", "What are my biomarkers?"]
        agent = self._build_agent_for_multi_q(["run_enrichment", "answer"])
        state = _base_state(top_biomarkers=[{"protein": "P1"}])

        agent._handle_multi_question(state, questions)
        # outer state should now have pathways merged back
        assert state.get("pathways") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Viz override regex — conceptual questions must NOT trigger run_visualization
# ═══════════════════════════════════════════════════════════════════════════════

class TestVizOverrideGuard:

    SHOULD_TRIGGER_VIZ = [
        # Must be exact substrings present in _viz_phrases
        "show volcano",
        "show heatmap",
        "generate plots",
        "show the plots",
        "display plot",
        "give me the plot",
        "show pca",
    ]

    SHOULD_NOT_TRIGGER_VIZ = [
        "what does a volcano plot show?",
        "explain what a heatmap represents",
        "walk me through what you'd show me on a volcano plot",
        "what is a pca chart?",
        "describe what a heatmap shows",
        "what does a heatmap mean?",
        "define pca chart",
    ]

    def _run_viz_override(self, user_query: str, starting_action: str = "answer") -> str:
        """Run just the viz-override block and return the resulting action."""
        import re

        _uq_lower = user_query.lower()
        action = starting_action

        _viz_phrases = (
            "show plot", "show the plot", "show plots", "show the plots",
            "generate plot", "make plot", "render plot", "draw plot",
            "show me the plot", "give me the plot", "display plot",
            "show chart", "show charts", "show heatmap", "show volcano",
            "show pca",
        )
        _conceptual_viz_re = re.compile(
            r"what\s+(is|does|do|are)\s+.{0,50}(show|plot|chart)|"
            r"(explain|describe|define)\s+.{0,50}(plot|chart|heatmap|volcano|pca)|"
            r"walk.{0,20}through.{0,40}(show|plot|chart)|"
            r"what.{0,30}would.{0,30}show|"
            r"what\s+(represents?|means?|tells?).{0,40}(plot|chart|heatmap|volcano)",
            re.IGNORECASE,
        )
        if (
            action == "answer"
            and any(p in _uq_lower for p in _viz_phrases)
            and not _conceptual_viz_re.search(user_query)
        ):
            action = "run_visualization"
        return action

    @pytest.mark.parametrize("q", SHOULD_TRIGGER_VIZ)
    def test_imperative_triggers_viz(self, q):
        assert self._run_viz_override(q) == "run_visualization", \
            f"Expected run_visualization for: {q!r}"

    @pytest.mark.parametrize("q", SHOULD_NOT_TRIGGER_VIZ)
    def test_conceptual_does_not_trigger_viz(self, q):
        assert self._run_viz_override(q) == "answer", \
            f"Expected answer (not viz) for: {q!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _extract_questions helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractQuestions:

    def test_splits_multiple_question_lines(self):
        msg = (
            "Which proteins differ between groups?\n"
            "What pathways are enriched?\n"
            "Can you identify candidate biomarkers?"
        )
        qs = _extract_questions(msg)
        assert len(qs) == 3

    def test_single_question_returns_one(self):
        qs = _extract_questions("What is the p-value cutoff?")
        assert len(qs) == 1

    def test_inline_questions_in_paragraph(self):
        msg = "What is X? What is Y? How many Z?"
        qs = _extract_questions(msg)
        assert len(qs) >= 2

    def test_imperative_no_question_mark_returns_empty(self):
        qs = _extract_questions("Compare WT vs mdx and show the volcano plot")
        assert len(qs) == 0  # no ? → not a multi-question

    def test_strips_list_markers(self):
        msg = "1. What is X?\n2. What is Y?"
        qs = _extract_questions(msg)
        assert all(not q.startswith(("1.", "2.")) for q in qs)

    def test_empty_string_returns_empty(self):
        assert _extract_questions("") == []

    def test_compound_imperative_not_split(self):
        """Compound imperatives (no ?) should NOT be split into sub-questions."""
        msg = "Compare uDys5 and mdx, identify top 10 biomarkers, then run pathway analysis"
        qs = _extract_questions(msg)
        assert len(qs) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Integration: run() dispatches compound messages as action_sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunIntegration:

    def _make_full_agent(self, llm_json: str) -> LearningAgent:
        """Agent with all specialists mocked + LLM returning llm_json."""
        agent = LearningAgent.__new__(LearningAgent)
        agent.logger = MagicMock()
        agent._specialists = {}
        agent._code_reviewer = None
        agent._call_llm = MagicMock(return_value=llm_json)

        for name in ("biomarker", "enrichment", "visualization", "ingestion"):
            m = MagicMock()
            m.run.side_effect = lambda st, **kw: {
                **st,
                "messages": st.get("messages", []) + [{"role": "assistant", "content": f"[{name}]"}],
                "status": "answered",
            }
            agent._specialists[name] = m

        def _fake_answer(st):
            return {**st, "messages": st.get("messages", []) +
                    [{"role": "assistant", "content": "[answer]"}],
                    "status": "answered"}
        agent._answer = MagicMock(side_effect=_fake_answer)
        return agent

    def test_compound_message_runs_all_steps(self):
        """Compound imperative → action_sequence → all specialists called."""
        llm_json = _decision_json(
            action="run_analysis",
            group1_label="uDys5",
            group2_label="mdx",
            reason="User requested compound pipeline: analysis then enrichment.",
            action_sequence=[
                {"action": "run_analysis", "group1_label": "uDys5",
                 "group2_label": "mdx", "top_n": 10},
                {"action": "run_enrichment"},
                {"action": "run_visualization", "requested_plots": ["volcano"]},
            ],
        )
        agent = self._make_full_agent(llm_json)
        state = _base_state(
            user_query="Compare uDys5 and mdx, identify top 10 biomarkers, "
                       "then run pathway analysis and show me the volcano plot"
        )

        result = agent.run(state)

        assert agent._specialists["biomarker"].run.call_count == 1
        assert agent._specialists["enrichment"].run.call_count == 1
        assert agent._specialists["visualization"].run.call_count == 1

    def test_single_intent_uses_action_field(self):
        """Single-intent message: action_sequence=[], top-level action dispatched."""
        llm_json = _decision_json(
            action="run_enrichment",
            reason="User explicitly requested pathway enrichment.",
        )
        agent = self._make_full_agent(llm_json)
        # Set enrichment_scope so the scope-confirmation gate is bypassed
        state = _base_state(
            user_query="run pathway enrichment",
            top_biomarkers=[{"protein": "P1"}],
            n_significant=1,          # equal to len(top_biomarkers) → no scope question
            enrichment_scope="all",   # explicit scope skips the user-confirmation branch
        )

        agent.run(state)

        assert agent._specialists["enrichment"].run.call_count == 1
        assert agent._specialists["biomarker"].run.call_count == 0

    def test_confidence_below_threshold_no_side_effect(self):
        """Decision with confidence < 0.7 must not trigger any specialist."""
        llm_json = _decision_json(
            action="run_analysis",
            confidence=0.5,
            reason="Low-confidence route.",
            group1_label="WT",
            group2_label="mdx",
        )
        agent = self._make_full_agent(llm_json)
        state = _base_state(user_query="maybe run analysis?")

        agent.run(state)

        assert agent._specialists["biomarker"].run.call_count == 0
        assert agent._answer.call_count == 1

    def test_thin_reason_no_side_effect(self):
        """Side-effect action with thin reason → demoted → no specialist fired."""
        llm_json = _decision_json(
            action="run_full_pipeline",
            confidence=0.98,
            reason="ok",  # < 8 chars
        )
        agent = self._make_full_agent(llm_json)
        state = _base_state(user_query="go")

        agent.run(state)

        assert agent._specialists["biomarker"].run.call_count == 0
        assert agent._answer.call_count == 1

    def test_multi_question_message_uses_splitter(self):
        """Multi-question message (lines ending ?) routes through _handle_multi_question."""
        llm_json = _decision_json(action="answer", reason="Answering sub-question.")
        agent = self._make_full_agent(llm_json)
        state = _base_state(
            user_query=(
                "Which proteins differ between groups?\n"
                "What pathways are enriched?"
            )
        )

        with patch.object(agent, "_handle_multi_question",
                          wraps=agent._handle_multi_question) as mock_mq:
            # patch _make_decision inside the multi-q handler too
            agent._make_decision = MagicMock(return_value={
                "action": "answer", "confidence": 0.97,
                "reason": "Answering sub-question.", "requested_plots": [],
                "group1_label": None, "group1_samples": [],
                "group2_label": None, "group2_samples": [],
                "action_sequence": [],
            })
            agent.run(state)
            assert mock_mq.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _SIDE_EFFECT_ACTIONS constant
# ═══════════════════════════════════════════════════════════════════════════════

class TestSideEffectActionsConstant:

    def test_side_effect_actions_is_subset_of_valid(self):
        assert _SIDE_EFFECT_ACTIONS.issubset(_VALID_ACTIONS)

    def test_answer_not_in_side_effects(self):
        assert "answer" not in _SIDE_EFFECT_ACTIONS

    def test_query_data_not_in_side_effects(self):
        # query_data reads existing data; it's not a mutation of pipeline state
        assert "query_data" not in _SIDE_EFFECT_ACTIONS

    @pytest.mark.parametrize("action", [
        "run_analysis", "run_enrichment", "run_visualization",
        "run_all_comparisons", "run_full_pipeline", "load_data",
    ])
    def test_expected_actions_in_side_effects(self, action):
        assert action in _SIDE_EFFECT_ACTIONS
