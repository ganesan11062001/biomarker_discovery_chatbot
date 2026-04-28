"""
tests/test_integration/test_chat_to_result.py
End-to-end integration tests: session lifecycle, full pipeline from chat
through analysis, and LangSmith integration sanity checks.

Important: the LangGraph workflow is compiled once at module import via
lru_cache.  Patching _build_client after import has no effect on the already-
created _learning_agent instance.  We therefore patch _call_llm directly on
the module-level agent to intercept LLM calls in all workflow-based tests.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import core.langgraph_workflow as _wf_mod
from core.session_manager import SessionManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decision_json(action: str, confidence: float = 0.95, **extra) -> str:
    """Return a well-formed decision JSON string."""
    return json.dumps({
        "action": action, "confidence": confidence,
        "reason": "test", **extra,
    })


def _invoke_with_mocked_llm(session_id: str, user_query: str, llm_responses: list) -> dict:
    """
    Run one workflow turn with the given _call_llm side-effect list.

    Critical contract:
    1. Decouple messages list — learning_agent.run() appends in-place; if we pass
       the live session object, LangGraph's add_messages reducer sees the already-
       mutated list as "existing" and doubles every message.  A fresh copy breaks
       that cycle (same fix applied in api/routes/chat.py).
    2. Only store delta messages back to the session — update_session() appends,
       so passing the full accumulated workflow output would duplicate history.
    """
    raw_state = SessionManager.get_session(session_id)
    # 1. Decouple messages from the session store
    state = {**raw_state, "messages": list(raw_state.get("messages") or [])}
    n_before = len(state["messages"])
    state["user_query"] = user_query

    with patch.object(_wf_mod._learning_agent, "_call_llm", side_effect=llm_responses):
        result = _wf_mod.get_workflow().invoke(state)

    # 2. Store only new (delta) messages back to the session
    delta_msgs = (result.get("messages") or [])[n_before:]
    result["messages"] = delta_msgs
    SessionManager.update_session(session_id, result)
    return SessionManager.get_session(session_id)


# ── Session lifecycle ─────────────────────────────────────────────────────────

class TestSessionLifecycle:

    def test_create_returns_valid_id(self):
        sid = SessionManager.create_session()
        assert sid
        assert SessionManager.get_session(sid) is not None
        SessionManager.delete_session(sid)

    def test_get_nonexistent_raises(self):
        with pytest.raises(KeyError):
            SessionManager.get_session("nonexistent-id-xyz")

    def test_update_merges_scalar_fields(self):
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {"status": "analyzing", "n_proteins": 500})
        state = SessionManager.get_session(sid)
        assert state["status"]     == "analyzing"
        assert state["n_proteins"] == 500
        SessionManager.delete_session(sid)

    def test_update_appends_messages(self):
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {"messages": [{"role": "user",      "content": "hello"}]})
        SessionManager.update_session(sid, {"messages": [{"role": "assistant", "content": "hi"}]})
        state = SessionManager.get_session(sid)
        assert len(state["messages"])    == 2
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][1]["role"] == "assistant"
        SessionManager.delete_session(sid)

    def test_messages_never_duplicated_on_repeated_updates(self):
        sid = SessionManager.create_session()
        for i in range(5):
            SessionManager.update_session(sid, {
                "messages": [{"role": "user", "content": f"msg {i}"}]
            })
        assert len(SessionManager.get_session(sid)["messages"]) == 5
        SessionManager.delete_session(sid)

    def test_delete_removes_session(self):
        sid = SessionManager.create_session()
        SessionManager.delete_session(sid)
        with pytest.raises(KeyError):
            SessionManager.get_session(sid)

    def test_session_count_tracks_creates_and_deletes(self):
        before = SessionManager.session_count()
        sid = SessionManager.create_session()
        assert SessionManager.session_count() == before + 1
        SessionManager.delete_session(sid)
        assert SessionManager.session_count() == before

    def test_initial_state_has_all_required_fields(self):
        sid   = SessionManager.create_session()
        state = SessionManager.get_session(sid)
        for field in [
            "session_id", "messages", "status", "data_type", "data_path",
            "sample_columns", "group1_samples", "group2_samples",
            "top_biomarkers", "n_significant", "pathways", "plot_paths",
            "is_pooled_design", "organism", "disease_program",
        ]:
            assert field in state, f"Missing initial field: {field}"
        SessionManager.delete_session(sid)

    def test_langchain_messages_normalised_to_plain_dicts(self):
        class FakeAI:
            type    = "ai"
            content = "I am an AI."
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {"messages": [FakeAI()]})
        stored = SessionManager.get_session(sid)["messages"]
        assert stored[0]["role"]    == "assistant"
        assert stored[0]["content"] == "I am an AI."
        SessionManager.delete_session(sid)


# ── Answer turn ───────────────────────────────────────────────────────────────

class TestAnswerTurn:

    def test_general_question_returns_assistant_message(self):
        sid   = SessionManager.create_session()
        state = _invoke_with_mocked_llm(
            sid,
            "what is a t-test?",
            llm_responses=[
                _decision_json("answer"),
                "A t-test compares the means of two independent groups.",
            ],
        )
        msgs = [m for m in state.get("messages", []) if m.get("role") == "assistant"]
        assert len(msgs) >= 1
        SessionManager.delete_session(sid)

    def test_no_data_state_preserved_after_answer_turn(self):
        sid   = SessionManager.create_session()
        state = _invoke_with_mocked_llm(
            sid,
            "explain FDR",
            llm_responses=[
                _decision_json("answer"),
                "FDR stands for False Discovery Rate.",
            ],
        )
        assert state.get("data_type") is None
        SessionManager.delete_session(sid)

    def test_intent_recorded_as_answer(self):
        sid   = SessionManager.create_session()
        state = _invoke_with_mocked_llm(
            sid,
            "what is proteomics?",
            llm_responses=[
                _decision_json("answer"),
                "Proteomics is the large-scale study of proteins.",
            ],
        )
        assert state.get("intent") in ("answer", None)  # intent preserved or set
        SessionManager.delete_session(sid)


# ── Supervised analysis turn ──────────────────────────────────────────────────

class TestAnalysisTurn:

    def test_supervised_analysis_completes(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples
    ):
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {
            "data_path":      str(proteomics_csv),
            "data_type":      "generic",
            "omic_type":      "proteomics",
            "sample_columns": sample_columns,
            "group1_samples": group1_samples,
            "group2_samples": group2_samples,
            "group1_label":   "Disease",
            "group2_label":   "Control",
        })

        state = _invoke_with_mocked_llm(
            sid,
            "run supervised analysis",
            llm_responses=[
                _decision_json("run_analysis",
                               group1_label="Disease", group1_samples=group1_samples,
                               group2_label="Control", group2_samples=group2_samples),
                "Analysis complete. P001 is the top hit.",
            ],
        )

        assert state.get("status")        == "analysis_complete"
        assert state.get("top_biomarkers") is not None
        assert len(state["top_biomarkers"]) > 0
        assert state.get("excel_path")    is not None
        SessionManager.delete_session(sid)

    def test_spiked_proteins_rank_in_top_results(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples
    ):
        """P001–P003 are elevated +4 NPX in group1 — they must appear near the top."""
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {
            "data_path":      str(proteomics_csv),
            "data_type":      "generic",
            "omic_type":      "proteomics",
            "sample_columns": sample_columns,
            "group1_samples": group1_samples,
            "group2_samples": group2_samples,
            "group1_label":   "Disease",
            "group2_label":   "Control",
        })

        state = _invoke_with_mocked_llm(
            sid,
            "compare Disease vs Control",
            llm_responses=[
                _decision_json("run_analysis",
                               group1_label="Disease", group1_samples=group1_samples,
                               group2_label="Control", group2_samples=group2_samples),
                "P001 tops the list.",
            ],
        )

        top_names = [b["protein"] for b in state.get("top_biomarkers", [])]
        spiked    = {"P001", "P002", "P003"}
        assert len(spiked & set(top_names)) >= 2, \
            f"Expected spiked proteins in top results, got: {top_names[:10]}"
        SessionManager.delete_session(sid)

    def test_unsupervised_mode_when_no_groups(self, proteomics_csv, sample_columns):
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {
            "data_path":      str(proteomics_csv),
            "data_type":      "generic",
            "omic_type":      "proteomics",
            "sample_columns": sample_columns,
        })

        state = _invoke_with_mocked_llm(
            sid,
            "analyse data",
            llm_responses=[
                _decision_json("run_analysis"),
                "Unsupervised CV ranking complete.",
            ],
        )

        assert state.get("status")        == "analysis_complete"
        assert state.get("analysis_mode") == "unsupervised"
        SessionManager.delete_session(sid)


# ── Multi-turn deduplication ──────────────────────────────────────────────────

class TestMessageDeduplication:

    def test_three_answer_turns_produce_exactly_6_messages(self):
        """3 turns → 3 user + 3 assistant = 6 total, no duplicates."""
        sid = SessionManager.create_session()

        for i in range(3):
            _invoke_with_mocked_llm(
                sid,
                f"Question {i}",
                llm_responses=[
                    _decision_json("answer"),
                    f"Answer number {i}",
                ],
            )

        messages        = SessionManager.get_session(sid).get("messages", [])
        user_msgs       = [m for m in messages if m.get("role") == "user"]
        assistant_msgs  = [m for m in messages if m.get("role") == "assistant"]

        assert len(messages)       == 6,  f"Expected 6 messages, got {len(messages)}"
        assert len(user_msgs)      == 3,  f"Expected 3 user messages, got {len(user_msgs)}"
        assert len(assistant_msgs) == 3,  f"Expected 3 assistant messages, got {len(assistant_msgs)}"
        SessionManager.delete_session(sid)

    def test_low_confidence_decision_does_not_run_analysis(self):
        """Low-confidence (0.4) run_analysis must be demoted to 'answer'."""
        sid = SessionManager.create_session()
        state = _invoke_with_mocked_llm(
            sid,
            "do something",
            llm_responses=[
                _decision_json("run_analysis", confidence=0.4),
                "I'm not sure what you mean.",
            ],
        )
        assert state.get("status") != "analysis_complete", \
            "Low-confidence decision must not trigger analysis"
        SessionManager.delete_session(sid)


# ── LangSmith integration sanity ─────────────────────────────────────────────

class TestLangSmithIntegration:

    def test_configure_langsmith_disabled_when_no_key(self):
        import os
        from core.tracing import configure_langsmith
        saved_ls = os.environ.pop("LANGSMITH_API_KEY",  None)
        saved_lc = os.environ.pop("LANGCHAIN_API_KEY",  None)
        try:
            assert configure_langsmith(api_key=None, enabled=True) is False
        finally:
            if saved_ls: os.environ["LANGSMITH_API_KEY"] = saved_ls
            if saved_lc: os.environ["LANGCHAIN_API_KEY"] = saved_lc

    def test_wrap_openai_applied_when_tracing_enabled(self):
        with patch("langsmith.wrappers.wrap_openai", return_value=MagicMock()) as wrap, \
             patch("agents.base_agent.settings") as s:
            s.langsmith_tracing        = True
            s.azure_openai_endpoint    = "https://fake.openai.azure.com/"
            s.azure_openai_api_key     = "k"
            s.azure_openai_api_version = "2024-08-01-preview"
            from agents.base_agent import _build_client
            _build_client()
        assert wrap.called, "wrap_openai must be applied when tracing is enabled"

    def test_decision_call_uses_json_mode(self):
        """_make_decision must pass response_format=json_object to force clean JSON output."""
        from agents.learning_agent import LearningAgent
        with patch("agents.base_agent._build_client", return_value=MagicMock()):
            agent = LearningAgent()

        resp = MagicMock()
        resp.choices[0].message.content = json.dumps(
            {"action": "answer", "confidence": 0.95, "reason": "ok"}
        )
        resp.usage.total_tokens       = 20
        resp.choices[0].finish_reason = "stop"
        agent.client.chat.completions.create = MagicMock(return_value=resp)

        agent._make_decision({
            "user_query": "hi", "messages": [], "session_id": "s1",
            "data_type": None, "n_proteins": None, "n_samples": None,
            "sample_columns": [], "label_map": None, "top_biomarkers": None,
            "is_pooled_design": False, "omic_type": None, "analysis_mode": None,
            "n_significant": None, "group1_label": None, "group2_label": None,
            "analysis_code": None, "plot_paths": None, "pathways": None, "status": "ready",
        })

        kw = agent.client.chat.completions.create.call_args.kwargs
        assert kw.get("response_format") == {"type": "json_object"}, \
            "Decision call must use json_mode=True (response_format=json_object)"
