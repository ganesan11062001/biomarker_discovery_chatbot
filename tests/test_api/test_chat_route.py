"""
tests/test_api/test_chat_route.py
Tests for POST /chat/ and POST /chat/session endpoints.
No real LLM or Azure calls — workflow and LLM are fully mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.session_manager import SessionManager

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_workflow_response(action: str = "answer", message: str = "Test response.") -> dict:
    """Return a minimal state dict that the mocked workflow.invoke() will return."""
    return {
        "session_id":    "test-session-id",
        "messages":      [
            {"role": "user",      "content": "test query"},
            {"role": "assistant", "content": message},
        ],
        "status":        "ok",
        "intent":        action,
        "user_query":    "test query",
        "data_type":     None,
        "top_biomarkers": None,
        "plot_paths":    None,
        "excel_path":    None,
        "error_message": None,
    }


# ── POST /chat/session ────────────────────────────────────────────────────────

class TestCreateSession:

    def test_returns_201_with_session_id(self):
        r = client.post("/chat/session")
        assert r.status_code == 201
        body = r.json()
        assert "session_id" in body
        assert body["session_id"]
        SessionManager.delete_session(body["session_id"])

    def test_session_persisted_after_create(self):
        r = client.post("/chat/session")
        sid = r.json()["session_id"]
        state = SessionManager.get_session(sid)
        assert state["session_id"] == sid
        SessionManager.delete_session(sid)

    def test_custom_disease_program(self):
        r = client.post("/chat/session?disease_program=Oncology&organism=mouse")
        assert r.status_code == 201
        sid = r.json()["session_id"]
        state = SessionManager.get_session(sid)
        assert state["disease_program"] == "Oncology"
        assert state["organism"]        == "mouse"
        SessionManager.delete_session(sid)

    def test_multiple_sessions_are_independent(self):
        r1 = client.post("/chat/session")
        r2 = client.post("/chat/session")
        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]
        assert sid1 != sid2
        SessionManager.delete_session(sid1)
        SessionManager.delete_session(sid2)


# ── POST /chat/ ───────────────────────────────────────────────────────────────

class TestChat:

    def _create_session(self) -> str:
        sid = SessionManager.create_session()
        return sid

    def _mock_workflow(self, sid: str, message: str = "Mock answer."):
        """Return a context manager that patches get_workflow so no real LLM is called."""
        mock_wf = MagicMock()
        mock_wf.invoke.return_value = _make_workflow_response(message=message)
        mock_wf.invoke.return_value["session_id"] = sid
        mock_wf.invoke.return_value["messages"] = [
            {"role": "user",      "content": "test"},
            {"role": "assistant", "content": message},
        ]
        return patch("api.routes.chat.get_workflow", return_value=mock_wf)

    def test_valid_message_returns_200(self):
        sid = self._create_session()
        with self._mock_workflow(sid):
            r = client.post("/chat/", json={"session_id": sid, "message": "hello"})
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert "response" in body
        assert body["status"]
        SessionManager.delete_session(sid)

    def test_response_contains_assistant_text(self):
        sid = self._create_session()
        with self._mock_workflow(sid, message="Proteomics answer here."):
            r = client.post("/chat/", json={"session_id": sid, "message": "what is proteomics?"})
        assert r.status_code == 200
        assert "Proteomics answer here." in r.json()["response"]
        SessionManager.delete_session(sid)

    def test_unknown_session_triggers_session_expired_response(self):
        r = client.post("/chat/", json={
            "session_id": "nonexistent-session-xyz",
            "message":    "hello",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "session_expired"
        assert body["session_id"] != "nonexistent-session-xyz"
        SessionManager.delete_session(body["session_id"])

    def test_inline_group_overrides_applied(self):
        sid = self._create_session()
        with self._mock_workflow(sid) as mock_wf_factory:
            r = client.post("/chat/", json={
                "session_id":    sid,
                "message":       "compare WT vs KO",
                "group1_label":  "WT",
                "group2_label":  "KO",
                "group1_samples": ["WT_1", "WT_2"],
                "group2_samples": ["KO_1", "KO_2"],
            })
        assert r.status_code == 200
        state = SessionManager.get_session(sid)
        assert state.get("group1_label") == "WT"
        assert state.get("group2_label") == "KO"
        SessionManager.delete_session(sid)

    def test_workflow_error_returns_500(self):
        sid = self._create_session()
        mock_wf = MagicMock()
        mock_wf.invoke.side_effect = RuntimeError("pipeline crashed")
        with patch("api.routes.chat.get_workflow", return_value=mock_wf):
            r = client.post("/chat/", json={"session_id": sid, "message": "run analysis"})
        assert r.status_code == 500
        assert "pipeline" in r.json()["detail"].lower() or "crashed" in r.json()["detail"]
        SessionManager.delete_session(sid)

    def test_intent_field_in_response(self):
        sid = self._create_session()
        with self._mock_workflow(sid):
            r = client.post("/chat/", json={"session_id": sid, "message": "explain FDR"})
        body = r.json()
        assert "intent" in body
        SessionManager.delete_session(sid)

    def test_messages_not_duplicated_across_two_turns(self):
        """Two sequential chat turns must produce exactly 2 assistant messages total."""
        sid = self._create_session()

        for i in range(2):
            mock_wf = MagicMock()

            def _side_effect(state, _i=i):
                # Simulate real workflow: return input messages + new assistant message
                existing = list(state.get("messages") or [])
                return {
                    **state,
                    "session_id":    sid,
                    "messages":      existing + [{"role": "assistant", "content": f"Answer {_i}"}],
                    "status":        "ok",
                    "intent":        "answer",
                }
            mock_wf.invoke.side_effect = _side_effect
            with patch("api.routes.chat.get_workflow", return_value=mock_wf):
                client.post("/chat/", json={"session_id": sid, "message": f"question {i}"})

        state = SessionManager.get_session(sid)
        msgs = state.get("messages", [])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 2, f"Expected 2 assistant messages, got {len(assistant_msgs)}"
        SessionManager.delete_session(sid)
