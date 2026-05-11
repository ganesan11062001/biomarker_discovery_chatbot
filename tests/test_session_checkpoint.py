"""
tests/test_session_checkpoint.py
Tests for SessionManager disk-checkpoint persistence and rehydration.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_checkpoint_dir(tmp_path, monkeypatch):
    """Point _CHECKPOINT_DIR at a tmp path for this test module."""
    import core.session_manager as sm
    monkeypatch.setattr(sm, "_CHECKPOINT_DIR", tmp_path / "sessions")
    # Reset the in-memory store between tests
    sm.SessionManager._sessions.clear()
    sm.SessionManager._last_accessed.clear()
    yield


class TestCheckpointSaveLoad:

    def test_create_session_writes_file(self, tmp_path):
        from core.session_manager import SessionManager, _CHECKPOINT_DIR
        sid = SessionManager.create_session(disease_program="DMD", organism="mouse")
        path = _CHECKPOINT_DIR / f"{sid}.json"
        assert path.exists()
        SessionManager.delete_session(sid)
        assert not path.exists()  # Delete also removes the file

    def test_update_session_updates_file(self):
        from core.session_manager import SessionManager, _CHECKPOINT_DIR
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {"n_proteins": 500, "status": "data_loaded"})
        payload = json.loads((_CHECKPOINT_DIR / f"{sid}.json").read_text())
        assert payload["state"]["n_proteins"] == 500
        assert payload["state"]["status"] == "data_loaded"
        SessionManager.delete_session(sid)

    def test_load_from_disk_rehydrates(self):
        from core.session_manager import SessionManager
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {"n_samples": 42})
        # Simulate process restart: drop in-memory store, reload from disk
        SessionManager._sessions.clear()
        SessionManager._last_accessed.clear()
        n = SessionManager.load_from_disk()
        assert n == 1
        state = SessionManager.get_session(sid)
        assert state["n_samples"] == 42
        SessionManager.delete_session(sid)

    def test_dataframe_not_serialised(self):
        """DataFrames are dropped from the on-disk checkpoint to keep it small."""
        import pandas as pd
        from core.session_manager import SessionManager, _CHECKPOINT_DIR
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {
            "all_sheets": {"main": pd.DataFrame({"a": [1, 2, 3]})},
        })
        payload = json.loads((_CHECKPOINT_DIR / f"{sid}.json").read_text())
        # all_sheets entry should be a string placeholder, not the actual DataFrame
        assert "DataFrame" in str(payload["state"]["all_sheets"]["main"])
        SessionManager.delete_session(sid)

    def test_messages_persist_across_reload(self):
        from core.session_manager import SessionManager
        sid = SessionManager.create_session()
        SessionManager.update_session(sid, {
            "messages": [{"role": "user", "content": "hi"}],
        })
        SessionManager._sessions.clear()
        SessionManager.load_from_disk()
        msgs = SessionManager.get_session(sid).get("messages", [])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"
        SessionManager.delete_session(sid)
