"""
tests/test_api/test_upload_route.py
Tests for POST /upload/ endpoint.
The IngestionAgent is mocked so no real file parsing or LLM calls occur.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.session_manager import SessionManager

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_bytes(rows: int = 5, cols: int = 6) -> bytes:
    """Generate a minimal CSV proteomics matrix."""
    header  = "Protein," + ",".join(f"S{i}" for i in range(1, cols + 1))
    data    = "\n".join(
        f"P{r:03d}," + ",".join(str(10.0 + r * 0.1 + c * 0.01) for c in range(cols))
        for r in range(1, rows + 1)
    )
    return (header + "\n" + data).encode()


def _mock_ingestion_success(session_id: str) -> dict:
    return {
        "session_id":       session_id,
        "status":           "data_loaded",
        "data_type":        "generic",
        "data_format":      "csv",
        "n_proteins":       5,
        "n_samples":        6,
        "sample_columns":   ["S1", "S2", "S3", "S4", "S5", "S6"],
        "metadata_columns": [],
        "is_pooled_design": False,
        "label_map":        None,
        "messages":         [{"role": "assistant", "content": "Data loaded."}],
    }


# ── File type validation ──────────────────────────────────────────────────────

class TestFileTypeValidation:

    def test_csv_accepted(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = _mock_ingestion_success(sid)
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
                data={"session_id": sid},
            )
        assert r.status_code == 201
        SessionManager.delete_session(sid)

    def test_xlsx_accepted(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = _mock_ingestion_success(sid)
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.xlsx", b"fake-excel-bytes", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"session_id": sid},
            )
        # xlsx passes format check; ingestion may fail on fake bytes — that's ok
        assert r.status_code in (201, 422)
        SessionManager.delete_session(sid)

    def test_txt_rejected_with_400(self):
        r = client.post(
            "/upload/",
            files={"file": ("data.txt", b"col1\tcol2\n1\t2", "text/plain")},
        )
        assert r.status_code == 400
        assert ".csv" in r.json()["detail"].lower() or "rename" in r.json()["detail"].lower()

    def test_pdf_rejected_with_400(self):
        r = client.post(
            "/upload/",
            files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 400

    def test_tsv_rejected_with_rename_hint(self):
        r = client.post(
            "/upload/",
            files={"file": ("data.tsv", b"P\tS1\tS2\nP001\t1.0\t2.0", "text/tab-separated-values")},
        )
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "rename" in detail.lower() or ".csv" in detail.lower()


# ── File size validation ──────────────────────────────────────────────────────

class TestFileSizeValidation:

    def test_oversized_file_returns_413(self):
        from config.settings import get_settings
        max_bytes = get_settings().max_file_size_mb * 1024 * 1024
        oversized = b"P," + b"S1," * 100 + b"\n" + b"0.1," * 1000
        # Artificially claim a large content via raw bytes
        big_content = b"x" * (max_bytes + 1)
        r = client.post(
            "/upload/",
            files={"file": ("big.csv", big_content, "text/csv")},
        )
        assert r.status_code == 413


# ── Session handling ──────────────────────────────────────────────────────────

class TestSessionHandling:

    def test_new_session_created_when_no_session_id(self):
        mock_agent = MagicMock()
        created_sid = None

        def _capture_run(state):
            nonlocal created_sid
            created_sid = state["session_id"]
            return _mock_ingestion_success(created_sid)

        mock_agent.run.side_effect = _capture_run
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
            )
        assert r.status_code == 201
        assert r.json()["session_id"]
        if created_sid:
            SessionManager.delete_session(created_sid)

    def test_existing_session_reused(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = _mock_ingestion_success(sid)
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
                data={"session_id": sid},
            )
        assert r.status_code == 201
        assert r.json()["session_id"] == sid
        SessionManager.delete_session(sid)

    def test_unknown_session_id_creates_new_session(self):
        mock_agent = MagicMock()
        new_sid = None

        def _capture(state):
            nonlocal new_sid
            new_sid = state["session_id"]
            return _mock_ingestion_success(new_sid)

        mock_agent.run.side_effect = _capture
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
                data={"session_id": "completely-unknown-xyz"},
            )
        assert r.status_code == 201
        returned_sid = r.json()["session_id"]
        assert returned_sid != "completely-unknown-xyz"
        if new_sid:
            SessionManager.delete_session(new_sid)


# ── Response schema ───────────────────────────────────────────────────────────

class TestUploadResponseSchema:

    def test_response_contains_required_fields(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = _mock_ingestion_success(sid)
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
                data={"session_id": sid},
            )
        body = r.json()
        for field in ("session_id", "file_id", "filename", "status"):
            assert field in body, f"Missing field: {field}"
        SessionManager.delete_session(sid)

    def test_n_proteins_and_n_samples_returned(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = _mock_ingestion_success(sid)
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
                data={"session_id": sid},
            )
        body = r.json()
        assert body["n_proteins"] == 5
        assert body["n_samples"]  == 6
        SessionManager.delete_session(sid)

    def test_sample_columns_returned(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = _mock_ingestion_success(sid)
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
                data={"session_id": sid},
            )
        cols = r.json()["sample_columns"]
        assert isinstance(cols, list)
        assert len(cols) == 6
        SessionManager.delete_session(sid)

    def test_ingestion_error_returns_422(self):
        sid = SessionManager.create_session()
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            **_mock_ingestion_success(sid),
            "status": "error",
            "error_message": "No numeric columns found.",
        }
        with patch("api.routes.upload._ingestion_agent", mock_agent):
            r = client.post(
                "/upload/",
                files={"file": ("bad.csv", b"col1\ncol2\n", "text/csv")},
                data={"session_id": sid},
            )
        assert r.status_code == 422
        SessionManager.delete_session(sid)
