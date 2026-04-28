"""
tests/test_tracing.py
Unit tests for core/tracing.py — LangSmith configuration helpers.
All tests are offline (no real LangSmith API calls).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestConfigureLangsmith:

    def test_returns_false_when_disabled(self):
        from core.tracing import configure_langsmith
        result = configure_langsmith(enabled=False)
        assert result is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"

    def test_returns_false_when_no_api_key(self):
        from core.tracing import configure_langsmith
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGSMITH_API_KEY",  None)
            os.environ.pop("LANGCHAIN_API_KEY",  None)
            result = configure_langsmith(api_key=None, enabled=True)
        assert result is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"

    def test_returns_true_when_key_provided_directly(self):
        from core.tracing import configure_langsmith
        result = configure_langsmith(api_key="ls__fake_key", project="test-proj", enabled=True)
        assert result is True
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"]     == "ls__fake_key"
        assert os.environ["LANGCHAIN_PROJECT"]     == "test-proj"

    def test_reads_key_from_env_var(self):
        from core.tracing import configure_langsmith
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "ls__env_key"}):
            result = configure_langsmith(api_key=None, enabled=True)
        assert result is True

    def test_langchain_api_key_env_var_fallback(self):
        from core.tracing import configure_langsmith
        with patch.dict(os.environ, {
            "LANGCHAIN_API_KEY": "ls__lc_key",
            "LANGSMITH_API_KEY": "",
        }):
            # Unset LANGSMITH_API_KEY completely so fallback triggers
            env = {k: v for k, v in os.environ.items() if k != "LANGSMITH_API_KEY"}
            env["LANGCHAIN_API_KEY"] = "ls__lc_key"
            with patch.dict(os.environ, env, clear=True):
                result = configure_langsmith(api_key=None, enabled=True)
        assert result is True

    def test_project_name_defaults_to_biomarker_discovery(self):
        from core.tracing import configure_langsmith
        configure_langsmith(api_key="ls__key", enabled=True)
        assert os.environ["LANGCHAIN_PROJECT"] == "biomarker-discovery"

    def test_custom_project_name(self):
        from core.tracing import configure_langsmith
        configure_langsmith(api_key="ls__key", project="my-custom-project", enabled=True)
        assert os.environ["LANGCHAIN_PROJECT"] == "my-custom-project"


class TestGetTraceMetadata:

    def test_returns_dict_with_session_id(self):
        from core.tracing import get_trace_metadata
        state = {"session_id": "abc-123", "data_type": "olink_npx"}
        meta  = get_trace_metadata(state)
        assert meta["session_id"] == "abc-123"

    def test_handles_empty_state(self):
        from core.tracing import get_trace_metadata
        meta = get_trace_metadata({})
        assert meta["session_id"] == "unknown"
        assert meta["data_type"]  is None

    def test_includes_all_expected_keys(self):
        from core.tracing import get_trace_metadata
        expected = {
            "session_id", "data_type", "omic_type", "n_proteins",
            "n_samples", "analysis_mode", "is_pooled", "status",
            "active_agent", "intent",
        }
        meta = get_trace_metadata({})
        assert expected.issubset(meta.keys())

    def test_is_pooled_defaults_false(self):
        from core.tracing import get_trace_metadata
        meta = get_trace_metadata({})
        assert meta["is_pooled"] is False

    def test_all_values_json_serialisable(self):
        import json
        from core.tracing import get_trace_metadata
        state = {
            "session_id": "s1", "data_type": "ms_lfq", "n_proteins": 500,
            "n_samples": 12, "analysis_mode": "supervised", "is_pooled_design": True,
            "status": "analysis_complete", "active_agent": "biomarker", "intent": "run_analysis",
        }
        meta = get_trace_metadata(state)
        # Must not raise
        json.dumps(meta)
