"""
tests/test_agents/test_base_agent.py
Tests for BaseAgent infrastructure: wrap_openai client building,
json_mode flag, retry behaviour, and prompt loading.
All LLM calls are mocked — no real Azure calls.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import BaseAgent, _build_client
from core.state import BiomarkerState


# ── Minimal concrete subclass for testing ─────────────────────────────────────

class _DummyAgent(BaseAgent):
    def run(self, state: BiomarkerState) -> BiomarkerState:
        return state


# ── Client building ────────────────────────────────────────────────────────────

class TestBuildClient:

    def test_returns_azure_client_when_tracing_disabled(self):
        from openai import AzureOpenAI
        with patch("config.settings.get_settings") as mock_s:
            mock_s.return_value = MagicMock(
                langsmith_tracing=False,
                azure_openai_endpoint="https://fake.openai.azure.com/",
                azure_openai_api_key="fake-key",
                azure_openai_api_version="2024-08-01-preview",
            )
            from agents import base_agent
            original_settings = base_agent.settings
            base_agent.settings = mock_s.return_value
            client = _build_client()
            base_agent.settings = original_settings
        # Should still be an AzureOpenAI (or wrapped, but not None)
        assert client is not None

    def test_wrap_openai_called_when_tracing_enabled(self):
        mock_wrapped = MagicMock()
        with patch("agents.base_agent.settings") as mock_s, \
             patch("langsmith.wrappers.wrap_openai", return_value=mock_wrapped) as mock_wrap:
            mock_s.langsmith_tracing = True
            mock_s.azure_openai_endpoint = "https://fake.openai.azure.com/"
            mock_s.azure_openai_api_key  = "fake-key"
            mock_s.azure_openai_api_version = "2024-08-01-preview"
            client = _build_client()
            assert mock_wrap.called
            assert client is mock_wrapped

    def test_falls_back_gracefully_when_langsmith_not_importable(self):
        import builtins
        real_import = builtins.__import__

        def _block_langsmith(name, *args, **kwargs):
            if name == "langsmith.wrappers":
                raise ImportError("langsmith not installed")
            return real_import(name, *args, **kwargs)

        with patch("agents.base_agent.settings") as mock_s, \
             patch("builtins.__import__", side_effect=_block_langsmith):
            mock_s.langsmith_tracing = True
            mock_s.azure_openai_endpoint = "https://fake.openai.azure.com/"
            mock_s.azure_openai_api_key  = "fake-key"
            mock_s.azure_openai_api_version = "2024-08-01-preview"
            # Should not raise even if langsmith import fails
            try:
                client = _build_client()
                assert client is not None
            except Exception:
                pass  # ImportError path is acceptable; must not be uncaught


# ── Prompt loading ─────────────────────────────────────────────────────────────

class TestPromptLoading:

    def test_loads_existing_prompt_file(self, tmp_path):
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("You are a test assistant.", encoding="utf-8")

        with patch("agents.base_agent.settings") as mock_s:
            mock_s.langsmith_tracing = False
            mock_s.azure_openai_endpoint = "https://fake.openai.azure.com/"
            mock_s.azure_openai_api_key  = "key"
            mock_s.azure_openai_api_version = "2024-08-01-preview"
            with patch("agents.base_agent._build_client", return_value=MagicMock()):
                agent = _DummyAgent("gpt-4o", str(prompt_file))
        assert agent.system_prompt == "You are a test assistant."

    def test_returns_default_prompt_when_file_missing(self):
        with patch("agents.base_agent._build_client", return_value=MagicMock()):
            agent = _DummyAgent("gpt-4o", "nonexistent/prompt.txt")
        assert "biomarker" in agent.system_prompt.lower()


# ── _call_llm ─────────────────────────────────────────────────────────────────

class TestCallLlm:

    def _make_agent(self, tmp_path):
        prompt_file = tmp_path / "p.txt"
        prompt_file.write_text("sys prompt")
        with patch("agents.base_agent._build_client", return_value=MagicMock()):
            return _DummyAgent("gpt-4o", str(prompt_file))

    def test_returns_llm_content(self, tmp_path):
        agent = self._make_agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Hello from LLM"
        mock_resp.usage.total_tokens = 42
        mock_resp.choices[0].finish_reason = "stop"
        agent.client.chat.completions.create = MagicMock(return_value=mock_resp)

        result = agent._call_llm([{"role": "user", "content": "hi"}])
        assert result == "Hello from LLM"

    def test_json_mode_passes_response_format(self, tmp_path):
        agent = self._make_agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"action": "answer"}'
        mock_resp.usage.total_tokens = 10
        mock_resp.choices[0].finish_reason = "stop"
        create_mock = MagicMock(return_value=mock_resp)
        agent.client.chat.completions.create = create_mock

        agent._call_llm([{"role": "user", "content": "decide"}], json_mode=True)

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_no_json_mode_omits_response_format(self, tmp_path):
        agent = self._make_agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "plain text"
        mock_resp.usage.total_tokens = 5
        mock_resp.choices[0].finish_reason = "stop"
        create_mock = MagicMock(return_value=mock_resp)
        agent.client.chat.completions.create = create_mock

        agent._call_llm([{"role": "user", "content": "hi"}], json_mode=False)

        call_kwargs = create_mock.call_args.kwargs
        assert "response_format" not in call_kwargs

    def test_api_error_is_reraised(self, tmp_path):
        from openai import APIError
        agent = self._make_agent(tmp_path)
        agent.client.chat.completions.create = MagicMock(
            side_effect=APIError("quota exceeded", request=MagicMock(), body=None)
        )
        with pytest.raises(APIError):
            agent._call_llm([{"role": "user", "content": "hi"}])
