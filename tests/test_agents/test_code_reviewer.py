"""
tests/test_agents/test_code_reviewer.py
Tests for CodeReviewerAgent and the review_and_revise loop.
All LLM calls are mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.code_reviewer import (
    CodeReviewerAgent,
    ExecutionRecord,
    ReviewResult,
    review_and_revise,
)


@pytest.fixture()
def reviewer():
    with patch("agents.base_agent._build_client", return_value=MagicMock()):
        return CodeReviewerAgent()


def _mock_review_response(reviewer, payload: dict):
    """Stub the reviewer's LLM call to return the given JSON."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(payload)
    mock_resp.usage.total_tokens = 30
    mock_resp.choices[0].finish_reason = "stop"
    reviewer.client.chat.completions.create = MagicMock(return_value=mock_resp)


class TestCodeReviewerAgent:

    def test_approved_simple_code(self, reviewer):
        _mock_review_response(reviewer, {
            "approved": True, "severity": "ok",
            "issues": [], "suggestion": "",
        })
        r = reviewer.review(
            user_question="how many sheets?",
            schema_context="sheets dict",
            candidate_code="answer = len(sheets)",
        )
        assert r.approved is True
        assert r.severity == "ok"
        assert r.issues == []

    def test_major_rejection(self, reviewer):
        _mock_review_response(reviewer, {
            "approved": False, "severity": "major",
            "issues": ["Column 'Spectral Count' doesn't exist; use 'A SpC'"],
            "suggestion": "Use df['A SpC'] instead",
        })
        r = reviewer.review(
            user_question="spectral count for X in sample A",
            schema_context="cols: A SpC, B SpC",
            candidate_code="answer = df.loc[df['Protein']=='X', 'Spectral Count'].iloc[0]",
        )
        assert r.approved is False
        assert r.severity == "major"
        assert r.needs_revision is True
        assert len(r.issues) == 1

    def test_malformed_llm_response_auto_approves(self, reviewer):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "NOT JSON"
        mock_resp.usage.total_tokens = 5
        mock_resp.choices[0].finish_reason = "stop"
        reviewer.client.chat.completions.create = MagicMock(return_value=mock_resp)
        r = reviewer.review("q", "s", "answer = 1")
        # Fail-open: malformed LLM response must not block the pipeline
        assert r.approved is True

    def test_llm_exception_auto_approves(self, reviewer):
        reviewer.client.chat.completions.create = MagicMock(
            side_effect=Exception("network error")
        )
        r = reviewer.review("q", "s", "answer = 1")
        assert r.approved is True


class TestReviewAndReviseLoop:

    def test_first_round_succeeds(self, reviewer):
        _mock_review_response(reviewer, {
            "approved": True, "severity": "ok", "issues": [], "suggestion": "",
        })
        calls = {"gen": 0, "exec": 0}

        def _gen(extra):
            calls["gen"] += 1
            return "answer = 42"

        def _exec(code):
            calls["exec"] += 1
            return 42, None

        rec = review_and_revise(
            generator=_gen, executor=_exec, reviewer=reviewer,
            user_question="q", schema_context="s",
        )
        assert rec.ok
        assert rec.result == 42
        assert rec.rounds_used == 1
        assert calls["gen"] == 1
        assert calls["exec"] == 1

    def test_runtime_error_triggers_retry(self, reviewer):
        # Reviewer always approves; executor fails first time, succeeds second.
        _mock_review_response(reviewer, {
            "approved": True, "severity": "ok", "issues": [], "suggestion": "",
        })
        exec_calls = {"n": 0}
        gen_calls  = {"n": 0, "extras": []}

        def _gen(extra):
            gen_calls["n"] += 1
            gen_calls["extras"].append(extra)
            return "answer = ..."

        def _exec(code):
            exec_calls["n"] += 1
            if exec_calls["n"] == 1:
                return None, "KeyError: 'A SpC'"
            return 7, None

        rec = review_and_revise(
            generator=_gen, executor=_exec, reviewer=reviewer,
            user_question="q", schema_context="s",
        )
        assert rec.ok
        assert rec.result == 7
        assert rec.rounds_used == 2
        # On the retry, the generator must have been given the error feedback
        assert gen_calls["extras"][1] is not None
        assert "KeyError" in gen_calls["extras"][1]

    def test_reviewer_rejection_triggers_revision(self, reviewer):
        # First review: rejected major. Second review: approved.
        responses = [
            json.dumps({"approved": False, "severity": "major",
                        "issues": ["wrong column"], "suggestion": "use 'A SpC'"}),
            json.dumps({"approved": True, "severity": "ok",
                        "issues": [], "suggestion": ""}),
        ]
        side_effects = []
        for r in responses:
            mock = MagicMock()
            mock.choices[0].message.content = r
            mock.usage.total_tokens = 20
            mock.choices[0].finish_reason = "stop"
            side_effects.append(mock)
        reviewer.client.chat.completions.create = MagicMock(
            side_effect=side_effects,
        )

        def _gen(extra):
            return "answer = 42"

        def _exec(code):
            return 42, None

        rec = review_and_revise(
            generator=_gen, executor=_exec, reviewer=reviewer,
            user_question="q", schema_context="s",
        )
        assert rec.ok
        assert rec.rounds_used == 2

    def test_all_rounds_fail_returns_last_error(self, reviewer):
        _mock_review_response(reviewer, {
            "approved": True, "severity": "ok", "issues": [], "suggestion": "",
        })

        def _gen(extra):
            return "answer = ..."

        def _exec(code):
            return None, "ValueError: data missing"

        rec = review_and_revise(
            generator=_gen, executor=_exec, reviewer=reviewer,
            user_question="q", schema_context="s", max_rounds=2,
        )
        assert not rec.ok
        assert "ValueError" in rec.error
        assert rec.rounds_used == 3  # initial + 2 revisions
