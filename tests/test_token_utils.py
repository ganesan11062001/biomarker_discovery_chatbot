"""Tests for core/token_utils.py — token-aware context-window helpers."""
import core.token_utils as token_utils
from core.token_utils import count_tokens, truncate_text_to_tokens, window_messages_by_tokens


def test_count_tokens_empty():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_tokens_scales_with_length():
    short = count_tokens("hello")
    long = count_tokens("hello world, this is a much longer sentence with many more words")
    assert long > short


def test_truncate_text_to_tokens_no_op_when_under_budget():
    text = "short text"
    assert truncate_text_to_tokens(text, max_tokens=1000) == text


def test_truncate_text_to_tokens_cuts_and_marks():
    text = "word " * 500
    result = truncate_text_to_tokens(text, max_tokens=10)
    assert result.endswith("…[truncated]")
    assert count_tokens(result.replace("…[truncated]", "")) <= 10


def test_truncate_text_to_tokens_zero_budget():
    assert truncate_text_to_tokens("anything", max_tokens=0) == ""


def test_window_messages_by_tokens_keeps_all_when_under_budget():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = window_messages_by_tokens(messages, max_tokens=1000)
    assert result == messages


def test_window_messages_by_tokens_drops_oldest_first():
    messages = [
        {"role": "user", "content": "word " * 200},
        {"role": "assistant", "content": "word " * 200},
        {"role": "user", "content": "recent short message"},
    ]
    result = window_messages_by_tokens(messages, max_tokens=20)
    # Most recent message must survive; oldest should be dropped first.
    assert result[-1] == messages[-1]
    assert messages[0] not in result


def test_window_messages_by_tokens_always_keeps_latest_even_if_over_budget():
    messages = [{"role": "user", "content": "word " * 5000}]
    result = window_messages_by_tokens(messages, max_tokens=1)
    assert result == messages


def test_window_messages_by_tokens_empty_input():
    assert window_messages_by_tokens([], max_tokens=100) == []


def test_fallback_path_without_tiktoken(monkeypatch):
    monkeypatch.setattr(token_utils, "_ENCODING", None)
    assert count_tokens("hello world") > 0
    text = "word " * 100
    truncated = truncate_text_to_tokens(text, max_tokens=5)
    assert truncated.endswith("…[truncated]")
