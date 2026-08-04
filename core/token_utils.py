"""
core/token_utils.py

Token-aware helpers for building LLM prompts.

The existing conversation-history truncation in agents/learning_agent.py and
agents/chat_agent.py caps history by character count and/or message count.
Those heuristics bound the *worst case* reasonably well, but they don't know
the actual token cost of what they're sending — a handful of dense messages
can still add up to more tokens than intended even while "under the cap".

This module adds a real token count (via tiktoken, when installed) with a
graceful character-based fallback so nothing breaks in environments where
the dependency hasn't been installed yet. It's additive: existing callers
are unaffected unless they opt in to the new `max_tokens` parameters.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - exercised when tiktoken isn't installed
    _ENCODING = None

# Rough fallback used only when tiktoken is unavailable: ~4 chars/token is a
# standard approximation for English text with the cl100k tokenizer.
_CHARS_PER_TOKEN_FALLBACK = 4


def count_tokens(text: str) -> int:
    """Return the token count of `text`, falling back to a char-based estimate."""
    if not text:
        return 0
    text = str(text)
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK)


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens`, appending an ellipsis marker if cut."""
    if not text or max_tokens <= 0:
        return ""
    text = str(text)
    if _ENCODING is not None:
        tokens = _ENCODING.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _ENCODING.decode(tokens[:max_tokens]) + "…[truncated]"
    # Character fallback, using the same approximation as count_tokens.
    max_chars = max_tokens * _CHARS_PER_TOKEN_FALLBACK
    return text if len(text) <= max_chars else text[:max_chars] + "…[truncated]"


def window_messages_by_tokens(
    messages: List[Dict[str, Any]],
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Return the most-recent suffix of `messages` whose combined content token
    count fits within `max_tokens`. Always keeps at least the single most
    recent message, even if it alone exceeds the budget.

    Messages are expected to be plain {"role": ..., "content": ...} dicts, as
    already produced by the existing history-building helpers.
    """
    if not messages:
        return []

    kept: List[Dict[str, Any]] = []
    running_total = 0
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        msg_tokens = count_tokens(content)
        if kept and running_total + msg_tokens > max_tokens:
            break
        kept.append(msg)
        running_total += msg_tokens

    kept.reverse()
    return kept
