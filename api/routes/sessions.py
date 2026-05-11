"""
api/routes/sessions.py
GET  /sessions          – list every session known to SessionManager
GET  /sessions/{id}     – full session detail with messages + files

The new Next.js frontend uses these for the left-sidebar conversation list
and for restoring a conversation after a page reload.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    id:           str
    title:        str
    createdAt:    str
    lastActiveAt: str
    messageCount: int


class UploadedFileEntry(BaseModel):
    fileId:     str
    filename:   str
    size:       int = 0
    dataType:   Optional[str] = None
    software:   Optional[str] = None
    rowCount:   Optional[int] = None
    colCount:   Optional[int] = None
    uploadedAt: str = ""


class ChatMessageEntry(BaseModel):
    id:        str
    role:      str
    content:   str
    createdAt: str
    hasPlots:  bool = False


class SessionDetail(SessionSummary):
    messages: List[ChatMessageEntry]
    files:    List[UploadedFileEntry] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise(session_id: str, state: Dict[str, Any], last_accessed: float) -> SessionSummary:
    """Build a SessionSummary from the raw state dict."""
    from datetime import datetime
    messages = state.get("messages") or []
    # First user message → title (truncated)
    first_user = next(
        (m.get("content", "") for m in messages
         if isinstance(m, dict) and m.get("role") == "user"
         and not m.get("content", "").startswith("📎")),
        "",
    )
    title = (first_user.strip()[:60] + "…") if len(first_user) > 60 \
            else (first_user.strip() or "New conversation")
    last_iso = datetime.fromtimestamp(last_accessed).isoformat() if last_accessed else ""
    return SessionSummary(
        id            = session_id,
        title         = title,
        createdAt     = last_iso,
        lastActiveAt  = last_iso,
        messageCount  = sum(1 for m in messages
                            if isinstance(m, dict) and m.get("role") in ("user", "assistant")),
    )


def _coerce_messages(raw_messages: List[Any]) -> List[ChatMessageEntry]:
    """Normalise persisted message dicts (or LangChain message objects) for the API."""
    out: List[ChatMessageEntry] = []
    for i, m in enumerate(raw_messages or []):
        if isinstance(m, dict):
            role     = m.get("role") or "assistant"
            content  = str(m.get("content", ""))
            hasplots = bool(m.get("has_plots"))
            mid      = m.get("id") or f"persisted-{i}"
            created  = m.get("createdAt") or m.get("created_at") or ""
        elif hasattr(m, "content"):
            msg_type = getattr(m, "type", "") or type(m).__name__.lower()
            role     = "assistant" if ("ai" in msg_type or "assistant" in msg_type) else "user"
            content  = str(m.content)
            hasplots = False
            mid      = f"persisted-{i}"
            created  = ""
        else:
            continue
        if role not in ("user", "assistant"):
            continue
        out.append(ChatMessageEntry(
            id        = mid,
            role      = role,
            content   = content,
            createdAt = created,
            hasPlots  = hasplots,
        ))
    return out


def _coerce_files(state: Dict[str, Any]) -> List[UploadedFileEntry]:
    """Project the single-file-per-session model into the new list shape.

    Today the backend tracks exactly one upload per session in `file_id` +
    `data_path`. We expose it as a one-element list so the frontend can grow
    into multi-file support later without an API change.
    """
    file_id  = state.get("file_id")
    filename = state.get("raw_data_path") or state.get("data_path") or ""
    if not file_id and not filename:
        return []
    from pathlib import Path
    name_only = Path(str(filename)).name if filename else ""
    return [UploadedFileEntry(
        fileId     = str(file_id or "primary"),
        filename   = name_only,
        size       = 0,
        dataType   = state.get("data_type"),
        software   = state.get("software"),
        rowCount   = state.get("n_proteins"),
        colCount   = state.get("n_samples"),
        uploadedAt = "",
    )]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[SessionSummary])
def list_sessions():
    """Return every known session, sorted most-recently-active first."""
    summaries: List[SessionSummary] = []
    # Access SessionManager internals — single-process in-memory store is OK
    with SessionManager._lock:
        items = list(SessionManager._sessions.items())
    for sid, state in items:
        last = SessionManager._last_accessed.get(sid, 0.0)
        try:
            summaries.append(_summarise(sid, state, last))
        except Exception as exc:
            logger.debug("Skipping unsummarisable session %s: %s", sid, exc)
    summaries.sort(key=lambda s: s.lastActiveAt or "", reverse=True)
    return summaries


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str):
    """Return the full state of a session — messages + files metadata."""
    try:
        state = SessionManager.get_session(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id!r} not found")

    last     = SessionManager._last_accessed.get(session_id, 0.0)
    summary  = _summarise(session_id, state, last)
    messages = _coerce_messages(state.get("messages") or [])
    files    = _coerce_files(state)

    return SessionDetail(
        id            = summary.id,
        title         = summary.title,
        createdAt     = summary.createdAt,
        lastActiveAt  = summary.lastActiveAt,
        messageCount  = summary.messageCount,
        messages      = messages,
        files         = files,
    )
