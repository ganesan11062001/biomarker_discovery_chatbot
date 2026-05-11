"""
api/routes/chat.py
POST /chat/         – send a message; runs the LangGraph workflow.
                     Returns JSON by default; streams SSE when the
                     request carries  Accept: text/event-stream.
POST /chat/session  – create a new session.
"""
import json
import logging
import time
import uuid
from typing import Iterator, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.langgraph_workflow import get_workflow
from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Serialise a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _chunk_text(text: str, chunk: int = 80) -> Iterator[str]:
    """Yield successive `chunk`-sized slices of `text` for token streaming.

    We don't actually have token-level streaming from LangGraph (the workflow
    is synchronous), but chunking the final response gives the frontend a
    smooth typing animation instead of a single huge jump.
    """
    for i in range(0, len(text), chunk):
        yield text[i:i + chunk]


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    # Per-message analysis config overrides
    sample_group_col:  Optional[str]        = None
    contrast_groups:   Optional[List[str]]  = None
    disease_program:   Optional[str]        = None
    organism:          Optional[str]        = None
    # New: direct group assignment
    group1_samples:    Optional[List[str]]  = None
    group2_samples:    Optional[List[str]]  = None
    group1_label:      Optional[str]        = None
    group2_label:      Optional[str]        = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: Optional[str] = None
    status: str


class SessionCreateResponse(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _run_workflow_turn(request: ChatRequest) -> dict:
    """Shared body for SSE and JSON paths: runs LangGraph for one turn and
    returns ``{session_id, new_assistant_messages, intent, status, expired}``.

    `new_assistant_messages` is the list of NEW assistant messages produced
    during this turn (typically 1, but can be 2 when DomainExpertAgent fires).
    """
    try:
        state = SessionManager.get_session(request.session_id)
        expired = False
    except KeyError:
        # Session lost (e.g. server restart) — create a fresh one
        logger.warning("Session '%s' not found — creating replacement.", request.session_id)
        new_sid = SessionManager.create_session(
            disease_program=request.disease_program,
            organism=request.organism,
        )
        state = SessionManager.get_session(new_sid)
        state["session_id"] = new_sid
        expired_msg = (
            "⚠️ **Session expired** — the server was restarted and in-memory "
            "session data was lost. Please re-upload your data file to continue."
        )
        state["messages"].append({"role": "assistant", "content": expired_msg})
        SessionManager.update_session(new_sid, state)
        return {
            "session_id":              new_sid,
            "new_assistant_messages":  [{"role": "assistant", "content": expired_msg}],
            "intent":                  None,
            "status":                  "session_expired",
            "expired":                 True,
        }

    # Apply inline overrides
    overrides: dict = {"user_query": request.message}
    if request.sample_group_col: overrides["sample_group_col"]  = request.sample_group_col
    if request.contrast_groups:  overrides["contrast_groups"]   = request.contrast_groups
    if request.disease_program:  overrides["disease_program"]   = request.disease_program
    if request.organism:         overrides["organism"]          = request.organism
    if request.group1_samples:   overrides["group1_samples"]    = request.group1_samples
    if request.group2_samples:   overrides["group2_samples"]    = request.group2_samples
    if request.group1_label:     overrides["group1_label"]      = request.group1_label
    if request.group2_label:     overrides["group2_label"]      = request.group2_label
    state.update(overrides)

    # Decouple messages list — see comment in earlier revision: passing the
    # live list causes LangGraph's add_messages reducer to double every message.
    state["messages"] = list(state.get("messages") or [])
    n_msgs_before = len(state["messages"])

    workflow = get_workflow()
    try:
        updated_state = workflow.invoke(state)
    except Exception as exc:
        logger.exception("Workflow error session=%s: %s", request.session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {exc}",
        )

    # Persist only delta messages
    delta_msgs = (updated_state.get("messages") or [])[n_msgs_before:]
    state_to_store = {**updated_state, "messages": delta_msgs}
    SessionManager.update_session(updated_state["session_id"], state_to_store)

    new_assistant_messages: List[dict] = []
    for m in delta_msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            new_assistant_messages.append(m)
        elif hasattr(m, "content"):
            msg_type = getattr(m, "type", "") or type(m).__name__.lower()
            if "ai" in msg_type or "assistant" in msg_type:
                new_assistant_messages.append({
                    "role": "assistant",
                    "content": str(m.content),
                })

    return {
        "session_id":             updated_state["session_id"],
        "new_assistant_messages": new_assistant_messages,
        "intent":                 updated_state.get("intent"),
        "status":                 updated_state.get("status", "ok"),
        "expired":                False,
    }


def _stream_chat_response(request: ChatRequest) -> Iterator[str]:
    """Generator that yields SSE frames for a single chat turn."""
    try:
        result = _run_workflow_turn(request)
    except HTTPException as exc:
        yield _sse("error", {"error": exc.detail})
        yield _sse("done",  {"sessionId": request.session_id})
        return
    except Exception as exc:                              # safety net
        logger.exception("Chat stream crash: %s", exc)
        yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})
        yield _sse("done",  {"sessionId": request.session_id})
        return

    sid = result["session_id"]
    new_msgs: List[dict] = result["new_assistant_messages"] or [
        {"role": "assistant", "content": "Analysis complete."}
    ]

    # ── Token stream: chunk EACH assistant message, then emit message_complete
    for idx, msg in enumerate(new_msgs):
        content   = msg.get("content", "") or ""
        msg_id    = msg.get("id") or str(uuid.uuid4())
        # Add a thin visual separator between consecutive assistant messages
        if idx > 0:
            yield _sse("token", {"delta": "\n\n---\n\n"})
        for piece in _chunk_text(content, chunk=120):
            yield _sse("token", {"delta": piece})
            # No real latency — but a microscopic sleep helps the browser
            # render incrementally on fast loopback connections.
            time.sleep(0.005)
        completed = {
            "id":        msg_id,
            "role":      "assistant",
            "content":   content,
            "createdAt": msg.get("createdAt") or "",
            "hasPlots":  bool(msg.get("has_plots")),
            "skills":    msg.get("skills") or [],
        }
        yield _sse("message_complete", {"message": completed, "sessionId": sid})

    yield _sse("done", {"sessionId": sid, "intent": result["intent"],
                         "status": result["status"]})


@router.post("/")
def chat(request: ChatRequest, http_request: Request):
    """Send a chat message.

    Content negotiation:
      Accept: text/event-stream  →  Server-Sent Events stream (StreamEvent union)
      otherwise                  →  legacy JSON  ChatResponse
    """
    accept = (http_request.headers.get("accept") or "").lower()
    wants_sse = "text/event-stream" in accept

    if wants_sse:
        return StreamingResponse(
            _stream_chat_response(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control":            "no-cache, no-transform",
                "X-Accel-Buffering":        "no",      # disable nginx buffering
                "Connection":               "keep-alive",
            },
        )

    # ── Legacy JSON path ─────────────────────────────────────────────────────
    result = _run_workflow_turn(request)
    msgs = result["new_assistant_messages"]
    last_response = (msgs[-1]["content"] if msgs else "Analysis complete.")
    return ChatResponse(
        session_id=result["session_id"],
        response=last_response,
        intent=result["intent"],
        status=result["status"],
    )


@router.post(
    "/session",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    disease_program: Optional[str] = None,
    organism: Optional[str] = None,
):
    """Create a new analysis session.

    Both fields are optional and left blank by default — organism is
    auto-detected from protein-name OS= suffixes during ingestion, and
    disease_program is a free-form user-provided label.
    """
    session_id = SessionManager.create_session(
        disease_program=disease_program,
        organism=organism,
    )
    logger.info("Session created: %s (dp=%s, org=%s)", session_id, disease_program, organism)
    return SessionCreateResponse(session_id=session_id)
