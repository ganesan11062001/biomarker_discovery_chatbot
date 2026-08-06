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
from typing import Any, Dict, Iterator, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.backend_service import BackendError
from core.backend_service import build_plot_artifacts as _build_plot_artifacts
from core.backend_service import create_session as _create_session
from core.backend_service import run_chat_turn

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


# ── Plot-artifact helpers ─────────────────────────────────────────────────────

# Image suffixes that the frontend can render via <img>. Anything else with a
# non-html extension is treated as a downloadable static asset; .html files
# go through the interactive iframe path.
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
    """Thin wrapper: runs LangGraph for one turn via the shared service layer.

    Translates the ChatRequest pydantic model into keyword args and converts
    BackendError into the FastAPI HTTPException the rest of this module expects.
    """
    try:
        return run_chat_turn(
            request.session_id,
            request.message,
            sample_group_col=request.sample_group_col,
            contrast_groups=request.contrast_groups,
            disease_program=request.disease_program,
            organism=request.organism,
            group1_samples=request.group1_samples,
            group2_samples=request.group2_samples,
            group1_label=request.group1_label,
            group2_label=request.group2_label,
        )
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


def _stream_chat_response(request: ChatRequest) -> Iterator[str]:
    """Generator that yields SSE frames for a single chat turn.

    Emits an immediate ``: connected`` comment frame BEFORE starting the
    (potentially long-running) LangGraph workflow. Without this prelude the
    browser sees an idle TCP connection for 10–30 seconds while the LLM /
    analysis runs, and intermediate proxies (or the browser's own retry
    logic during HMR / Fast Refresh in dev) can re-fire the request,
    causing the workflow to run repeatedly on the same session.
    """
    # 1. Immediate heartbeat — keeps the SSE channel open while the workflow runs
    yield ": connected\n\n"
    yield _sse("token", {"delta": ""})  # a token event keeps client parsers happy

    # 2. Synchronous workflow (blocks until LangGraph completes)
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

    # Build PlotArtifact dicts from any plots produced during this turn.
    # These travel TWO paths:
    #   1. Standalone "artifact" SSE events → appended to the session-level
    #      artifact panel as they arrive.
    #   2. Bundled into the last message_complete payload → so the frontend
    #      can render inline thumbnails attached to that specific message.
    plot_artifacts: List[Dict[str, Any]] = _build_plot_artifacts(
        sid, result.get("new_plot_paths") or [],
    )
    for art in plot_artifacts:
        yield _sse("artifact", {"sessionId": sid, "artifact": art})

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
        completed: Dict[str, Any] = {
            "id":        msg_id,
            "role":      "assistant",
            "content":   content,
            "createdAt": msg.get("createdAt") or "",
            "hasPlots":  bool(msg.get("has_plots")) or bool(plot_artifacts),
            "skills":    msg.get("skills") or [],
        }
        # Attach plot artifacts to the LAST assistant message of this turn so
        # MessageBubble can render the inline thumbnail grid right beneath
        # the text the user just read.
        if idx == len(new_msgs) - 1 and plot_artifacts:
            completed["artifacts"] = plot_artifacts
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
    last_response = (
        "\n\n---\n\n".join(m["content"] for m in msgs if m.get("content"))
        if msgs else "Analysis complete."
    )
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
    session_id = _create_session(disease_program=disease_program, organism=organism)
    logger.info("Session created: %s (dp=%s, org=%s)", session_id, disease_program, organism)
    return SessionCreateResponse(session_id=session_id)
