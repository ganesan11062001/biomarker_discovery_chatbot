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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

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


# ── Plot-artifact helpers ─────────────────────────────────────────────────────

# Image suffixes that the frontend can render via <img>. Anything else with a
# non-html extension is treated as a downloadable static asset; .html files
# go through the interactive iframe path.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _flatten_plot_paths(plot_paths: Any) -> List[str]:
    """Flatten state['plot_paths'] into a flat list of file paths.

    Different agents store this field differently:
      • visualization_agent → list[str]
      • biomarker_agent._build_plots → dict[str, dict[str, str]]
        (category → {format: path}) and sometimes dict[str, str].
    We normalise to a deduplicated list while preserving order.
    """
    if not plot_paths:
        return []
    out: List[str] = []
    seen: set = set()

    def _push(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        if value in seen:
            return
        seen.add(value)
        out.append(value)

    if isinstance(plot_paths, list):
        for v in plot_paths:
            if isinstance(v, str):
                _push(v)
            elif isinstance(v, dict):
                for inner in v.values():
                    _push(inner)
    elif isinstance(plot_paths, dict):
        for v in plot_paths.values():
            if isinstance(v, str):
                _push(v)
            elif isinstance(v, dict):
                for inner in v.values():
                    _push(inner)
            elif isinstance(v, list):
                for inner in v:
                    _push(inner)
    return out


def _humanise_plot_title(path: str) -> str:
    """Derive a human-friendly title from a plot filename.

    "volcano_plot.png" → "Volcano Plot"
    "pca_scatter.html"  → "Pca Scatter"
    """
    stem = Path(path).stem
    # Strip the usual leading 'plot_' / trailing '_plot' tokens for a cleaner label
    stem = stem.replace("plot_", "").replace("_plot", "")
    return stem.replace("_", " ").strip().title() or "Plot"


def _build_plot_artifacts(session_id: str, paths: List[str]) -> List[Dict[str, Any]]:
    """Convert a list of plot file paths into PlotArtifact dicts ready for SSE.

    Sister files (`foo.png` + `foo.html`) are merged into one artifact so the
    interactive HTML and the static PNG land in the same panel entry.
    """
    if not paths:
        return []

    base_url = f"/results/{session_id}/file?path="
    by_stem: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for raw in paths:
        p = Path(raw)
        suffix = p.suffix.lower()
        # Normalise to outputs/-relative paths if needed (the /file endpoint
        # accepts absolute paths too, but relative keeps URLs shorter).
        try:
            rel = str(p.resolve().relative_to(Path("outputs").resolve()))
            file_param = f"outputs/{rel}"
        except (ValueError, FileNotFoundError, RuntimeError):
            file_param = str(p)

        stem_key = p.stem
        if stem_key not in by_stem:
            order.append(stem_key)
            by_stem[stem_key] = {
                "id":        f"plot-{session_id[:8]}-{stem_key}",
                "kind":      "plot",
                "title":     _humanise_plot_title(raw),
                "createdAt": now,
            }
        entry = by_stem[stem_key]
        full_url = base_url + quote(file_param, safe="/:?=&")
        if suffix == ".html":
            entry["htmlUrl"] = full_url
        elif suffix in _IMAGE_SUFFIXES:
            entry["imageUrl"] = full_url
        else:
            # Unknown extension — fall back to imageUrl so PlotArtifactView
            # still shows the link, even if the embedded preview is empty.
            entry.setdefault("imageUrl", full_url)

    return [by_stem[k] for k in order]


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
    # Snapshot the plot inventory BEFORE the turn so we can diff afterwards
    # and report only the plots produced during this turn (vs. carried over
    # from a previous comparison).
    plots_before = set(_flatten_plot_paths(state.get("plot_paths")))

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

    # Identify plots produced during this turn
    plots_after = _flatten_plot_paths(updated_state.get("plot_paths"))
    new_plot_paths = [p for p in plots_after if p not in plots_before]

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
        "new_plot_paths":         new_plot_paths,
    }


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
    session_id = SessionManager.create_session(
        disease_program=disease_program,
        organism=organism,
    )
    logger.info("Session created: %s (dp=%s, org=%s)", session_id, disease_program, organism)
    return SessionCreateResponse(session_id=session_id)
