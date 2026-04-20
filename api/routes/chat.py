"""
api/routes/chat.py
POST /chat/         – send a message; runs the LangGraph workflow
POST /chat/session  – create a new session
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from core.langgraph_workflow import get_workflow
from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)


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

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Process a user message through the full LangGraph pipeline."""
    try:
        state = SessionManager.get_session(request.session_id)
    except KeyError:
        # Session lost (e.g. server restart) — create a fresh one so the UI
        # gets a proper error message rather than a 404 crash.
        logger.warning("Session '%s' not found — creating replacement.", request.session_id)
        new_sid = SessionManager.create_session(
            disease_program=request.disease_program or "FA",
            organism=request.organism or "human",
        )
        state = SessionManager.get_session(new_sid)
        state["session_id"] = new_sid  # adopt the new id so response routes back correctly
        # Tell the UI to re-upload — the old session's data was in memory only.
        state["messages"].append({
            "role": "assistant",
            "content": (
                "⚠️ **Session expired** — the server was restarted and in-memory session data was lost. "
                "Please re-upload your data file to continue."
            ),
        })
        SessionManager.update_session(new_sid, state)
        return ChatResponse(
            session_id=new_sid,
            response=state["messages"][-1]["content"],
            intent=None,
            status="session_expired",
        )

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

    # Run LangGraph
    workflow = get_workflow()
    try:
        updated_state = workflow.invoke(state)
    except Exception as exc:
        logger.exception("Workflow error session=%s: %s", request.session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {exc}",
        )

    SessionManager.update_session(updated_state["session_id"], updated_state)

    # Extract last assistant message
    messages = updated_state.get("messages", [])
    last_response = "Analysis complete."
    for m in reversed(messages):
        if isinstance(m, dict):
            if m.get("role") == "assistant":
                last_response = m["content"]
                break
        else:
            msg_type = type(m).__name__
            if msg_type in ("AIMessage", "ChatMessage") or getattr(m, "type", "") == "ai":
                last_response = m.content
                break

    return ChatResponse(
        session_id=updated_state["session_id"],
        response=last_response,
        intent=updated_state.get("intent"),
        status=updated_state.get("status", "ok"),
    )


@router.post(
    "/session",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(disease_program: str = "FA", organism: str = "human"):
    """Create a new analysis session."""
    session_id = SessionManager.create_session(
        disease_program=disease_program,
        organism=organism,
    )
    logger.info("Session created: %s (dp=%s, org=%s)", session_id, disease_program, organism)
    return SessionCreateResponse(session_id=session_id)
