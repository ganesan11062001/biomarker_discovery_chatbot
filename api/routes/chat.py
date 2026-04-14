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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{request.session_id}' not found. POST /chat/session to create one.",
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
