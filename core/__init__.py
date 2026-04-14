"""
core — shared infrastructure for the biomarker discovery platform.

Public API
----------
BiomarkerState     LangGraph TypedDict representing the full pipeline state.
SessionManager     Thread-safe in-memory session lifecycle manager.
get_workflow       Returns the compiled (singleton) LangGraph workflow.
"""
from core.session_manager import SessionManager
from core.state import BiomarkerState

__all__ = [
    "BiomarkerState",
    "SessionManager",
]
