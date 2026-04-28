"""
core/tracing.py
LangSmith observability setup for the multi-agent biomarker pipeline.

Call configure_langsmith() once at application startup (done automatically
by core/langgraph_workflow.py).  Every subsequent LLM call is then auto-traced
via wrap_openai (in base_agent.py) and every LangGraph node run appears as a
nested span under the same project.

Trace hierarchy produced:
  LangGraph run  (auto — set by LANGCHAIN_TRACING_V2)
    └─ learning_agent  (traceable)
          ├─ orchestrator.decision  (traceable)
          │      └─ [Azure OpenAI call]  (wrap_openai)
          ├─ [specialist agent LLM calls]  (wrap_openai, auto-nested)
          └─ orchestrator.answer  (traceable)
                 └─ [Azure OpenAI call]  (wrap_openai)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def configure_langsmith(
    api_key: Optional[str] = None,
    project: str = "biomarker-discovery",
    enabled: bool = True,
) -> bool:
    """
    Activate LangSmith tracing by setting the three env-vars that both
    LangGraph and langsmith.wrappers look for at call time.

    Returns True when tracing was successfully enabled, False otherwise.
    Must be called BEFORE the first agent run (ideally at module import).
    """
    if not enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled by config.")
        return False

    key = (
        api_key
        or os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
    )
    if not key:
        logger.warning(
            "LangSmith: no API key found (LANGSMITH_API_KEY not set) — "
            "tracing disabled.  Add it to .env to enable full observability."
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = key
    os.environ["LANGCHAIN_PROJECT"]     = project

    logger.info("LangSmith tracing enabled | project=%s", project)
    return True


def get_trace_metadata(state: dict) -> dict:
    """
    Extract a compact set of session fields to attach as LangSmith metadata.
    Keeps payload small — avoids serialising large lists like sample_columns.
    """
    return {
        "session_id":    state.get("session_id", "unknown"),
        "data_type":     state.get("data_type"),
        "omic_type":     state.get("omic_type"),
        "n_proteins":    state.get("n_proteins"),
        "n_samples":     state.get("n_samples"),
        "analysis_mode": state.get("analysis_mode"),
        "is_pooled":     state.get("is_pooled_design", False),
        "status":        state.get("status"),
        "active_agent":  state.get("active_agent"),
        "intent":        state.get("intent"),
    }
