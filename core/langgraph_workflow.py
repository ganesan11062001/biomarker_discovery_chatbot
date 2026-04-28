"""
core/langgraph_workflow.py
LangGraph StateGraph — single-node architecture.

Flow
────
  START → learning_agent → END

The LearningAgent is the sole node. It uses LLM reasoning on every message
to decide what to do and calls specialist agents internally. No keyword
routing, no hard-coded conditional edges.

LangSmith tracing
─────────────────
configure_langsmith() is called here at module import time (before the first
agent run) so the LANGCHAIN_TRACING_V2 env-var is set before any LLM client
is created.  LangGraph then automatically creates a top-level trace for every
graph invocation, and all child LLM calls (via wrap_openai) + @traceable
spans nest under it.

Compiled once at startup; shared across all requests.
"""
import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from agents.learning_agent import LearningAgent
from config.settings import get_settings
from core.state import BiomarkerState
from core.tracing import configure_langsmith

logger   = logging.getLogger(__name__)
_settings = get_settings()

# Activate LangSmith tracing before the first graph invocation
configure_langsmith(
    api_key = _settings.langsmith_api_key or None,
    project = _settings.langsmith_project,
    enabled = _settings.langsmith_tracing,
)

_learning_agent = LearningAgent()


def _run_learning(state: BiomarkerState) -> BiomarkerState:
    logger.info("Node: learning_agent | session=%s", state.get("session_id"))

    # Isolate the messages list from LangGraph's channel reference BEFORE run().
    # learning_agent.run() appends to state["messages"] in-place.  If we pass the
    # channel's own list reference, the add_messages reducer will see the
    # already-mutated list as "existing" history and then append the delta on top,
    # doubling every message on every turn.  A fresh copy breaks that cycle:
    # the channel reference stays unchanged → reducer computes correctly.
    msgs_snapshot = list(state.get("messages") or [])
    n_before      = len(msgs_snapshot)
    working_state = {**state, "messages": msgs_snapshot}

    updated  = _learning_agent.run(working_state)
    all_msgs = updated.get("messages") or []

    # Return ONLY the newly appended messages so add_messages reducer
    # doesn't duplicate the existing history.
    result = dict(updated)
    result["messages"] = all_msgs[n_before:]
    return result


def _build_graph() -> StateGraph:
    builder = StateGraph(BiomarkerState)
    builder.add_node("learning_agent", _run_learning)
    builder.set_entry_point("learning_agent")
    builder.add_edge("learning_agent", END)
    return builder.compile()


@lru_cache(maxsize=1)
def get_workflow():
    """Return the compiled LangGraph workflow (singleton)."""
    logger.info("Compiling LangGraph workflow (LearningAgent single-node) …")
    return _build_graph()
