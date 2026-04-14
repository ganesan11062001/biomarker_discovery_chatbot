"""
core/langgraph_workflow.py
LangGraph StateGraph wiring all pipeline agents.

Flow
────
  START
    │
    ▼
  chat_agent  ──── intent ────►  ingestion_agent  ──► END
                            ├──► biomarker_agent   ──► END
                            ├──► enrichment_agent  ──► END
                            ├──► visualization_agent► END
                            └──► END  (general chat)

Compiled once at startup; shared across all requests.
"""
import logging
from functools import lru_cache
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.biomarker_agent import BiomarkerAgent
from agents.chat_agent import ChatAgent
from agents.ingestion_agent import IngestionAgent
from core.state import BiomarkerState

logger = logging.getLogger(__name__)

# ── Agent singletons ──────────────────────────────────────────────────────────
_chat_agent      = ChatAgent()
_ingestion_agent = IngestionAgent()
_biomarker_agent = BiomarkerAgent()

# Optional agents — imported lazily so the platform starts even if their
# dependencies aren't installed yet
try:
    from agents.enrichment_agent import EnrichmentAgent
    _enrichment_agent = EnrichmentAgent()
except Exception:
    _enrichment_agent = None  # type: ignore

try:
    from agents.visualization_agent import VisualizationAgent
    _visualization_agent = VisualizationAgent()
except Exception:
    _visualization_agent = None  # type: ignore


# ── Node wrappers ─────────────────────────────────────────────────────────────

def _run_chat(state: BiomarkerState) -> BiomarkerState:
    logger.info("Node: chat_agent | session=%s", state.get("session_id"))
    return _chat_agent.run(state)


def _run_ingestion(state: BiomarkerState) -> BiomarkerState:
    logger.info("Node: ingestion_agent | session=%s", state.get("session_id"))
    return _ingestion_agent.run(state)


def _run_biomarker(state: BiomarkerState) -> BiomarkerState:
    logger.info("Node: biomarker_agent | session=%s", state.get("session_id"))
    return _biomarker_agent.run(state)


def _run_enrichment(state: BiomarkerState) -> BiomarkerState:
    logger.info("Node: enrichment_agent | session=%s", state.get("session_id"))
    if _enrichment_agent is not None:
        return _enrichment_agent.run(state)
    state["messages"].append({
        "role": "assistant",
        "content": "Pathway enrichment is not available in this deployment.",
    })
    return state


def _run_visualization(state: BiomarkerState) -> BiomarkerState:
    logger.info("Node: visualization_agent | session=%s", state.get("session_id"))
    if _visualization_agent is not None:
        return _visualization_agent.run(state)
    state["messages"].append({
        "role": "assistant",
        "content": "Visualization agent is not available in this deployment.",
    })
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

_SPECIALIST_NODES = {
    "ingestion_agent",
    "biomarker_agent",
    "enrichment_agent",
    "visualization_agent",
}


def _route_from_chat(
    state: BiomarkerState,
) -> Literal[
    "ingestion_agent",
    "biomarker_agent",
    "enrichment_agent",
    "visualization_agent",
    "__end__",
]:
    intent = state.get("intent", "chat_agent")
    if intent in _SPECIALIST_NODES:
        logger.debug("Routing to: %s", intent)
        return intent  # type: ignore[return-value]
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    builder = StateGraph(BiomarkerState)

    builder.add_node("chat_agent",           _run_chat)
    builder.add_node("ingestion_agent",      _run_ingestion)
    builder.add_node("biomarker_agent",      _run_biomarker)
    builder.add_node("enrichment_agent",     _run_enrichment)
    builder.add_node("visualization_agent",  _run_visualization)

    builder.set_entry_point("chat_agent")

    builder.add_conditional_edges(
        "chat_agent",
        _route_from_chat,
        {
            "ingestion_agent":     "ingestion_agent",
            "biomarker_agent":     "biomarker_agent",
            "enrichment_agent":    "enrichment_agent",
            "visualization_agent": "visualization_agent",
            END:                   END,
        },
    )

    for node in _SPECIALIST_NODES:
        builder.add_edge(node, END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_workflow():
    """Return the compiled LangGraph workflow (singleton)."""
    logger.info("Compiling LangGraph workflow …")
    return _build_graph()
