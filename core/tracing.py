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
    Compact session metadata attached to every LangSmith span as metadata.
    Avoids large lists (sample_columns, top_biomarkers) to keep payload small.
    """
    top_bm = state.get("top_biomarkers") or []
    return {
        # Session identity
        "session_id":      state.get("session_id", "unknown"),
        "disease_program": state.get("disease_program"),
        "organism":        state.get("organism"),
        # Data
        "data_type":       state.get("data_type"),
        "omic_type":       state.get("omic_type"),
        "n_proteins":      state.get("n_proteins"),
        "n_samples":       state.get("n_samples"),
        "is_pooled":       state.get("is_pooled_design", False),
        # Analysis
        "analysis_mode":   state.get("analysis_mode"),
        "group1_label":    state.get("group1_label"),
        "group2_label":    state.get("group2_label"),
        "n_significant":   state.get("n_significant"),
        "n_top_biomarkers": len(top_bm),
        # Enrichment & visualisation
        "n_pathways":      len(state.get("pathways") or []),
        "n_plots":         len(state.get("plot_paths") or []),
        "excel_ready":     bool(state.get("excel_path")),
        "enrichment_done": bool(state.get("pathways")),
        # Orchestration
        "status":          state.get("status"),
        "active_agent":    state.get("active_agent"),
        "intent":          state.get("intent"),
        # User query (truncated for safety)
        "user_query":      str(state.get("user_query", ""))[:120],
    }


def get_ingestion_metadata(state: dict, result: dict | None = None) -> dict:
    """Metadata for the ingestion span."""
    meta = {
        "session_id":   state.get("session_id", "unknown"),
        "data_format":  state.get("data_format"),
        "raw_data_path": state.get("raw_data_path"),
    }
    if result:
        meta.update({
            "n_proteins":      result.get("n_proteins"),
            "n_samples":       result.get("n_samples"),
            "data_type":       result.get("data_type"),
            "is_pooled":       result.get("is_pooled_design", False),
            "n_sample_cols":   len(result.get("sample_columns") or []),
            "n_metadata_cols": len(result.get("metadata_columns") or []),
            "label_map_keys":  list((result.get("label_map") or {}).keys()),
        })
    return meta


def get_biomarker_metadata(state: dict, result: dict | None = None) -> dict:
    """Metadata for the biomarker analysis span."""
    meta = {
        "session_id":    state.get("session_id", "unknown"),
        "omic_type":     state.get("omic_type"),
        "analysis_mode": state.get("analysis_mode"),
        "group1_label":  state.get("group1_label"),
        "group2_label":  state.get("group2_label"),
        "n_group1":      len(state.get("group1_samples") or []),
        "n_group2":      len(state.get("group2_samples") or []),
    }
    if result:
        meta.update({
            "n_significant":     result.get("n_significant"),
            "n_top_biomarkers":  len(result.get("top_biomarkers") or []),
            "excel_path":        result.get("excel_path"),
            "qc_proteins_after": (result.get("qc_summary") or {}).get("proteins_after_qc"),
            "log2_transformed":  (result.get("qc_summary") or {}).get("log2_transformed"),
            "skill_error":       result.get("error"),
        })
    return meta


def get_enrichment_metadata(state: dict, result: dict | None = None) -> dict:
    """Metadata for the enrichment span."""
    meta = {
        "session_id":  state.get("session_id", "unknown"),
        "organism":    state.get("organism"),
        "omic_type":   state.get("omic_type"),
        "n_input_proteins": len(
            [p for p in (state.get("top_biomarkers") or []) if p.get("protein")]
        ),
    }
    if result:
        meta.update({
            "n_kegg_significant": result.get("n_kegg_significant", 0),
            "n_go_significant":   result.get("n_go_significant", 0),
            "n_top_pathways":     len(result.get("top_pathways") or []),
            "genes_submitted":    result.get("genes_submitted"),
            "n_gene_symbols":     len(result.get("gene_symbols") or []),
        })
    return meta


def get_visualization_metadata(state: dict, result: dict | None = None) -> dict:
    """Metadata for the visualization span."""
    meta = {
        "session_id":    state.get("session_id", "unknown"),
        "analysis_mode": state.get("analysis_mode"),
        "omic_type":     state.get("omic_type"),
        "has_pathways":  bool(state.get("pathways")),
    }
    if result:
        meta.update({
            "n_plots":     len(result.get("plot_paths") or []),
            "plots_run":   result.get("plots_run", []),
            "report_path": result.get("report_path"),
        })
    return meta
