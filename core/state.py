"""
core/state.py

BiomarkerState — the single shared state dict that flows through the
LangGraph pipeline.

Multi-omic design
-----------------
``omic_type`` drives which analysis skill is dispatched by BiomarkerAgent.
All field names are intentionally generic (e.g. ``top_biomarkers`` rather
than ``top_proteins``) so they work across proteomics, transcriptomics,
metabolomics, and any future omic layer.

Supported omic types
--------------------
  "proteomics"         — ProteomicsAnalysisSkill    (implemented)
  "proteomics_pooled"  — PooledFoldChangeSkill      (implemented)
  "transcriptomics"    — TranscriptomicsSkill        (planned)
  "metabolomics"       — MetabolomicsSkill           (planned)
  "lipidomics"         — LipidomicsSkill             (planned)
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class BiomarkerState(TypedDict, total=False):
    # ── LangGraph message accumulation ───────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Session ───────────────────────────────────────────────────────────────
    session_id:    str
    user_query:    str
    intent:        Optional[str]
    active_agent:  Optional[str]

    # ── Omic type routing ─────────────────────────────────────────────────────
    # Set by the client or auto-detected during ingestion.
    # BiomarkerAgent uses this to dispatch to the correct analysis skill.
    omic_type: Optional[str]   # "proteomics" | "transcriptomics" | "metabolomics" …

    # ── Data ingestion ────────────────────────────────────────────────────────
    file_id:       Optional[str]
    raw_data_path: Optional[str]          # original uploaded file (before processing)
    data_path:     Optional[str]          # path to normalised CSV
    data_type:     Optional[str]          # "olink_npx" | "ms_lfq" | "generic"
    data_format:   Optional[str]          # "csv" | "excel"
    n_proteins:    Optional[int]          # generic alias for n_features
    n_samples:     Optional[int]

    # Detected column sets
    sample_columns:   Optional[List[str]] # numeric/intensity columns
    metadata_columns: Optional[List[str]] # non-numeric (group labels, IDs, etc.)

    # Pooled design support (multi-sheet MaxQuant / Olink Excel)
    label_map:        Optional[Dict[str, str]]  # e.g. {"A": "WT", "B": "mdx", …}
    is_pooled_design: Optional[bool]            # True when n=1 per group
    identifier_info:  Optional[Any]             # full parsed Identifier Info DataFrame (all mice)
    all_sheets:       Optional[Dict[str, Any]]  # every sheet parsed from the workbook

    # ── Analysis configuration ────────────────────────────────────────────────
    disease_program: Optional[str]        # e.g. "FA", "DMD"
    organism:        Optional[str]        # "human" | "mouse" | "rat"

    # Direct group assignment (used by all supervised omic skills)
    group1_samples: Optional[List[str]]   # column names assigned to group 1
    group2_samples: Optional[List[str]]   # column names assigned to group 2
    group1_label:   Optional[str]         # human-readable label (e.g. "Disease")
    group2_label:   Optional[str]         # human-readable label (e.g. "Control")
    analysis_mode:  Optional[str]         # "supervised" | "unsupervised"

    # Legacy fields — kept for backward compatibility with enrichment/viz agents
    sample_group_col: Optional[str]       # column containing group label
    contrast_groups:  Optional[List[str]] # [group1_name, group2_name]

    # ── QC ────────────────────────────────────────────────────────────────────
    qc_passed:  Optional[bool]
    qc_summary: Optional[Dict[str, Any]]

    # ── Analysis results (generic — omic-type agnostic) ───────────────────────
    top_biomarkers:   Optional[List[Dict[str, Any]]] # ranked biomarker list
    n_significant:    Optional[int]
    excel_path:       Optional[str]                  # formatted Excel report
    analysis_summary: Optional[str]                  # LLM plain-language summary

    # Legacy alias (used by enrichment & visualization agents)
    top_proteins:    Optional[List[Dict]]   # mirrors top_biomarkers
    dea_result_path: Optional[str]          # legacy CSV path

    # ── Enrichment results ────────────────────────────────────────────────────
    enrichment_result_path: Optional[str]
    pathways:               Optional[List[Dict]]

    # ── Visualization output ──────────────────────────────────────────────────
    plot_paths:  Optional[List[str]]
    report_path: Optional[str]

    # ── Status ────────────────────────────────────────────────────────────────
    status:        Optional[str]
    error_message: Optional[str]
