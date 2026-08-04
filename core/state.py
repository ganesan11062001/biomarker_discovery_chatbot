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
  "proteomics"         — ProteomicsAnalysisSkill (intensity-only, dual-engine
                          Python + R/limma; canonical 2-sheet template)
  "transcriptomics"    — TranscriptomicsSkill   (planned)
  "metabolomics"       — MetabolomicsSkill      (planned)
  "lipidomics"         — LipidomicsSkill        (planned)
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


# ── Comparison helpers ────────────────────────────────────────────────────────

def make_comparison_id(group1_label: str, group2_label: str) -> str:
    """Canonical slug used as the key for an analysis entry in state['analyses'].

    Normalises whitespace and unsafe characters so the same comparison resolves
    to the same id regardless of how the user types it. Order matters
    (group1_vs_group2 ≠ group2_vs_group1) because fold-change sign depends on it.
    """
    def _slug(label: str) -> str:
        s = re.sub(r"\s+", "_", (label or "").strip())
        s = re.sub(r"[^A-Za-z0-9_]+", "", s)
        return s or "Group"
    return f"{_slug(group1_label)}_vs_{_slug(group2_label)}"


def _normalise_for_match(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace — used for fuzzy match."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def find_analysis(
    state: "BiomarkerState",
    target: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve which entry in state['analyses'] the user is asking about.

    Resolution order:
      1. None / empty target → most recent entry (last appended).
      2. Exact comparison_id match.
      3. Fuzzy match against "<g1> vs <g2>" — checks both label orders so
         "WT vs KO" finds the entry stored as KO_vs_WT too.
      4. Single-label match — if `target` contains only one group name and
         exactly one entry has that group as g1 or g2, return that entry.

    Returns None when no entries exist OR no fuzzy/exact match is found.
    """
    analyses = state.get("analyses") or []
    if not analyses:
        return None
    if not target or not target.strip():
        return analyses[-1]

    # Exact id match
    for a in analyses:
        if a.get("comparison_id") == target:
            return a

    norm_target = _normalise_for_match(target)

    # Fuzzy match against both label orders
    for a in analyses:
        g1 = _normalise_for_match(a.get("group1_label", ""))
        g2 = _normalise_for_match(a.get("group2_label", ""))
        if not g1 or not g2:
            continue
        forward  = f"{g1} vs {g2}"
        backward = f"{g2} vs {g1}"
        if forward in norm_target or backward in norm_target:
            return a
        # Also accept "g1 g2" / "g2 g1" without "vs"
        if g1 in norm_target and g2 in norm_target:
            return a

    # Single-label fallback — only when exactly one entry mentions the label
    candidates = []
    for a in analyses:
        g1 = _normalise_for_match(a.get("group1_label", ""))
        g2 = _normalise_for_match(a.get("group2_label", ""))
        if g1 and g1 in norm_target:
            candidates.append(a)
        elif g2 and g2 in norm_target:
            candidates.append(a)
    if len(candidates) == 1:
        return candidates[0]

    return None


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
    label_map:        Optional[Dict[str, str]]  # short_code → condition_name, read from the file
    sample_map:       Optional[Dict[str, Dict]]  # MaxQuant code → {client_id, strain, treatment, mouse_id}
    sample_to_group:  Optional[Dict[str, str]]   # Sample ID (column name) → Group name; 2-sheet canonical template
    column_group_labels: Optional[Dict[str, str]] # real column name → friendly group label (e.g. "SpC J" → "DMD Soleus")
    software:         Optional[str]               # detected vendor: 'MaxQuant', 'FragPipe', etc.
    is_pooled_design: Optional[bool]            # True when n=1 per group
    identifier_info:  Optional[Any]             # full parsed Identifier Info DataFrame (all mice)
    all_sheets:       Optional[Dict[str, Any]]  # every sheet parsed from the workbook

    # ── Analysis configuration ────────────────────────────────────────────────
    disease_program: Optional[str]        # free-form label provided by the user
    organism:        Optional[str]        # "human" | "mouse" | "rat"

    # Direct group assignment (used by all supervised omic skills)
    group1_samples: Optional[List[str]]   # column names assigned to group 1
    group2_samples: Optional[List[str]]   # column names assigned to group 2
    group1_label:   Optional[str]         # human-readable label (e.g. "Disease")
    group2_label:   Optional[str]         # human-readable label (e.g. "Control")
    analysis_mode:  Optional[str]         # "supervised" | "unsupervised"

    # Per-session analysis parameter overrides (set when user asks to change thresholds)
    # These override the global defaults from config/settings.py for THIS session.
    # Keys: adj_pval_cutoff, log2fc_cutoff, missing_threshold, top_n,
    #       test_method, is_paired, all_groups, tmt_batches,
    #       dose_levels, subject_map, clinical_outcome
    analysis_params: Optional[Dict[str, Any]]

    # Statistical test selection
    # "auto"    → auto-select (limma when n≤4, Welch otherwise)
    # "welch"   → Welch two-sample t-test (default for n≥5)
    # "limma"   → empirical Bayes moderated t-test (recommended for n<5)
    # "paired_t"→ paired t-test (before/after, matched pairs)
    # "anova"   → one-way ANOVA for >2 groups simultaneously + Tukey HSD post-hoc
    # "dose_response"       → linear trend test across ordered dose groups (requires dose_levels)
    # "repeated_measures"   → repeated-measures ANOVA / mixed-effects model across
    #                         time points, subjects blocked (requires all_groups + subject_map)
    # "linear_regression"   → OLS regression of protein vs. a continuous clinical outcome
    # "logistic_regression" → logistic regression vs. a binary clinical outcome (+ ROC AUC)
    # "cox_regression"      → Cox proportional-hazards regression vs. a survival outcome
    test_method: Optional[str]

    # Paired design — g1_samples[i] is the same biological unit as g2_samples[i]
    is_paired:    Optional[bool]

    # Multi-group ANOVA / dose-response / repeated-measures: {"GroupA": ["col1","col2"], ...}
    # Used when the user specifies >2 groups for simultaneous testing
    all_groups:   Optional[Dict[str, List[str]]]

    # TMT multi-batch structure for IRS normalisation
    # {"plex1": {"samples": ["ch1","ch2",...], "reference": "ref_col"}, ...}
    tmt_batches:  Optional[Dict[str, Any]]

    # Dose-response (test_method="dose_response"): group name → numeric dose
    # level, e.g. {"Vehicle": 0, "Low": 1, "Medium": 2, "High": 3}
    dose_levels: Optional[Dict[str, float]]

    # Time-course / repeated-measures (test_method="repeated_measures"):
    # sample column → subject/animal ID, so the same biological unit is
    # tracked across the time points defined in all_groups.
    subject_map: Optional[Dict[str, str]]

    # Clinical regression (test_method="linear_regression"|"logistic_regression"|
    # "cox_regression"): sample column → outcome value. Scalar for linear
    # (continuous) / logistic (0|1); {"time": float, "event": 0|1} dict for cox.
    clinical_outcome: Optional[Dict[str, Any]]

    # PTM / phosphoproteomics enrichment (case 9): when True, run_enrichment
    # adds kinase-substrate libraries (KEA, GEO kinase perturbations) on top
    # of the standard KEGG/GO/Reactome libraries. Does not affect DEA stats.
    ptm_analysis: Optional[bool]

    # Legacy fields — kept for backward compatibility with enrichment/viz agents
    sample_group_col: Optional[str]       # column containing group label
    contrast_groups:  Optional[List[str]] # [group1_name, group2_name]

    # ── QC ────────────────────────────────────────────────────────────────────
    qc_passed:  Optional[bool]
    qc_summary: Optional[Dict[str, Any]]

    # ── Analysis results — MOST RECENT analysis (legacy / convenience mirrors) ──
    # These mirror the *most recent* entry in `analyses`. They exist so older
    # code paths keep working without scanning the history list. Authoritative
    # source for multi-analysis sessions is `analyses` below.
    top_biomarkers:   Optional[List[Dict[str, Any]]] # ranked biomarker list
    n_significant:    Optional[int]
    excel_path:       Optional[str]                  # formatted Excel report
    analysis_summary: Optional[str]                  # LLM plain-language summary
    analysis_code:    Optional[str]                  # reproducible Python script
    biological_interpretation: Optional[str]         # DomainExpertAgent output
    last_query_code:   Optional[str]                 # most recent SQL/pandas snippet
    last_query_engine: Optional[str]                 # "sql" | "pandas"

    # Legacy alias (used by enrichment & visualization agents)
    top_proteins:    Optional[List[Dict]]   # mirrors top_biomarkers
    dea_result_path: Optional[str]          # legacy CSV path

    # Accumulated results across all comparisons run in this session.
    # Keyed by "Group1_vs_Group2"; each value holds the full biomarker list,
    # significance count, and excel path so cross-comparison questions (e.g.
    # overlap) can access every prior run, not just the last one.
    comparison_history: Optional[Dict[str, Any]]

    # ── Enrichment results (most-recent mirror) ───────────────────────────────
    enrichment_result_path: Optional[str]
    enrichment_scope:       Optional[str]   # "top_n" | "all" — user choice for gene set
    enrichment_top_n:       Optional[int]   # explicit N when scope is "top_n"
    pathways:               Optional[List[Dict]]
    # True once enrichment has actually executed, even if it found zero
    # significant pathways — distinguishes "ran, no hits" from "never run"
    # since an empty pathways list is falsy just like the unset default.
    enrichment_ran:         Optional[bool]

    # ── Visualization output (most-recent mirror) ─────────────────────────────
    plot_paths:  Optional[List[str]]
    report_path: Optional[str]

    # ── Multi-comparison history ──────────────────────────────────────────────
    # Each entry is one completed analysis. Filled by BiomarkerAgent on every
    # run (including each pair of `_run_all_comparisons`). Downstream agents
    # (visualization, enrichment, query) consult this list when the user asks
    # about a specific comparison ("plots for X vs Y", "top 10 in A vs B").
    # Entry shape:
    #   {
    #     "comparison_id":   "DMD_Quad_vs_BL6_Quad",   # canonical slug
    #     "group1_label":    "DMD_Quad",
    #     "group2_label":    "BL6_Quad",
    #     "group1_samples":  [...], "group2_samples": [...],
    #     "analysis_mode":   "supervised" | "unsupervised",
    #     "test_method":     "limma" | "welch" | ...,
    #     "top_biomarkers":  [...],
    #     "n_significant":   123,
    #     "excel_path":      "...",
    #     "plot_paths":      ["..."],
    #     "pathways":        [...],            # filled by EnrichmentAgent later
    #     "enrichment_result_path": "...",     # filled by EnrichmentAgent later
    #     "analysis_summary":  "...",
    #     "biological_interpretation": "...",
    #     "timestamp":       "ISO8601",
    #   }
    analyses: Optional[List[Dict[str, Any]]]

    # ── Status ────────────────────────────────────────────────────────────────
    status:        Optional[str]
    error_message: Optional[str]
