"""
agents/biomarker_agent.py
Analysis Layer — multi-omic biomarker discovery agent.

Architecture
------------
BiomarkerAgent uses an OmicsSkillRegistry to decouple itself from any
specific omic type.  On every run it:
  1. Reads ``state["omic_type"]`` (defaults to "proteomics")
  2. Looks up the registered skill for that omic type
  3. Executes the skill → standardised OmicsAnalysisResult
  4. Generates a plain-language LLM summary of the findings

Adding a new omic type
----------------------
1. Create a subclass of BaseOmicsSkill (e.g. TranscriptomicsSkill)
2. Set its ``omic_type`` property (e.g. "transcriptomics")
3. Register it in ``__init__``:
       self._registry.register(TranscriptomicsSkill())
No other changes are needed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.io_utils import read_csv_safe
from core.state import BiomarkerState, make_comparison_id
from core.tracing import get_biomarker_metadata
from skills.omics_registry import OmicsSkillRegistry
from skills.proteomics_analysis import ProteomicsAnalysisSkill

settings = get_settings()
logger = logging.getLogger(__name__)

# ── LangSmith @traceable (graceful no-op if not installed) ───────────────────
try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree as _get_run_tree
except ImportError:
    def _traceable(**_kw):          # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap
    def _get_run_tree():            # type: ignore[misc]
        return None

# Default omic type when none is set in state
_DEFAULT_OMIC_TYPE = "proteomics"


class BiomarkerAgent(BaseAgent):
    """
    Orchestrates omic-type analysis and summarises results with an LLM.

    The agent is intentionally omic-agnostic: all analysis logic lives in
    the registered skills.  BiomarkerAgent only handles routing, state
    management, and LLM-based interpretation.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_biomarker,
            system_prompt_path="prompts/biomarker_agent.txt",
        )
        # ── Register all available omic skills ────────────────────────────────
        # Add new omic types here as they are implemented.
        self._registry = OmicsSkillRegistry()
        self._registry.register(ProteomicsAnalysisSkill())
        # Future registrations (uncomment when implemented):
        # self._registry.register(TranscriptomicsSkill())
        # self._registry.register(MetabolomicsSkill())
        # self._registry.register(LipidomicsSkill())
        logger.info("BiomarkerAgent ready. Available omic types: %s", self._registry.available())

    # ── Main entry point ──────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="agent.biomarker",
                tags=["biomarker-discovery", "biomarker"])
    def run(self, state: BiomarkerState) -> BiomarkerState:
        rt = _get_run_tree()
        if rt is not None:
            try:
                rt.extra.setdefault("metadata", {}).update(get_biomarker_metadata(state))
            except Exception:
                pass
        if not state.get("data_path"):
            return self._error(
                state,
                "No data loaded. Please upload a file first.",
                "No data found. Please upload your file before running analysis.",
            )

        # ── Clear stale downstream artifacts from any previous analysis ───────
        # Every new analysis run replaces the active session context. Pathway
        # enrichment, plots, and biological interpretation from a prior run
        # are NOT valid for this run's biomarker list — so they're cleared.
        # If the user wants pathways/plots for this new analysis, they request
        # those next and the corresponding agent will repopulate.
        for stale_field in (
            "pathways",
            "enrichment_result_path",
            "plot_paths",
            "report_path",
            "biological_interpretation",
        ):
            state.pop(stale_field, None)

        # ── Route to the correct omic skill ──────────────────────────────────
        # The canonical 2-sheet template is intensity-only proteomics; legacy
        # SpC / SILAC / pooled-MaxQuant routing has been removed. The agent
        # always dispatches to ProteomicsAnalysisSkill for omic_type "proteomics".
        g1 = state.get("group1_samples") or []
        g2 = state.get("group2_samples") or []

        omic_type = state.get("omic_type") or _DEFAULT_OMIC_TYPE
        # Legacy sessions may still carry deprecated omic_type values from
        # earlier versions of the pipeline; coerce them to the canonical key.
        if omic_type in ("proteomics_pooled", "proteomics_silac"):
            state["omic_type"] = omic_type = "proteomics"

        # Validate omic type before doing any work
        if omic_type not in self._registry:
            available = self._registry.available()
            return self._error(
                state,
                f"Unsupported omic type: '{omic_type}'. Available: {available}",
                f"Omic type '{omic_type}' is not supported. Currently available: {available}.",
            )

        # Determine supervised vs unsupervised mode
        _overrides_early = state.get("analysis_params") or {}
        _method_hint = _overrides_early.get("test_method") or state.get("test_method") or "auto"
        # New scenario-coverage methods (dose-response, repeated-measures,
        # clinical regression) don't use group1/group2_samples — they key off
        # all_groups / clinical_outcome instead — but are still "supervised"
        # analyses (they produce p-values / effect sizes, not CV ranking).
        _new_method_flag = _method_hint in (
            "dose_response", "repeated_measures",
            "linear_regression", "logistic_regression", "cox_regression",
        )
        mode = "supervised" if (g1 and g2) or _new_method_flag else "unsupervised"
        state["analysis_mode"] = mode
        state["status"] = "analyzing"

        # Resolve overrides early — needed for the progress message and analysis
        _overrides = state.get("analysis_params") or {}

        # User-facing progress message — clearly names what is running and why
        if mode == "supervised":
            g1_lbl = state.get("group1_label", "Group 1")
            g2_lbl = state.get("group2_label", "Group 2")
            n1, n2 = len(g1), len(g2)
            _req_method = _overrides.get("test_method") or state.get("test_method") or "auto"
            # Mirror _resolve_test_method so the label matches what will actually run
            if _req_method == "auto":
                if min(n1, n2) <= 1:
                    _effective_label_method = "fold_change_only"
                elif min(n1, n2) <= 4:
                    _effective_label_method = "limma"
                else:
                    _effective_label_method = "welch"
            else:
                _effective_label_method = _req_method
            _method_label = {
                "limma":               "limma moderated t-test (eBayes)",
                "paired_t":            "paired t-test",
                "anova":               "one-way ANOVA + Tukey HSD post-hoc",
                "welch":               "Welch t-test",
                "fold_change_only":    "fold-change only (n=1, no statistical test)",
                "dose_response":       "dose-response linear trend test",
                "repeated_measures":   "repeated-measures ANOVA / mixed-effects model",
                "linear_regression":   "linear regression vs. clinical outcome",
                "logistic_regression": "logistic regression vs. clinical outcome (+ ROC AUC)",
                "cox_regression":      "Cox proportional-hazards regression",
            }.get(_effective_label_method, _effective_label_method)
            _pipeline_suffix = (
                "BH FDR." if _effective_label_method != "fold_change_only"
                else "ranked by |log₂FC|."
            )
            if _new_method_flag:
                mode_label = (
                    f"{_method_label} analysis. "
                    f"Pipeline: log₂ transform → median normalisation → "
                    f"group-aware filter → half-min imputation → {_method_label} → {_pipeline_suffix}"
                )
            else:
                mode_label = (
                    f"differential expression analysis — **{g1_lbl}** (n={n1}) vs "
                    f"**{g2_lbl}** (n={n2}) — {_method_label}. "
                    f"Pipeline: log₂ transform → median normalisation → "
                    f"group-aware filter → half-min imputation → {_method_label} → {_pipeline_suffix}"
                )
        else:
            mode_label = (
                "unsupervised variability analysis (no group labels). "
                "Proteins ranked by CV%, MAD, and IQR across all samples."
            )
        state["messages"].append({
            "role": "assistant",
            "content": f"Running {mode_label}",
        })

        # ── Resolve analysis parameters (session overrides > global settings) ──
        adj_pval_cutoff   = float(_overrides.get("adj_pval_cutoff",   settings.adj_pval_cutoff))
        log2fc_cutoff     = float(_overrides.get("log2fc_cutoff",     settings.log2fc_cutoff))
        missing_threshold = float(_overrides.get("missing_threshold", settings.missing_value_threshold))
        top_n             = int(  _overrides.get("top_n",             settings.top_n_biomarkers))

        if _overrides:
            logger.info(
                "Analysis params (session overrides active): adj_p=%.3f log2fc=%.2f "
                "missing=%.2f top_n=%d",
                adj_pval_cutoff, log2fc_cutoff, missing_threshold, top_n,
            )

        # Dispatch to the registered skill
        skill = self._registry.get(omic_type)
        raw_path  = state.get("raw_data_path") or state.get("data_path", "analysis")
        file_name = Path(raw_path).stem

        # Resolve extended test parameters from session overrides + state
        _test_method = (
            _overrides.get("test_method")
            or state.get("test_method")
            or "auto"
        )
        _is_paired   = bool(state.get("is_paired") or _overrides.get("is_paired", False))
        _tmt_batches = state.get("tmt_batches")

        # Only pass all_groups when the user explicitly requested ANOVA across 3+
        # groups, or a group-based new-scenario method (dose-response /
        # repeated-measures). For pairwise run_analysis (group1/group2 set),
        # all_groups is inherited from ingestion-time label detection and must
        # NOT be forwarded — the skill's is_anova condition fires on any
        # all_groups with ≥2 entries, causing a spurious ANOVA attempt that
        # fails when n=1 per group.
        _group_based_method = _test_method in ("anova", "dose_response", "repeated_measures")
        _override_groups = _overrides.get("all_groups")
        if _group_based_method or _override_groups:
            _all_groups = _override_groups or state.get("all_groups")
        else:
            _all_groups = None

        # Dose-response (case 6): group name -> numeric dose level
        _dose_levels = _overrides.get("dose_levels") or state.get("dose_levels")
        # Time-course (case 7): sample column -> subject/animal ID
        _subject_map = _overrides.get("subject_map") or state.get("subject_map")
        # Clinical regression (case 10): sample column -> outcome value
        _clinical_outcome = _overrides.get("clinical_outcome") or state.get("clinical_outcome")

        result = skill.execute(
            # Standard parameters
            data_path=state.get("data_path", ""),
            sample_columns=state.get("sample_columns") or [],
            group1_samples=g1,
            group2_samples=g2,
            group1_label=state.get("group1_label") or "Group1",
            group2_label=state.get("group2_label") or "Group2",
            analysis_mode=mode,
            data_type=state.get("data_type") or "generic",
            adj_pval_cutoff=adj_pval_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            missing_threshold=missing_threshold,
            top_n=top_n,
            output_dir=settings.output_dir,
            file_name=file_name,
            # Extended test method (ProteomicsAnalysisSkill)
            test_method=_test_method,
            is_paired=_is_paired,
            all_groups=_all_groups,
            tmt_batches=_tmt_batches,
            # Pooled / no-replicates designs → fold-change-only ranking
            is_pooled_design=bool(state.get("is_pooled_design")),
            # New scenario-coverage parameters (cases 6, 7, 10)
            dose_levels=_dose_levels,
            subject_map=_subject_map,
            clinical_outcome=_clinical_outcome,
        )

        if result.get("error"):
            return self._error(
                state,
                result["error"],
                f"Analysis failed: {result['error']}",
            )

        # ── Dual-engine: run R/limma + intersect with Python results ──────────
        # Only meaningful for supervised two-group comparisons (Welch / limma
        # output, which expose log2_fold_change). ANOVA results use max_log2fc
        # and unsupervised uses cv_percent — neither is compatible.
        # Any R failure is non-fatal: the Python result is kept.
        dual = None
        py_top = result.get("top_biomarkers") or []
        has_log2fc = bool(py_top) and ("log2_fold_change" in py_top[0])
        has_adj_p  = bool(py_top) and ("adj_p_value"      in py_top[0])
        if (
            mode == "supervised"
            and len(g1) >= 2 and len(g2) >= 2
            and has_log2fc and has_adj_p          # need both for dual-engine intersection
        ):
            dual = self._run_dual_engine(
                state=state,
                python_top=py_top,
                g1=g1, g2=g2,
                g1_label=state.get("group1_label") or "Group1",
                g2_label=state.get("group2_label") or "Group2",
                adj_pval_cutoff=adj_pval_cutoff,
                log2fc_cutoff=log2fc_cutoff,
            )

        # Persist results — prefer dual-engine intersection when available
        state["omic_type"]      = omic_type
        if dual and not dual["intersected"].empty:
            top_records = dual["intersected"].head(top_n).to_dict("records")
            state["top_biomarkers"] = top_records
            state["top_proteins"]   = top_records
            state["n_significant"]  = int(dual["qc"]["intersection"])
            state["dual_engine_qc"] = dual["qc"]
            state["r_results_path"] = dual["r_csv"]
        else:
            state["top_biomarkers"] = result.get("top_biomarkers") or []
            state["top_proteins"]   = state["top_biomarkers"]
            state["n_significant"]  = result.get("n_significant") or 0
        state["excel_path"]     = result.get("excel_path")
        state["qc_summary"]     = result.get("qc_summary") or {}
        state["qc_passed"]      = True
        state["status"]         = "analysis_complete"

        # Accumulate this comparison so cross-comparison queries (e.g. overlap)
        # can access every prior run's biomarker list, not just the last one.
        _cmp_key = (
            f"{state.get('group1_label') or 'G1'}"
            f"_vs_"
            f"{state.get('group2_label') or 'G2'}"
        )
        _hist = dict(state.get("comparison_history") or {})
        _hist[_cmp_key] = {
            "top_biomarkers": state["top_biomarkers"],
            "n_significant":  state["n_significant"],
            "excel_path":     state["excel_path"],
        }
        state["comparison_history"] = _hist

        # ── Plotly visualisation suite (interactive HTML + PNG) ───────────────
        # The Python helper internally skips per-plot failures, so it's safe
        # to invoke even when log2_fold_change is missing — volcano/MA will
        # be dropped but PCA/heatmap/boxplots still produced.
        if mode == "supervised":
            try:
                plot_suite = self._build_plots(
                    state=state, python_top=py_top,
                    dual=dual,
                    adj_pval_cutoff=adj_pval_cutoff,
                    log2fc_cutoff=log2fc_cutoff,
                )
                if plot_suite:
                    flat_paths = [
                        p for variants in plot_suite.values()
                        for p in variants.values() if p
                    ]
                    # Extend existing paths; never overwrite plots from prior
                    # agents (e.g. enrichment dotplot already in state).
                    existing_plots = list(state.get("plot_paths") or [])
                    merged = existing_plots + [p for p in flat_paths if p not in existing_plots]
                    state["plot_paths"] = merged
                    result["plot_paths"] = flat_paths
            except Exception as exc:
                logger.warning("Plot generation failed: %s", exc)

        if result.get("analysis_code"):
            state["analysis_code"] = result["analysis_code"]

        summary = self._build_summary(result, state)
        state["analysis_summary"] = summary

        # Code is stored in state but NOT shown unless user asks ("show me the code")
        state["messages"].append({
            "role":      "assistant",
            "content":   summary,
            "has_plots": bool(result.get("plot_paths")),
        })

        # ── Domain Expert biological interpretation pass ──────────────────────
        # Inspired by GenoMAS — a separate, focused LLM call grounded ONLY in
        # the computed biomarker list produces a tighter biological interpretation
        # than baking it into the analysis summary prompt.
        try:
            from agents.domain_expert import DomainExpertAgent
            if not hasattr(self, "_domain_expert"):
                self._domain_expert = DomainExpertAgent()
            interpretation = self._domain_expert.interpret(state)
            if interpretation:
                state["messages"].append({
                    "role": "assistant",
                    "content": interpretation,
                })
                state["biological_interpretation"] = interpretation
        except Exception as exc:
            logger.warning("Domain expert interpretation failed: %s", exc)

        # ── Snapshot this analysis into state["analyses"] history ──────────────
        # Each call to BiomarkerAgent.run() appends one entry so multi-comparison
        # sessions can recall a specific analysis later by comparison_id or
        # by fuzzy "g1 vs g2" match. The legacy single-valued state fields
        # (top_biomarkers, excel_path, plot_paths) keep mirroring the most
        # recent entry for backward-compat with older code paths.
        self._append_to_history(state, g1=g1, g2=g2, mode=mode,
                                test_method=_test_method, result=result)

        logger.info(
            "Analysis complete | session=%s omic=%s significant=%d",
            state.get("session_id"), omic_type, result["n_significant"],
        )

        # Update span with post-analysis values
        rt = _get_run_tree()
        if rt is not None:
            try:
                rt.extra.setdefault("metadata", {}).update(
                    get_biomarker_metadata(state, result)
                )
            except Exception:
                pass

        return state

    # ── Dual-engine helpers ───────────────────────────────────────────────────

    def _run_dual_engine(
        self,
        state: BiomarkerState,
        python_top,
        g1, g2, g1_label, g2_label,
        adj_pval_cutoff: float,
        log2fc_cutoff:   float,
    ) -> Dict[str, Any] | None:
        """Run R/limma on the cached expression CSV and intersect with Python."""
        try:
            from skills.r_analysis import RAnalysisSkill, RAnalysisError
            from skills.dual_engine import combine_engines
            import pandas as pd
        except Exception as exc:
            logger.warning("Dual-engine deps not importable: %s — skipping R", exc)
            return None

        r_skill = RAnalysisSkill(adj_pval_cutoff=adj_pval_cutoff,
                                  log2fc_cutoff=log2fc_cutoff)
        if not r_skill.is_available():
            logger.info("Rscript or limma not available — running Python-only.")
            return None

        # Load the CSV the Python skill consumed (already cleaned by ingestion)
        try:
            expr_df = read_csv_safe(state["data_path"], index_col=0)
            expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
        except Exception as exc:
            logger.warning("Could not reload expression CSV for R: %s", exc)
            return None

        try:
            r_res = r_skill.run(
                expression_df=expr_df,
                group1_samples=g1, group2_samples=g2,
                group1_label=g1_label, group2_label=g2_label,
            )
        except RAnalysisError as exc:
            logger.warning("R/limma failed (%s) — falling back to Python only", exc)
            return None

        # Python results: convert top_biomarkers list-of-dicts back to a DF
        py_df = pd.DataFrame(python_top)
        combined = combine_engines(
            python_results = py_df,
            r_results      = r_res.results_df,
            adj_pval_cutoff = adj_pval_cutoff,
            log2fc_cutoff   = log2fc_cutoff,
        )
        return {
            "intersected": combined.intersected_df,
            "python":      combined.python_df,
            "r":           combined.r_df,
            "qc":          combined.qc,
            "r_csv":       r_res.csv_path,
            "expr_df":     expr_df,
        }

    def _build_plots(
        self,
        state: BiomarkerState,
        python_top,
        dual: Dict[str, Any] | None,
        adj_pval_cutoff: float,
        log2fc_cutoff:   float,
    ) -> Dict[str, Dict[str, str]]:
        """Build the Plotly suite using Accession Number as the primary label."""
        try:
            from skills.plotly_visuals import build_visualisation_suite
            import pandas as pd
        except Exception as exc:
            logger.warning("Plotly visuals not importable: %s", exc)
            return {}

        if dual is not None:
            expr_df = dual["expr_df"]
            inter_df = dual["intersected"]
        else:
            try:
                expr_df = read_csv_safe(state["data_path"], index_col=0)
                expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
                inter_df = None
            except Exception as exc:
                logger.warning("Could not load expression CSV for plotting: %s", exc)
                return {}

        sample_to_group = state.get("sample_to_group") or {}
        if not sample_to_group:
            # Build from group labels in state
            g1 = state.get("group1_samples") or []
            g2 = state.get("group2_samples") or []
            sample_to_group = {
                **{s: state.get("group1_label", "Group1") for s in g1},
                **{s: state.get("group2_label", "Group2") for s in g2},
            }

        py_df = pd.DataFrame(python_top)
        if "accession" not in py_df.columns and "protein" in py_df.columns:
            py_df = py_df.rename(columns={"protein": "accession"})

        out_dir = str(Path(settings.output_dir) / "plots" /
                       (state.get("session_id") or "session"))

        suite = build_visualisation_suite(
            python_results   = py_df,
            expression_df    = expr_df,
            sample_to_group  = sample_to_group,
            intersected_df   = inter_df,
            out_dir          = out_dir,
            adj_pval_cutoff  = adj_pval_cutoff,
            log2fc_cutoff    = log2fc_cutoff,
        )
        return suite

    # ── History management ────────────────────────────────────────────────────

    def _append_to_history(
        self,
        state: BiomarkerState,
        g1: list,
        g2: list,
        mode: str,
        test_method: str,
        result: Dict[str, Any],
    ) -> None:
        """Append a frozen snapshot of this analysis to state["analyses"].

        If an entry with the same comparison_id already exists, it is replaced
        rather than duplicated — so re-running the same comparison with new
        thresholds updates in place instead of bloating the list.
        """
        g1_label = state.get("group1_label") or "Group1"
        g2_label = state.get("group2_label") or "Group2"
        cmp_id   = make_comparison_id(g1_label, g2_label)

        entry: Dict[str, Any] = {
            "comparison_id":   cmp_id,
            "group1_label":    g1_label,
            "group2_label":    g2_label,
            "group1_samples":  list(g1),
            "group2_samples":  list(g2),
            "analysis_mode":   mode,
            "test_method":     test_method,
            "top_biomarkers":  list(state.get("top_biomarkers") or []),
            "n_significant":   int(state.get("n_significant") or 0),
            "excel_path":      state.get("excel_path"),
            "plot_paths":      state.get("plot_paths") or {},
            "qc_summary":      result.get("qc_summary") or {},
            "analysis_summary": state.get("analysis_summary"),
            "biological_interpretation": state.get("biological_interpretation"),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

        history = list(state.get("analyses") or [])
        history = [a for a in history if a.get("comparison_id") != cmp_id]
        history.append(entry)
        state["analyses"] = history

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _error(
        self, state: BiomarkerState, log_msg: str, user_msg: str
    ) -> BiomarkerState:
        logger.warning("BiomarkerAgent error: %s", log_msg)
        state["status"] = "error"
        state["error_message"] = log_msg
        state["messages"].append({"role": "assistant", "content": user_msg})
        return state

    # ── LLM summary generation ────────────────────────────────────────────────

    @_traceable(run_type="chain", name="biomarker.summary",
                tags=["biomarker-discovery", "biomarker"])
    def _build_summary(self, result: Dict[str, Any], state: BiomarkerState) -> str:
        """Ask the LLM to write a plain-language summary of the analysis."""
        mode      = state.get("analysis_mode", "supervised")
        omic_type = state.get("omic_type", "proteomics")
        g1        = state.get("group1_label", "Group 1")
        g2        = state.get("group2_label", "Group 2")
        qc        = result.get("qc_summary") or {}

        def _id(b: Dict[str, Any]) -> str:
            acc  = b.get("protein", "?")
            gene = (b.get("gene_name") or "").strip()
            return f"{gene} ({acc})" if gene and gene != "?" else acc

        top5 = (result.get("top_biomarkers") or [])[:5]
        if mode == "supervised":
            top5_lines = "\n".join(
                f"  {b.get('rank','?')}. {_id(b)}  "
                f"log2FC={b.get('log2_fold_change','?')},  "
                f"adj_p={b.get('adj_p_value','n/a')},  "
                f"sig={b.get('significance','?')}"
                for b in top5
            )
        else:
            top5_lines = "\n".join(
                f"  {b.get('rank','?')}. {_id(b)}  CV={b.get('cv_percent','?')}%"
                for b in top5
            )

        # Use the actual parameters that were passed to the skill (including overrides)
        _overrides    = state.get("analysis_params") or {}
        _adj_pval     = float(_overrides.get("adj_pval_cutoff",   settings.adj_pval_cutoff))
        _log2fc       = float(_overrides.get("log2fc_cutoff",     settings.log2fc_cutoff))
        _overrides_note = (
            " *(custom thresholds set by user)*" if _overrides else ""
        )

        # Describe the exact statistical method used
        _actual_method = qc.get("test_method", "welch")
        if mode == "supervised":
            _pipeline_steps = "log₂ transform → median normalisation → group-aware missing filter → half-min imputation"
            if _actual_method == "fold_change_only":
                method_str = (
                    f"**Method:** Fold-change only (no statistical test) — {g1} vs {g2}. "
                    f"n=1 per group: statistical testing is invalid with a single observation. "
                    f"Pipeline: {_pipeline_steps} → log₂ fold-change ranking. "
                    f"Proteins ranked by |log₂FC| ≥ {_log2fc}{_overrides_note}. "
                    f"No p-values or FDR correction are reported."
                )
            elif _actual_method == "limma":
                method_str = (
                    f"**Method:** limma moderated t-test (eBayes empirical Bayes) — {g1} vs {g2}. "
                    f"Pipeline: {_pipeline_steps} → limma eBayes → Benjamini-Hochberg FDR. "
                    f"Significance thresholds: adj. p < {_adj_pval}{_overrides_note}, "
                    f"|log₂FC| ≥ {_log2fc}{_overrides_note}. "
                    f"Effect size: Cohen's d (pooled SD)."
                )
            elif _actual_method == "paired_t":
                method_str = (
                    f"**Method:** Paired t-test — {g1} vs {g2}. "
                    f"Pipeline: {_pipeline_steps} → paired t-test → Benjamini-Hochberg FDR. "
                    f"Significance thresholds: adj. p < {_adj_pval}{_overrides_note}, "
                    f"|log₂FC| ≥ {_log2fc}{_overrides_note}."
                )
            elif _actual_method == "anova":
                method_str = (
                    f"**Method:** One-way ANOVA across groups. "
                    f"Pipeline: {_pipeline_steps} → one-way ANOVA → Benjamini-Hochberg FDR. "
                    f"Significance threshold: adj. p < {_adj_pval}{_overrides_note}."
                )
            else:  # welch (default)
                method_str = (
                    f"**Method:** Welch two-sample t-test — {g1} vs {g2}. "
                    f"Pipeline: {_pipeline_steps} → Welch t-test → Benjamini-Hochberg FDR. "
                    f"Significance thresholds: adj. p < {_adj_pval}{_overrides_note}, "
                    f"|log₂FC| ≥ {_log2fc}{_overrides_note}. "
                    f"Effect size: Cohen's d (pooled SD)."
                )
        else:
            method_str = (
                "**Method:** Unsupervised variability ranking (no group labels). "
                "Proteins ranked by CV%, MAD, and IQR across all samples. "
                "Pipeline: log₂ transform → median normalisation → half-min imputation → ranking."
            )

        prompt = (
            f"{omic_type.capitalize()} biomarker analysis complete.\n\n"
            f"Mode: {mode}\n"
            + (f"Comparison: {g1} vs {g2}\n" if mode == "supervised" else "")
            + f"Proteins input: {qc.get('proteins_input', 'N/A')}\n"
            f"Contaminants removed: {qc.get('contaminants_removed', 0)}\n"
            f"Proteins after QC: {qc.get('proteins_after_qc', 'N/A')}\n"
            f"Log2 transformed: {qc.get('log2_transformed', False)}\n"
            f"Normalised: {qc.get('normalised', False)} ({qc.get('normalisation_method','none')})\n"
            f"Significant biomarkers: {result.get('n_significant', 0)}\n\n"
            f"Top 5:\n{top5_lines}\n\n"
            f"Statistical method: {method_str}\n\n"
            "Write a concise (≤200 words) plain-language summary for a researcher:\n"
            "1. Which statistical method was used (be specific about the pipeline steps)\n"
            "2. Key findings and comparison made\n"
            "3. Most interesting biomarker(s) with their values\n"
            "4. QC notes (proteins removed, normalisation applied)\n"
            "5. What the downloaded Excel file contains\n"
            "Always include the method name in the summary. Use markdown formatting."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]
        try:
            return self._call_llm(messages, max_tokens=400)
        except Exception as exc:
            logger.warning("LLM summary failed: %s — using fallback.", exc)
            return self._fallback_summary(result, state)

    def _fallback_summary(self, result: Dict[str, Any], state: BiomarkerState) -> str:
        mode      = state.get("analysis_mode", "supervised")
        omic_type = state.get("omic_type", "proteomics").capitalize()
        qc        = result.get("qc_summary") or {}
        top5      = (result.get("top_biomarkers") or [])[:5]

        def _id(b: Dict[str, Any]) -> str:
            acc  = b.get("protein", "?")
            gene = (b.get("gene_name") or "").strip()
            return f"{gene} ({acc})" if gene and gene != "?" else acc

        lines = []
        for b in top5:
            if "log2_fold_change" in b:
                lines.append(
                    f"  {b.get('rank','?')}. **{_id(b)}** — "
                    f"log2FC={b.get('log2_fold_change','?')}, "
                    f"adj_p={b.get('adj_p_value','n/a')}, "
                    f"{b.get('significance','NS')}"
                )
            else:
                lines.append(
                    f"  {b.get('rank','?')}. **{_id(b)}** — "
                    f"CV={b.get('cv_percent','?')}%"
                )

        g_note = (
            f" ({state.get('group1_label','G1')} vs {state.get('group2_label','G2')})"
            if mode == "supervised" else ""
        )

        return (
            f"### {omic_type} Analysis Complete{g_note}\n\n"
            f"- Features analysed: **{qc.get('proteins_after_qc', 'N/A')}**\n"
            f"- Significant biomarkers: **{result.get('n_significant', 0)}**\n"
            f"- Mode: {mode.capitalize()}\n\n"
            f"**Top 5 biomarkers:**\n" + "\n".join(lines) + "\n\n"
            "Download the Excel file for the full ranked list, QC metrics, and parameters."
        )
