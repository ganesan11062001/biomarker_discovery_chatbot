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
from pathlib import Path
from typing import Any, Dict

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from core.tracing import get_biomarker_metadata
from skills.omics_registry import OmicsSkillRegistry
from skills.pooled_fold_change import PooledFoldChangeSkill
from skills.proteomics_analysis import ProteomicsAnalysisSkill
from skills.silac_analysis import SilacAnalysisSkill

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
        self._registry.register(PooledFoldChangeSkill())
        self._registry.register(SilacAnalysisSkill())
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

        # ── Route to the correct omic skill ──────────────────────────────────
        #
        # Routing rules (in priority order):
        #   1. omic_type already set in state (e.g. "proteomics_pooled") → honour it
        #   2. is_pooled_design flag set (label_map extracted from Identifier Info sheet,
        #      AND each group truly has a single sample) → proteomics_pooled
        #   3. group labels provided with ≥2 replicates each → proteomics (supervised)
        #   4. No group labels → proteomics (unsupervised CV ranking)
        #
        # Downgrade pooled→standard if both group samples are given AND each has
        # multiple replicates — the user explicitly assigned replicated groups,
        # so Welch t-test is more appropriate than a fold-change-only analysis.
        g1 = state.get("group1_samples") or []
        g2 = state.get("group2_samples") or []

        if (
            state.get("omic_type") == "proteomics_pooled"
            and len(g1) >= 2
            and len(g2) >= 2
        ):
            logger.info(
                "Downgrading pooled→proteomics: replicated groups detected "
                "(g1=%d, g2=%d samples).", len(g1), len(g2),
            )
            state["omic_type"]        = "proteomics"
            state["is_pooled_design"] = False

        if state.get("is_pooled_design") and not state.get("omic_type"):
            state["omic_type"] = "proteomics_pooled"

        omic_type = state.get("omic_type") or _DEFAULT_OMIC_TYPE

        # Validate omic type before doing any work
        if omic_type not in self._registry:
            available = self._registry.available()
            return self._error(
                state,
                f"Unsupported omic type: '{omic_type}'. Available: {available}",
                f"Omic type '{omic_type}' is not supported. Currently available: {available}.",
            )

        # Determine supervised vs unsupervised mode
        mode = "supervised" if (g1 and g2) else "unsupervised"
        state["analysis_mode"] = mode
        state["status"] = "analyzing"

        # Resolve overrides early — needed for the progress message and analysis
        _overrides = state.get("analysis_params") or {}

        # User-facing progress message — clearly names what is running and why
        if omic_type == "proteomics_pooled":
            mode_label = (
                "pooled log₂ fold-change analysis (n=1 per group — no replicates). "
                "All pairwise contrasts will be computed."
            )
        elif mode == "supervised":
            g1_lbl = state.get("group1_label", "Group 1")
            g2_lbl = state.get("group2_label", "Group 2")
            n1, n2 = len(g1), len(g2)
            _req_method = _overrides.get("test_method") or state.get("test_method") or "auto"
            _method_label = {
                "limma":    "limma moderated t-test (eBayes)",
                "paired_t": "paired t-test",
                "anova":    "one-way ANOVA",
                "welch":    "Welch t-test",
            }.get(_req_method, "auto-selected test (limma for n≤4, Welch for n≥5)")
            mode_label = (
                f"differential expression analysis — **{g1_lbl}** (n={n1}) vs "
                f"**{g2_lbl}** (n={n2}) — {_method_label}. "
                f"Pipeline: log₂ transform → median normalisation → "
                f"group-aware filter → half-min imputation → {_method_label} → BH FDR."
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
        _all_groups  = state.get("all_groups") or _overrides.get("all_groups")
        _tmt_batches = state.get("tmt_batches")

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
            # Pooled-design parameters (PooledFoldChangeSkill)
            raw_data_path=state.get("raw_data_path", ""),
            label_map=state.get("label_map"),
        )

        if result.get("error"):
            return self._error(
                state,
                result["error"],
                f"Analysis failed: {result['error']}",
            )

        # Persist results
        state["omic_type"]      = omic_type
        state["top_biomarkers"] = result["top_biomarkers"]
        state["top_proteins"]   = result["top_biomarkers"]  # legacy field
        state["n_significant"]  = result["n_significant"]
        state["excel_path"]     = result["excel_path"]
        state["qc_summary"]     = result["qc_summary"]
        state["qc_passed"]      = True
        state["status"]         = "analysis_complete"

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

        top5 = (result.get("top_biomarkers") or [])[:5]
        if omic_type == "proteomics_pooled":
            # Pooled: show per-contrast fold changes
            def _fc_str(b: Dict[str, Any]) -> str:
                parts = [
                    f"{k}={v}" for k, v in b.items()
                    if k not in ("rank", "protein", "rescue_score") and isinstance(v, float)
                ]
                return ", ".join(parts[:3]) or "n/a"

            top5_lines = "\n".join(
                f"  {b.get('rank','?')}. {b.get('protein','?')}  "
                f"{_fc_str(b)}  rescue={b.get('rescue_score','?')}"
                for b in top5
            )
        elif mode == "supervised":
            top5_lines = "\n".join(
                f"  {b.get('rank','?')}. {b.get('protein','?')}  "
                f"log2FC={b.get('log2_fold_change','?')},  "
                f"adj_p={b.get('adj_p_value','?')},  "
                f"sig={b.get('significance','?')}"
                for b in top5
            )
        else:
            top5_lines = "\n".join(
                f"  {b.get('rank','?')}. {b.get('protein','?')}  CV={b.get('cv_percent','?')}%"
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
        if omic_type == "proteomics_pooled":
            method_str = (
                "**Method:** Log₂ fold-change (pooled n=1 design — no replicates, no p-values). "
                "Pseudocount +1 applied before log₂ transform. "
                "All pairwise contrasts computed from the uploaded label map."
            )
        elif mode == "supervised":
            method_str = (
                f"**Method:** Welch two-sample t-test — {g1} vs {g2}. "
                f"Pipeline: log₂ transform → median normalisation → group-aware missing filter "
                f"→ half-min imputation → Welch t-test → Benjamini-Hochberg FDR. "
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

        lines = []
        for b in top5:
            if "log2_fold_change" in b:
                lines.append(
                    f"  {b.get('rank','?')}. **{b.get('protein','?')}** — "
                    f"log2FC={b.get('log2_fold_change','?')}, "
                    f"adj_p={b.get('adj_p_value','?')}, "
                    f"{b.get('significance','NS')}"
                )
            elif "rescue_score" in b:
                fc_parts = ", ".join(
                    f"{k}={v}" for k, v in b.items()
                    if k not in ("rank", "protein", "rescue_score") and isinstance(v, float)
                )
                lines.append(
                    f"  {b.get('rank','?')}. **{b.get('protein','?')}** — "
                    f"{fc_parts}  rescue={b.get('rescue_score','?')}"
                )
            else:
                lines.append(
                    f"  {b.get('rank','?')}. **{b.get('protein','?')}** — "
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
