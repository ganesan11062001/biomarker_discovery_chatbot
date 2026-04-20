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
from skills.omics_registry import OmicsSkillRegistry
from skills.pooled_fold_change import PooledFoldChangeSkill
from skills.proteomics_analysis import ProteomicsAnalysisSkill

settings = get_settings()
logger = logging.getLogger(__name__)

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
        # Future registrations (uncomment when implemented):
        # self._registry.register(TranscriptomicsSkill())
        # self._registry.register(MetabolomicsSkill())
        # self._registry.register(LipidomicsSkill())
        logger.info("BiomarkerAgent ready. Available omic types: %s", self._registry.available())

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: BiomarkerState) -> BiomarkerState:
        if not state.get("data_path"):
            return self._error(
                state,
                "No data loaded. Please upload a file first.",
                "No data found. Please upload your file before running analysis.",
            )

        # Auto-detect pooled design: ingestion agent sets omic_type to
        # "proteomics_pooled" when it finds a label map in the Excel file.
        # Also honour an explicit is_pooled_design flag as a fallback.
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
        g1 = state.get("group1_samples") or []
        g2 = state.get("group2_samples") or []
        mode = "supervised" if (g1 and g2) else "unsupervised"
        state["analysis_mode"] = mode
        state["status"] = "analyzing"

        if omic_type == "proteomics_pooled":
            mode_label = "pooled fold-change"
        elif mode == "supervised":
            mode_label = "differential expression"
        else:
            mode_label = "unsupervised CV"
        state["messages"].append({
            "role": "assistant",
            "content": f"Running {mode_label} analysis ({omic_type}) — please wait…",
        })

        # Dispatch to the registered skill
        skill = self._registry.get(omic_type)
        # For pooled designs use the original raw file name; for processed CSV
        # use its stem so output files are named sensibly.
        raw_path = state.get("raw_data_path") or state.get("data_path", "analysis")
        file_name = Path(raw_path).stem

        result = skill.execute(
            # Standard parameters (used by ProteomicsAnalysisSkill)
            data_path=state.get("data_path", ""),
            sample_columns=state.get("sample_columns") or [],
            group1_samples=g1,
            group2_samples=g2,
            group1_label=state.get("group1_label") or "Group1",
            group2_label=state.get("group2_label") or "Group2",
            analysis_mode=mode,
            data_type=state.get("data_type") or "generic",
            adj_pval_cutoff=settings.adj_pval_cutoff,
            log2fc_cutoff=settings.log2fc_cutoff,
            missing_threshold=settings.missing_value_threshold,
            top_n=settings.top_n_biomarkers,
            output_dir=settings.output_dir,
            file_name=file_name,
            # Pooled-design parameters (used by PooledFoldChangeSkill)
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
        state["qc_passed"]      = True   # analysis succeeded → data passed QC
        state["status"]         = "analysis_complete"

        summary = self._build_summary(result, state)
        state["analysis_summary"] = summary
        state["messages"].append({"role": "assistant", "content": summary})

        logger.info(
            "Analysis complete | session=%s omic=%s significant=%d",
            state.get("session_id"), omic_type, result["n_significant"],
        )
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

        prompt = (
            f"{omic_type.capitalize()} biomarker analysis complete.\n\n"
            f"Mode: {mode}\n"
            + (f"Comparison: {g1} vs {g2}\n" if mode == "supervised" else "")
            + f"Features after QC: {qc.get('proteins_after_qc', 'N/A')}\n"
            f"Log2 transformed: {qc.get('log2_transformed', False)}\n"
            f"Significant biomarkers: {result.get('n_significant', 0)}\n\n"
            f"Top 5:\n{top5_lines}\n\n"
            "Write a concise (≤150 words) plain-language summary for a researcher:\n"
            "1. Key findings\n"
            "2. Most interesting biomarker(s)\n"
            "3. Any QC notes\n"
            "4. What the downloaded Excel file contains\n"
            "Use markdown formatting."
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
