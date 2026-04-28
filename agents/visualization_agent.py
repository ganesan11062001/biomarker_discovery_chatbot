"""
agents/visualization_agent.py
Output Layer — standard suite + on-demand user-requested plots.

Generates the full standard proteomics plot suite after analysis, or
a specific subset when the user requests individual plot types.
Every user-facing message is produced by an LLM call.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from skills.run_visualization import ProteomicsPlotSuite, PLOT_REGISTRY, resolve_plot_names

settings = get_settings()
logger   = logging.getLogger(__name__)

_PLOT_DETECTION_PROMPT = f"""\
You are a bioinformatics assistant.
The user has asked for specific plots or visualizations.
Available plot types: {list(PLOT_REGISTRY.keys())}

Read the user message and return a JSON list of requested plot names from the list above.
If the user wants ALL standard plots or is vague ("generate plots", "visualize results"), return [].
If specific plots are mentioned, return only those canonical names.

Examples:
  "give me a volcano plot" → ["volcano"]
  "I want PCA and heatmap" → ["pca", "heatmap"]
  "show me the boxplot and MA plot" → ["boxplot", "ma_plot"]
  "generate all plots" → []
  "show me the results" → []

OUTPUT: JSON array only. No markdown, no explanation.
"""


class VisualizationAgent(BaseAgent):
    """
    Generates the standard proteomics plot suite or user-requested specific plots.
    Every user-facing message is produced by an LLM call.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_visualization,
            system_prompt_path="prompts/visualization_agent.txt",
        )
        self.plot_suite = ProteomicsPlotSuite()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: BiomarkerState, requested_plots: Optional[List[str]] = None) -> BiomarkerState:
        protein_source = state.get("top_biomarkers") or state.get("top_proteins") or []

        if not protein_source and not state.get("excel_path"):
            msg = self._llm_no_data()
            state["status"]        = "error"
            state["error_message"] = "No analysis results to visualize."
            state["messages"].append({"role": "assistant", "content": msg})
            return state

        # ── Determine requested plots from user query if not passed in ────────
        if requested_plots is None:
            user_query = state.get("user_query", "")
            requested_plots = self._detect_requested_plots(user_query)

        # ── Build stem from existing result path ──────────────────────────────
        result_path = state.get("excel_path") or state.get("dea_result_path") or ""
        stem = Path(result_path).stem if result_path else "biomarker"

        # ── Run plot suite ────────────────────────────────────────────────────
        try:
            result = self.plot_suite.execute(
                top_proteins       = protein_source,
                analysis_mode      = state.get("analysis_mode", "supervised"),
                data_path          = state.get("data_path", ""),
                sample_columns     = state.get("sample_columns") or [],
                group1_samples     = state.get("group1_samples") or [],
                group2_samples     = state.get("group2_samples") or [],
                group1_label       = state.get("group1_label", "Group1"),
                group2_label       = state.get("group2_label", "Group2"),
                top_pathways       = state.get("pathways"),
                enrichment_result_path = state.get("enrichment_result_path", ""),
                contrast_groups    = [
                    state.get("group1_label", "Group1"),
                    state.get("group2_label", "Group2"),
                ],
                plot_types         = requested_plots or None,
                output_dir         = settings.output_dir,
                stem               = stem,
            )

            state["plot_paths"]  = result.get("plot_paths", [])
            state["report_path"] = result.get("report_path")
            state["status"]      = "report_ready"

            msg = self._llm_visualization_summary(result, state)
            state["messages"].append({"role": "assistant", "content": msg})

            logger.info(
                "Visualization complete | session=%s plots=%d",
                state.get("session_id"), len(result.get("plot_paths", [])),
            )

        except Exception as exc:
            logger.exception("Visualization failed: %s", exc)
            state["status"]        = "error"
            state["error_message"] = str(exc)
            state["messages"].append({
                "role": "assistant",
                "content": self._llm_error(str(exc)),
            })

        return state

    # ── Plot type detection ───────────────────────────────────────────────────

    def _detect_requested_plots(self, user_query: str) -> List[str]:
        """Ask LLM which specific plots the user wants. Returns [] for 'all'."""
        if not user_query.strip():
            return []
        messages = [
            {"role": "system", "content": _PLOT_DETECTION_PROMPT},
            {"role": "user",   "content": user_query},
        ]
        try:
            raw = self._call_llm(messages, max_tokens=100, temperature=0.0).strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            plots = json.loads(raw)
            if isinstance(plots, list):
                valid   = [p for p in plots if p in PLOT_REGISTRY]
                unknown = [p for p in plots if p not in PLOT_REGISTRY]
                if unknown:
                    logger.warning("Unknown plot names ignored: %s. Valid: %s", unknown, list(PLOT_REGISTRY))
                return valid
        except Exception as exc:
            logger.warning("Plot detection LLM failed: %s — generating standard suite.", exc)
        return []

    # ── LLM messages ──────────────────────────────────────────────────────────

    def _llm_visualization_summary(self, result: dict, state: BiomarkerState) -> str:
        protein_source = state.get("top_biomarkers") or state.get("top_proteins") or []
        top10    = protein_source[:10]
        pathways = (state.get("pathways") or [])[:5]
        g1       = state.get("group1_label", "Group1")
        g2       = state.get("group2_label", "Group2")
        plots    = result.get("plot_paths", [])
        plots_run = result.get("plots_run", [])

        protein_lines = self._format_protein_lines(top10)
        pathway_lines = "\n".join(
            f"  {i+1}. {p.get('pathway','')} (adj_p={float(p.get('p_adjust') or 1.0):.3e})"
            for i, p in enumerate(pathways)
        ) if pathways else "  No pathway data."

        ctx = (
            f"Visualization complete for {state.get('omic_type','proteomics')} analysis.\n"
            f"Comparison: {g1} vs {g2}\n"
            f"Analysis mode: {state.get('analysis_mode','supervised')}\n"
            f"Significant biomarkers: {state.get('n_significant', 0)}\n\n"
            f"Plots generated ({len(plots)}):\n"
            + "\n".join(f"  - {Path(p).name}" for p in plots) + "\n\n"
            f"Plot types: {plots_run}\n\n"
            f"Top 10 biomarkers:\n{protein_lines}\n\n"
            f"Top pathways:\n{pathway_lines}\n\n"
            "Describe what was visualized and what the researcher should look for in each plot."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": ctx},
        ]
        try:
            return self._call_llm(messages, max_tokens=450)
        except Exception as exc:
            logger.warning("Visualization LLM summary failed: %s", exc)
            return self._fallback_summary(plots)

    def _llm_no_data(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": (
                "The user wants plots but no analysis results exist. "
                "Tell them to run the analysis first."
            )},
        ]
        try:
            return self._call_llm(messages, max_tokens=120)
        except Exception:
            return "No analysis results found. Please run the analysis first, then generate plots."

    def _llm_error(self, error_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": f"Visualization failed: {error_text}. Tell the user."},
        ]
        try:
            return self._call_llm(messages, max_tokens=120)
        except Exception:
            return f"Visualization failed: {error_text}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_protein_lines(proteins: list) -> str:
        if not proteins:
            return "  No proteins."
        lines = []
        for p in proteins:
            if "log2_fold_change" in p:
                lines.append(f"  {p.get('rank','')}.{p.get('protein','')} "
                             f"log2FC={p.get('log2_fold_change','?')} adj_p={p.get('adj_p_value','?')}")
            elif "rescue_score" in p:
                lines.append(f"  {p.get('rank','')}.{p.get('protein','')} "
                             f"rescue={p.get('rescue_score','?')}")
            else:
                lines.append(f"  {p.get('rank','')}.{p.get('protein','')} "
                             f"CV={p.get('cv_percent','?')}%")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(plots: list) -> str:
        if not plots:
            return "No plots could be generated — check data availability."
        items = "\n".join(f"  - {Path(p).name}" for p in plots)
        return f"**{len(plots)} plots generated:**\n{items}"
