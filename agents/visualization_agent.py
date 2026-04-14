"""
agents/visualization_agent.py
Output Layer — plots, reports, and plain-language summaries.

Reads from new state fields (top_biomarkers, excel_path) and falls back
to legacy fields where needed, so it works regardless of pipeline version.
"""
from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from skills.run_visualization import ReportingSkill

settings = get_settings()
logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """
    Generates volcano plots, heatmaps, and a plain-language summary report.

    Requires prior biomarker analysis (excel_path or dea_result_path in state).
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_visualization,
            system_prompt_path="prompts/visualization_agent.txt",
        )
        self.reporting_skill = ReportingSkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        # Accept both new and legacy result paths
        results_path = state.get("excel_path") or state.get("dea_result_path")

        if not results_path:
            state["status"] = "error"
            state["error_message"] = "No analysis results to visualize."
            state["messages"].append({
                "role": "assistant",
                "content": "No analysis results found. Please run the analysis first.",
            })
            return state

        # Use new field preferentially, fall back to legacy
        protein_source = state.get("top_biomarkers") or state.get("top_proteins") or []
        contrast = state.get("contrast_groups") or [
            state.get("group1_label", "Group1"),
            state.get("group2_label", "Group2"),
        ]

        try:
            result = self.reporting_skill.execute(
                dea_result_path=results_path,
                enrichment_result_path=state.get("enrichment_result_path"),
                top_proteins=protein_source,
                top_pathways=state.get("pathways") or [],
                contrast_groups=contrast,
                disease_program=state.get("disease_program", ""),
            )

            state["plot_paths"] = result.get("plot_paths", [])
            state["report_path"] = result.get("report_path")
            state["status"] = "report_ready"

            plots_list = "\n".join(
                f"  - {p}" for p in result.get("plot_paths", [])
            )
            msg = f"Visualizations ready.\n\nGenerated:\n{plots_list}"

            plain = result.get("plain_language_summary", "")
            if not plain and protein_source:
                plain = self._generate_summary(state)

            if plain:
                msg += f"\n\n**Summary:**\n{plain}"

            state["messages"].append({"role": "assistant", "content": msg})
            logger.info(
                "Visualization complete | session=%s plots=%d",
                state.get("session_id"), len(result.get("plot_paths", [])),
            )

        except Exception as exc:
            logger.exception("Visualization failed: %s", exc)
            state["status"] = "error"
            state["error_message"] = f"Visualization failed: {exc}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Visualization failed: {exc}",
            })

        return state

    def _generate_summary(self, state: BiomarkerState) -> str:
        protein_source = state.get("top_biomarkers") or state.get("top_proteins") or []
        top10    = protein_source[:10]
        pathways = (state.get("pathways") or [])[:5]
        g1       = state.get("group1_label") or (state.get("contrast_groups") or ["Group1"])[0]
        g2       = state.get("group2_label") or (
            state.get("contrast_groups") or ["Group1", "Group2"]
        )[-1]

        context = (
            f"Disease program: {state.get('disease_program', 'unspecified')}\n"
            f"Omic type: {state.get('omic_type', 'proteomics')}\n"
            f"Comparison: {g1} vs {g2}\n"
            f"Top biomarkers: {[p.get('protein', p) for p in top10]}\n"
            f"Top pathways: {[p.get('pathway', p) for p in pathways]}\n"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                "Write a concise plain-language summary (3–5 sentences) "
                f"of these findings:\n{context}"
            )},
        ]
        try:
            return self._call_llm(messages, max_tokens=300)
        except Exception as exc:
            logger.warning("LLM summary failed in VisualizationAgent: %s", exc)
            return ""
