"""
agents/enrichment_agent.py
Knowledge Layer — pathway enrichment on biomarker results.

Reads from the new state fields (top_biomarkers, excel_path) while also
falling back to legacy fields (top_proteins, dea_result_path) so it works
whether the analysis was run with the new or old pipeline.

Backend: clusterProfiler (R) via PathwaySkill.
"""
from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from skills.run_enrichment import PathwaySkill

settings = get_settings()
logger = logging.getLogger(__name__)


class EnrichmentAgent(BaseAgent):
    """
    Runs KEGG and GO pathway enrichment on significant biomarkers.

    Requires a prior biomarker analysis run (BiomarkerAgent must have
    populated ``top_biomarkers`` or ``top_proteins``).
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_enrichment,
            system_prompt_path="prompts/enrichment_agent.txt",
        )
        self.pathway_skill = PathwaySkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        # Accept both new and legacy protein lists
        protein_source = state.get("top_biomarkers") or state.get("top_proteins")

        if not protein_source:
            state["status"] = "error"
            state["error_message"] = (
                "No biomarker results found. Run the analysis first."
            )
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "No significant biomarkers found. "
                    "Please run the analysis before requesting pathway enrichment."
                ),
            })
            return state

        # Build protein list from whichever field is available
        protein_list = [p.get("protein") for p in protein_source if p.get("protein")]
        if not protein_list:
            state["status"] = "no_significant_proteins"
            state["messages"].append({
                "role": "assistant",
                "content": "No significant protein identifiers available for enrichment.",
            })
            return state

        # Use excel_path as a proxy for a results file; fall back to dea_result_path
        results_path = state.get("excel_path") or state.get("dea_result_path") or ""

        try:
            result = self.pathway_skill.execute(
                protein_list=protein_list,
                dea_result_path=results_path,
                organism=state.get("organism", "human"),
            )

            state["enrichment_result_path"] = result["enrichment_result_path"]
            state["pathways"]               = result["top_pathways"]
            state["status"]                 = "enrichment_complete"

            top5 = result["top_pathways"][:5]
            kegg_list = "\n".join(
                f"  {i+1}. {p['pathway']} "
                f"(p.adj={p.get('p_adjust', '?'):.3e}, "
                f"genes={p.get('gene_count', '?')})"
                for i, p in enumerate(top5)
            )
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Pathway enrichment complete.\n"
                    f"- KEGG pathways: {result.get('n_kegg_significant', 0)}\n"
                    f"- GO terms: {result.get('n_go_significant', 0)}\n\n"
                    f"Top pathways:\n{kegg_list}"
                ),
            })
            logger.info(
                "Enrichment complete | session=%s kegg=%d go=%d",
                state.get("session_id"),
                result.get("n_kegg_significant", 0),
                result.get("n_go_significant", 0),
            )

        except Exception as exc:
            logger.exception("Pathway enrichment failed: %s", exc)
            state["status"] = "error"
            state["error_message"] = f"Pathway enrichment failed: {exc}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Pathway enrichment failed: {exc}",
            })

        return state
