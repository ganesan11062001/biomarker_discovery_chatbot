"""
agents/enrichment_agent.py
Knowledge Layer — pathway enrichment with LLM-generated interpretation.

Runs KEGG/GO enrichment via PathwaySkill then asks the LLM to produce
a biologically meaningful summary of what was found.
"""
from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from skills.run_enrichment import PathwaySkill

settings = get_settings()
logger   = logging.getLogger(__name__)


class EnrichmentAgent(BaseAgent):
    """
    Runs KEGG and GO pathway enrichment then uses the LLM to interpret
    the biological meaning of the top pathways in context.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_enrichment,
            system_prompt_path="prompts/enrichment_agent.txt",
        )
        self.pathway_skill = PathwaySkill()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: BiomarkerState) -> BiomarkerState:
        protein_source = state.get("top_biomarkers") or state.get("top_proteins")

        if not protein_source:
            msg = self._llm_no_data()
            state["status"]        = "error"
            state["error_message"] = "No biomarker results found."
            state["messages"].append({"role": "assistant", "content": msg})
            return state

        protein_list = [p.get("protein") for p in protein_source if p.get("protein")]
        if not protein_list:
            state["messages"].append({
                "role": "assistant",
                "content": "No protein identifiers available for enrichment.",
            })
            return state

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

            # LLM-generated biological interpretation
            msg = self._llm_enrichment_summary(result, state)
            state["messages"].append({"role": "assistant", "content": msg})

            logger.info(
                "Enrichment complete | session=%s kegg=%d go=%d",
                state.get("session_id"),
                result.get("n_kegg_significant", 0),
                result.get("n_go_significant", 0),
            )

        except Exception as exc:
            logger.exception("Pathway enrichment failed: %s", exc)
            state["status"]        = "error"
            state["error_message"] = str(exc)
            msg = self._llm_error(str(exc))
            state["messages"].append({"role": "assistant", "content": msg})

        return state

    # ── LLM summary ───────────────────────────────────────────────────────────

    def _llm_enrichment_summary(self, result: dict, state: BiomarkerState) -> str:
        top5 = result.get("top_pathways", [])[:5]
        pathway_lines = "\n".join(
            f"  {i+1}. {p['pathway']} | lib={p.get('library','')} | "
            f"adj_p={p.get('p_adjust',1):.3e} | genes={p.get('gene_count','?')} | "
            f"overlap={p.get('overlap','')}"
            for i, p in enumerate(top5)
        )

        g1 = state.get("group1_label", "Group1")
        g2 = state.get("group2_label", "Group2")
        omic = state.get("omic_type", "proteomics")

        ctx = (
            f"Enrichment analysis complete for {omic} comparison: {g1} vs {g2}\n\n"
            f"Statistics:\n"
            f"  - Proteins submitted: {result.get('genes_submitted', len(result.get('gene_symbols', [])))}\n"
            f"  - KEGG pathways significant: {result.get('n_kegg_significant', 0)}\n"
            f"  - GO terms significant: {result.get('n_go_significant', 0)}\n\n"
            f"Top enriched pathways:\n{pathway_lines}\n\n"
            f"Gene symbols used: {result.get('gene_symbols', [])[:15]}\n\n"
            "Provide a biologically meaningful interpretation of these enrichment results."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": ctx},
        ]
        try:
            return self._call_llm(messages, max_tokens=400)
        except Exception as exc:
            logger.warning("Enrichment LLM summary failed: %s — using fallback.", exc)
            return self._fallback_summary(result)

    def _llm_no_data(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": (
                "The user requested pathway enrichment but no biomarker results exist yet. "
                "Tell them they need to run biomarker analysis first."
            )},
        ]
        try:
            return self._call_llm(messages, max_tokens=150)
        except Exception:
            return "No biomarker results found. Please run the analysis first, then request enrichment."

    def _llm_error(self, error_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": f"Enrichment failed with error: {error_text}. Tell the user what went wrong."},
        ]
        try:
            return self._call_llm(messages, max_tokens=150)
        except Exception:
            return f"Pathway enrichment failed: {error_text}"

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_summary(result: dict) -> str:
        top5 = result.get("top_pathways", [])[:5]
        lines = [
            f"**Pathway Enrichment Complete**\n",
            f"- KEGG pathways: **{result.get('n_kegg_significant', 0)}**",
            f"- GO terms: **{result.get('n_go_significant', 0)}**\n",
            "**Top pathways:**",
        ]
        for i, p in enumerate(top5):
            lines.append(
                f"  {i+1}. {p['pathway']} "
                f"(adj.p={p.get('p_adjust', '?'):.3e}, genes={p.get('gene_count','?')})"
            )
        return "\n".join(lines)
