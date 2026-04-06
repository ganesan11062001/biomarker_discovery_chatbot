from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings
from skills.run_enrichment import PathwaySkill

settings = get_settings()


class EnrichmentAgent(BaseAgent):
    """
    Knowledge Layer – runs pathway enrichment on DEA results:
      - KEGG pathway analysis
      - GO term enrichment (BP, MF, CC)
      - clusterProfiler (R) / GSEApy (Python) backends
    """

    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_enrichment,
            system_prompt_path="prompts/enrichment_agent.txt",
        )
        self.pathway_skill = PathwaySkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        if not state.get("dea_result_path"):
            state["status"] = "error"
            state["error_message"] = "No DEA results found. Run differential expression analysis first."
            state["messages"].append({
                "role": "assistant",
                "content": "No differential expression results found. Please run proteomics analysis first."
            })
            return state

        if not state.get("top_proteins"):
            state["messages"].append({
                "role": "assistant",
                "content": "No significant proteins found from DEA. Pathway enrichment requires significant hits."
            })
            state["status"] = "no_significant_proteins"
            return state

        try:
            protein_list = [p["protein"] for p in state["top_proteins"]]
            result = self.pathway_skill.execute(
                protein_list=protein_list,
                dea_result_path=state["dea_result_path"],
                organism=state.get("organism", "human"),
            )

            state["enrichment_result_path"] = result["enrichment_result_path"]
            state["pathways"] = result["top_pathways"]
            state["status"] = "enrichment_complete"

            kegg_list = "\n".join(
                f"  {i+1}. {p['pathway']} (p.adj={p['p_adjust']:.3e}, genes={p['gene_count']})"
                for i, p in enumerate(result["top_pathways"][:5])
            )
            msg = (
                f"Pathway enrichment analysis complete.\n"
                f"- KEGG pathways enriched: {result['n_kegg_significant']}\n"
                f"- GO terms enriched: {result['n_go_significant']}\n\n"
                f"Top KEGG pathways:\n{kegg_list}\n\n"
                "Would you like to generate a volcano plot and summary report?"
            )
            state["messages"].append({"role": "assistant", "content": msg})

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Pathway enrichment failed: {str(e)}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Pathway enrichment failed: {str(e)}"
            })

        return state
