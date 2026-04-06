from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings
from skills.run_visualization import ReportingSkill

settings = get_settings()


class VisualizationAgent(BaseAgent):
    """
    Output Layer – generates visualizations and a plain-language summary report:
      - Volcano plot (DEA results)
      - Heatmap (top proteins × samples)
      - Dot plot (pathway enrichment)
      - Biomarker ranking table (CSV)
      - Plain-language summary (LLM-generated)
    """

    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_visualization,
            system_prompt_path="prompts/visualization_agent.txt",
        )
        self.reporting_skill = ReportingSkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        if not state.get("dea_result_path"):
            state["status"] = "error"
            state["error_message"] = "No analysis results to visualize."
            state["messages"].append({
                "role": "assistant",
                "content": "No analysis results found. Please run proteomics analysis first."
            })
            return state

        try:
            result = self.reporting_skill.execute(
                dea_result_path=state["dea_result_path"],
                enrichment_result_path=state.get("enrichment_result_path"),
                top_proteins=state.get("top_proteins", []),
                top_pathways=state.get("pathways", []),
                contrast_groups=state.get("contrast_groups", []),
                disease_program=state.get("disease_program", ""),
            )

            state["plot_paths"] = result["plot_paths"]
            state["report_path"] = result.get("report_path")
            state["status"] = "report_ready"

            plots_list = "\n".join(f"  - {p}" for p in result["plot_paths"])
            plain_summary = result.get("plain_language_summary", "")

            msg = f"Report and visualizations ready.\n\nGenerated files:\n{plots_list}\n"
            if plain_summary:
                msg += f"\n**Summary:**\n{plain_summary}"

            # Generate plain-language summary via LLM
            if state.get("top_proteins") and not plain_summary:
                llm_summary = self._generate_summary(state)
                state["messages"].append({"role": "assistant", "content": llm_summary})
            else:
                state["messages"].append({"role": "assistant", "content": msg})

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Visualization failed: {str(e)}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Visualization failed: {str(e)}"
            })

        return state

    def _generate_summary(self, state: BiomarkerState) -> str:
        top_proteins = state.get("top_proteins", [])[:10]
        pathways = state.get("pathways", [])[:5]
        contrast = state.get("contrast_groups", [])
        disease = state.get("disease_program", "")

        context = (
            f"Disease program: {disease}\n"
            f"Comparison: {contrast[0]} vs {contrast[1] if len(contrast) > 1 else 'Control'}\n"
            f"Top proteins: {[p['protein'] for p in top_proteins]}\n"
            f"Top pathways: {[p['pathway'] for p in pathways]}\n"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Write a concise plain-language summary (3-5 sentences) of these proteomics findings:\n{context}"
            )}
        ]
        return self._call_llm(messages, max_tokens=300)
