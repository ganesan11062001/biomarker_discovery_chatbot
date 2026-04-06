from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings
from skills.run_qc import QCSkill
from skills.run_limma import ProteomicsSkill

settings = get_settings()


class BiomarkerAgent(BaseAgent):
    """
    Analysis Layer – orchestrates the proteomics pipeline:
      1. QC Skill          (Data Layer)     – missing value filter, CV cutoff, outlier detection
      2. Proteomics Skill  (Analysis Layer) – limma / DEP / MSstats differential expression
    """

    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_biomarker,
            system_prompt_path="prompts/biomarker_agent.txt",
        )
        self.qc_skill = QCSkill()
        self.proteomics_skill = ProteomicsSkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        if not state.get("data_path"):
            state["status"] = "error"
            state["error_message"] = "No data loaded. Please upload a CSV or Excel file first."
            state["messages"].append({
                "role": "assistant",
                "content": "No proteomics data found. Please upload your data file first."
            })
            return state

        # Step 1: QC (if not already done)
        if not state.get("qc_passed"):
            state = self._run_qc(state)
            if state["status"] == "error":
                return state

        # Step 2: Differential Expression (if group config is available)
        if state.get("sample_group_col") and state.get("contrast_groups"):
            if not state.get("dea_result_path"):
                state = self._run_dea(state)
        else:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "QC passed. To run differential expression analysis, please specify:\n"
                    "- **Sample group column**: the column in your data that defines groups "
                    "(e.g. 'Group', 'Condition')\n"
                    "- **Contrast groups**: the two groups to compare "
                    "(e.g. ['Disease', 'Control'])"
                )
            })
            state["status"] = "awaiting_config"

        return state

    def _run_qc(self, state: BiomarkerState) -> BiomarkerState:
        try:
            result = self.qc_skill.execute(
                data_path=state["data_path"],
                data_type=state.get("data_type", "generic"),
            )
            state["qc_report_path"] = result["qc_report_path"]
            state["qc_passed"] = result["qc_passed"]
            state["data_path"] = result.get("filtered_data_path", state["data_path"])
            state["status"] = "qc_complete"

            msg = (
                f"QC complete.\n"
                f"- Proteins passing filter: {result['proteins_retained']} / {result['proteins_total']}\n"
                f"- Samples retained: {result['samples_retained']}\n"
                f"- Missing value threshold: {result['missing_threshold']*100:.0f}%\n"
                f"- CV cutoff applied: {result.get('cv_cutoff', 'N/A')}\n"
                f"- Outlier samples removed: {result.get('outliers_removed', 0)}\n"
            )
            if result["qc_passed"]:
                msg += "\nData passed QC. Ready for differential expression analysis."
            else:
                msg += "\nWarning: data quality issues detected. Review QC report before proceeding."

            state["messages"].append({"role": "assistant", "content": msg})

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"QC failed: {str(e)}"
            state["messages"].append({"role": "assistant", "content": f"QC failed: {str(e)}"})

        return state

    def _run_dea(self, state: BiomarkerState) -> BiomarkerState:
        try:
            result = self.proteomics_skill.execute(
                data_path=state["data_path"],
                sample_group_col=state["sample_group_col"],
                contrast_groups=state["contrast_groups"],
                data_type=state.get("data_type", "generic"),
            )
            state["dea_result_path"] = result["dea_result_path"]
            state["top_proteins"] = result["top_proteins"]
            state["status"] = "dea_complete"

            top_list = "\n".join(
                f"  {i+1}. {p['protein']} (logFC={p['logFC']:.2f}, adj.P={p['adj_pval']:.3e})"
                for i, p in enumerate(result["top_proteins"][:10])
            )
            msg = (
                f"Differential expression analysis complete.\n"
                f"- Method: {result['method']}\n"
                f"- Significant proteins (adj.P < 0.05): {result['n_significant']}\n"
                f"- Up-regulated: {result['n_up']}  |  Down-regulated: {result['n_down']}\n\n"
                f"Top 10 proteins:\n{top_list}\n\n"
                "Would you like to run pathway enrichment analysis?"
            )
            state["messages"].append({"role": "assistant", "content": msg})

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"DEA failed: {str(e)}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Differential expression analysis failed: {str(e)}"
            })

        return state
