import os
from pathlib import Path
from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings
from skills.load_data import DataLoadingSkill

settings = get_settings()

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class IngestionAgent(BaseAgent):
    """
    Data Layer – ingests CSV or Excel proteomics files, validates structure,
    and surfaces a clean data_path for downstream skills.
    """

    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_chat,
            system_prompt_path="prompts/ingestion_agent.txt",
        )
        self.loader = DataLoadingSkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        data_path = state.get("data_path")
        data_format = state.get("data_format", "csv")

        if not data_path or not os.path.exists(data_path):
            state["status"] = "error"
            state["error_message"] = (
                "No data file provided. Please upload a CSV or Excel file first."
            )
            state["messages"].append({
                "role": "assistant",
                "content": "No data file found. Please upload a CSV or Excel proteomics file."
            })
            return state

        try:
            result = self.loader.execute(
                data_path=data_path,
                data_format=data_format,
                output_dir=str(PROCESSED_DIR),
            )

            state["data_path"] = result["processed_path"]
            state["data_type"] = result["data_type"]
            state["data_format"] = result["data_format"]
            state["n_proteins"] = result["n_proteins"]
            state["n_samples"] = result["n_samples"]
            state["status"] = "data_loaded"

            summary_msg = (
                f"Data loaded successfully.\n"
                f"- Samples: {result['n_samples']}\n"
                f"- Proteins: {result['n_proteins']}\n"
                f"- Data type detected: {result['data_type']}\n"
                f"- Format: {result['data_format'].upper()}\n\n"
                "Ready for QC. Would you like to run quality control now?"
            )
            state["messages"].append({"role": "assistant", "content": summary_msg})

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Data loading failed: {str(e)}"
            state["messages"].append(
                {"role": "assistant", "content": f"Failed to load data: {str(e)}"}
            )

        return state
