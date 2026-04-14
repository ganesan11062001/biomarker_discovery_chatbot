"""
agents/ingestion_agent.py
Data Layer — loads and validates the uploaded proteomics file.
Now also exposes sample_columns and metadata_columns in state.
"""
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
    Data Layer – ingests CSV/Excel, normalises orientation,
    detects data type, and surfaces column metadata.
    """

    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_ingestion,
            system_prompt_path="prompts/ingestion_agent.txt",
        )
        self.loader = DataLoadingSkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        data_path   = state.get("data_path")
        data_format = state.get("data_format", "csv")

        if not data_path or not os.path.exists(data_path):
            state["status"] = "error"
            state["error_message"] = "No data file found. Please upload a CSV or Excel file."
            state["messages"].append({
                "role": "assistant",
                "content": "No data file found. Please upload a CSV or Excel proteomics file.",
            })
            return state

        try:
            result = self.loader.execute(
                data_path=data_path,
                data_format=data_format,
                output_dir=str(PROCESSED_DIR),
            )

            state["data_path"]        = result["processed_path"]
            state["data_type"]        = result["data_type"]
            state["data_format"]      = result["data_format"]
            state["n_proteins"]       = result["n_proteins"]
            state["n_samples"]        = result["n_samples"]
            state["sample_columns"]   = result["sample_columns"]
            state["metadata_columns"] = result["metadata_columns"]
            state["status"]           = "data_loaded"

            sample_preview = result["sample_columns"][:8]
            meta_note = (
                f"\n- Metadata columns detected: {result['metadata_columns']}"
                if result["metadata_columns"] else ""
            )

            msg = (
                f"Data loaded successfully.\n\n"
                f"- **Proteins:** {result['n_proteins']}\n"
                f"- **Samples:** {result['n_samples']}\n"
                f"- **Data type:** {result['data_type']}\n"
                f"- **Sample columns (preview):** {sample_preview}"
                f"{meta_note}\n\n"
                "To run differential expression analysis, please assign your samples "
                "to Group 1 and Group 2 in the sidebar, then click **Run Analysis**."
            )
            state["messages"].append({"role": "assistant", "content": msg})

        except Exception as exc:
            state["status"] = "error"
            state["error_message"] = f"Data loading failed: {exc}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Failed to load data: {exc}",
            })

        return state
