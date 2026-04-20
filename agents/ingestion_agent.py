"""
agents/ingestion_agent.py
Data Layer — IngestionAgent

Loads and validates the uploaded proteomics file, surfaces dataset metadata
in BiomarkerState, and auto-routes pooled designs to PooledFoldChangeSkill.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from skills.load_data import DataLoadingSkill

logger  = logging.getLogger(__name__)
settings = get_settings()

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class IngestionAgent(BaseAgent):
    """
    Data Layer – ingests CSV / Excel, normalises orientation,
    detects data type, and surfaces column metadata in state.

    Auto-routing
    ------------
    When the loader detects a MaxQuant-style multi-sheet Excel file it sets
    ``is_pooled_design=True`` and provides a ``label_map``.  This agent
    propagates both to state and sets ``omic_type="proteomics_pooled"`` so
    BiomarkerAgent dispatches PooledFoldChangeSkill without any manual
    group assignment from the user.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_ingestion,
            system_prompt_path="prompts/ingestion_agent.txt",
        )
        self.loader = DataLoadingSkill()

    def run(self, state: BiomarkerState) -> BiomarkerState:
        data_path   = state.get("data_path")
        data_format = state.get("data_format", "csv")

        # ── Guard: skip re-ingestion if data is already loaded ───────────────
        if state.get("data_type") and state.get("n_proteins"):
            logger.info("IngestionAgent: data already loaded (%s), skipping.", state["data_type"])
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Data is already loaded — **{state['n_proteins']} proteins**, "
                    f"**{state['n_samples']} samples** ({state['data_type']}). "
                    "Ready for analysis."
                ),
            })
            return state

        # ── Guard: file must exist ────────────────────────────────────────────
        if not data_path or not os.path.exists(data_path):
            logger.warning("IngestionAgent: no file at '%s'", data_path)
            state["status"] = "error"
            state["error_message"] = "No data file found. Please upload a file."
            state["messages"].append({
                "role": "assistant",
                "content": "No data file found. Please upload a CSV or Excel proteomics file.",
            })
            return state

        # Preserve raw path — PooledFoldChangeSkill reads the original Excel.
        state["raw_data_path"] = data_path
        logger.info("IngestionAgent: loading '%s' (format=%s)", data_path, data_format)

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            result = self.loader.execute(
                data_path=data_path,
                data_format=data_format,
                output_dir=str(PROCESSED_DIR),
            )
        except Exception as exc:
            logger.error("DataLoadingSkill failed: %s", exc, exc_info=True)
            state["status"] = "error"
            state["error_message"] = f"Data loading failed: {exc}"
            state["messages"].append({
                "role": "assistant",
                "content": f"Failed to load data: {exc}",
            })
            return state

        # ── Propagate to state ────────────────────────────────────────────────
        state["data_path"]        = result["processed_path"]
        state["data_type"]        = result["data_type"]
        state["data_format"]      = result["data_format"]
        state["n_proteins"]       = result["n_proteins"]
        state["n_samples"]        = result["n_samples"]
        state["sample_columns"]   = result["sample_columns"]
        state["metadata_columns"] = result["metadata_columns"]
        state["is_pooled_design"] = result.get("is_pooled_design", False)
        state["status"]           = "data_loaded"

        is_pooled       = result.get("is_pooled_design", False)
        label_map       = result.get("label_map")
        identifier_info = result.get("identifier_info")
        all_sheets      = result.get("all_sheets", {})

        if label_map:
            state["label_map"] = label_map
        if identifier_info is not None:
            state["identifier_info"] = identifier_info
        if all_sheets:
            state["all_sheets"] = all_sheets
            logger.info(
                "All sheets stored: %s",
                {k: f"{len(v)} rows × {len(v.columns)} cols" for k, v in all_sheets.items()},
            )

        if is_pooled:
            # Auto-route: BiomarkerAgent will dispatch PooledFoldChangeSkill
            state["omic_type"] = "proteomics_pooled"
            logger.info(
                "Pooled design — omic_type set to 'proteomics_pooled'. Label map: %s",
                label_map,
            )

        # ── Build user-facing message ─────────────────────────────────────────
        state["messages"].append({
            "role": "assistant",
            "content": self._build_load_message(result, is_pooled, label_map),
        })

        logger.info(
            "IngestionAgent done: %d proteins, %d samples, pooled=%s",
            result["n_proteins"], result["n_samples"], is_pooled,
        )
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_load_message(
        result: dict,
        is_pooled: bool,
        label_map: dict | None,
    ) -> str:
        n_proteins = result["n_proteins"]
        n_samples  = result["n_samples"]
        data_type  = result["data_type"]
        sample_preview = result["sample_columns"][:6]
        meta_cols  = result["metadata_columns"]

        lines = [
            "### Data Loaded Successfully\n",
            f"| Field | Value |",
            f"|---|---|",
            f"| Proteins | **{n_proteins}** |",
            f"| Samples / groups | **{n_samples}** |",
            f"| Data type | **{data_type}** |",
        ]

        if sample_preview:
            lines.append(f"| Sample columns (preview) | `{'`, `'.join(str(c) for c in sample_preview)}` |")
        if meta_cols:
            lines.append(f"| Metadata columns | `{'`, `'.join(str(c) for c in meta_cols[:4])}` |")

        if is_pooled and label_map:
            group_str = ", ".join(f"**{k}** → {v}" for k, v in label_map.items())
            lines += [
                "",
                "---",
                "**Pooled design detected** — each group is a single pooled sample.",
                f"Groups: {group_str}",
                "",
                "Fold-change analysis will run automatically across all contrasts "
                "when you click **Run Fold-Change Analysis**.",
            ]
        elif is_pooled:
            lines += [
                "",
                "**Pooled design detected** — click **Run Fold-Change Analysis** to proceed.",
            ]
        else:
            lines += [
                "",
                "Assign samples to **Group 1** and **Group 2** in the sidebar, "
                "then click **Run Analysis**.",
            ]

        return "\n".join(lines)
