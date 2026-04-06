"""
Data Layer – DataLoadingSkill
Loads CSV or Excel proteomics intensity matrices and auto-detects data type.
Expected format: rows = proteins/features, columns = samples (or transposed).
"""
import os
import uuid
from pathlib import Path
import pandas as pd
from skills.base_skill import BaseSkill


def _detect_data_type(df: pd.DataFrame) -> str:
    """Heuristically detect whether this is Olink NPX, MS intensity, or generic."""
    cols_lower = [c.lower() for c in df.columns]
    index_lower = [str(i).lower() for i in df.index]

    if any("npx" in c or "olink" in c for c in cols_lower + index_lower):
        return "olink_npx"
    if any("intensity" in c or "lfq" in c or "tmt" in c for c in cols_lower + index_lower):
        return "ms_intensity"
    return "generic"


class DataLoadingSkill(BaseSkill):
    """
    Loads a CSV or Excel proteomics file.

    Conventions accepted:
      - Proteins as rows, samples as columns (proteins × samples)
      - Samples as rows, proteins as columns (samples × proteins) — auto-transposed

    Returns a normalised CSV (proteins × samples) plus basic metadata.
    """

    def __init__(self):
        super().__init__(script_path="")  # pure Python, no R script

    def execute(self, data_path: str, data_format: str = "csv", output_dir: str = "data/processed") -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data_format = data_format.lower()

        # Load
        if data_format in ("xlsx", "xls", "excel"):
            df = pd.read_excel(data_path, index_col=0)
            data_format = "excel"
        else:
            df = pd.read_csv(data_path, index_col=0)
            data_format = "csv"

        # Auto-transpose if samples appear to be rows (more rows than columns is a hint
        # that samples are columns; if equal, leave as is)
        if df.shape[0] < df.shape[1]:
            # More columns than rows → rows are proteins; correct orientation
            pass
        elif df.shape[0] > df.shape[1] * 2:
            # Likely samples × proteins orientation — transpose
            df = df.T

        # Drop fully empty rows / columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        data_type = _detect_data_type(df)

        # Save normalised CSV to processed/
        out_name = f"{Path(data_path).stem}_processed_{uuid.uuid4().hex[:8]}.csv"
        out_path = str(Path(output_dir) / out_name)
        df.to_csv(out_path)

        return {
            "processed_path": out_path,
            "data_type": data_type,
            "data_format": data_format,
            "n_proteins": df.shape[0],
            "n_samples": df.shape[1],
        }
