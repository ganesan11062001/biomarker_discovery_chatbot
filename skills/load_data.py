"""
skills/load_data.py
Data Layer – DataLoadingSkill

Loads CSV or Excel proteomics matrices, auto-detects orientation,
and separates sample (numeric) columns from metadata columns.
"""
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from skills.base_skill import BaseSkill


# Patterns that suggest a column is a protein identifier (not a sample name)
_PROTEIN_ID_PATTERNS = [
    re.compile(r"^(P|Q|O)\d{5}", re.I),       # UniProt accession
    re.compile(r"\w{2,}_HUMAN$", re.I),         # Gene_SPECIES
    re.compile(r"^ENSP\d+", re.I),             # Ensembl protein
    re.compile(r"^sp\|", re.I),                # SwissProt FASTA header fragment
]

# Column name hints that strongly suggest a metadata / group column
_METADATA_HINTS = {
    "group", "condition", "sample", "label", "type", "class",
    "patient", "disease", "treatment", "control", "status",
    "gender", "sex", "age", "batch", "time", "timepoint",
    "subject", "donor", "cohort", "replicate",
}


def _detect_data_type(df: pd.DataFrame, sample_cols: List[str]) -> str:
    """Heuristic: classify data as olink_npx, ms_lfq, or generic."""
    combined = [c.lower() for c in df.columns] + [str(i).lower() for i in df.index]

    if any("npx" in s or "olink" in s for s in combined):
        return "olink_npx"
    if any(kw in s for s in combined for kw in ("intensity", "lfq", "tmt", "itraq", "ms1")):
        return "ms_lfq"

    if not sample_cols:
        return "generic"

    numeric_data = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    max_val = numeric_data.max().max()
    min_val = numeric_data.min().min()

    if pd.notna(max_val):
        if -5 <= float(min_val) and float(max_val) <= 20:
            return "olink_npx"   # pre-logged values
        if float(max_val) > 1_000:
            return "ms_lfq"      # raw intensities

    return "generic"


def _separate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Split columns into numeric sample columns and metadata columns.
    Returns (sample_cols, metadata_cols).
    """
    sample_cols, metadata_cols = [], []

    for col in df.columns:
        col_lower = str(col).lower()

        # If column name contains a metadata hint, treat as metadata
        if any(hint in col_lower for hint in _METADATA_HINTS):
            metadata_cols.append(col)
            continue

        # Check numeric fraction
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        numeric_frac = numeric_vals.notna().sum() / max(len(df[col]), 1)

        if numeric_frac >= 0.7:
            sample_cols.append(col)
        else:
            metadata_cols.append(col)

    return sample_cols, metadata_cols


def _looks_like_protein_index(index) -> bool:
    """Return True if more than 30 % of index entries match protein ID patterns."""
    sample = [str(x) for x in list(index)[:20]]
    hits = sum(
        1 for s in sample
        if any(p.search(s) for p in _PROTEIN_ID_PATTERNS)
    )
    return hits / max(len(sample), 1) >= 0.3


class DataLoadingSkill(BaseSkill):
    """
    Loads a CSV or Excel proteomics file.

    Orientation accepted:
      - Proteins × Samples  (rows = proteins, columns = samples) — standard
      - Samples × Proteins  (rows = samples, columns = proteins) — auto-transposed
    """

    def __init__(self):
        super().__init__(script_path="")   # pure Python, no R

    def execute(
        self,
        data_path: str,
        data_format: str = "csv",
        output_dir: str = "data/processed",
    ) -> dict:
        """
        Load, orient, and persist a proteomics file.

        Returns
        -------
        dict with keys:
          processed_path, data_type, data_format,
          n_proteins, n_samples, sample_columns, metadata_columns
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data_format = data_format.lower()

        # ── 1. Load ──────────────────────────────────────────────────────────
        path = Path(data_path)
        suffix = path.suffix.lower()

        if suffix in (".xlsx", ".xls") or data_format in ("xlsx", "xls", "excel"):
            df = pd.read_excel(data_path, index_col=0)
            data_format = "excel"
        else:
            df = self._load_csv(data_path)
            data_format = "csv"

        # ── 2. Drop empty rows / columns ─────────────────────────────────────
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        # Convert index to strings
        df.index = df.index.astype(str)

        # ── 3. Auto-orient (proteins × samples) ──────────────────────────────
        df = self._ensure_proteins_are_rows(df)

        # ── 4. Separate sample vs metadata columns ───────────────────────────
        sample_cols, metadata_cols = _separate_columns(df)

        # ── 5. Detect data type ───────────────────────────────────────────────
        data_type = _detect_data_type(df, sample_cols)

        # ── 6. Persist processed CSV ─────────────────────────────────────────
        out_name = f"{path.stem}_processed_{uuid.uuid4().hex[:8]}.csv"
        out_path = str(Path(output_dir) / out_name)
        df.to_csv(out_path)

        return {
            "processed_path": out_path,
            "data_type": data_type,
            "data_format": data_format,
            "n_proteins": len(df),
            "n_samples": len(sample_cols),
            "sample_columns": sample_cols,
            "metadata_columns": metadata_cols,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_csv(self, path: str) -> pd.DataFrame:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, index_col=0, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, index_col=0)

    def _ensure_proteins_are_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transpose only when samples appear to be rows."""
        n_rows, n_cols = df.shape

        rows_are_proteins = _looks_like_protein_index(df.index)
        cols_are_proteins = _looks_like_protein_index(df.columns)

        # Explicit protein-column pattern → transpose
        if cols_are_proteins and not rows_are_proteins:
            return df.T

        # Many more columns than rows & rows don't look like proteins → transpose
        if n_cols > n_rows * 3 and not rows_are_proteins:
            return df.T

        return df
