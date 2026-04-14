"""
tests/conftest.py
Shared pytest fixtures used across the test suite.
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest


# ── Synthetic proteomics data ─────────────────────────────────────────────────

@pytest.fixture()
def sample_columns() -> List[str]:
    return ["D1", "D2", "D3", "C1", "C2", "C3"]


@pytest.fixture()
def group1_samples() -> List[str]:
    return ["D1", "D2", "D3"]


@pytest.fixture()
def group2_samples() -> List[str]:
    return ["C1", "C2", "C3"]


@pytest.fixture()
def proteomics_df(sample_columns) -> pd.DataFrame:
    """
    Small synthetic proteomics matrix (10 proteins × 6 samples).
    Proteins P001–P003 are artificially elevated in D1–D3 to guarantee
    significant hits in the t-test.
    """
    rng = np.random.default_rng(seed=42)
    proteins = [f"P{i:03d}" for i in range(1, 11)]
    data = rng.normal(loc=10.0, scale=1.0, size=(10, 6))
    # Spike P001–P003 up in group 1 (disease)
    data[:3, :3] += 4.0
    df = pd.DataFrame(data, index=proteins, columns=sample_columns)
    return df


@pytest.fixture()
def proteomics_csv(tmp_path: Path, proteomics_df: pd.DataFrame) -> Path:
    """Write the synthetic DataFrame to a CSV and return its path."""
    path = tmp_path / "test_proteomics.csv"
    proteomics_df.to_csv(path)
    return path
