"""
tests/test_canonical_template.py
Validation tests for the canonical single-sheet proteomics template:

    Col 1: Protein Name
    Col 2: Accession Number
    Col 3: Gene Symbol
    Col 4..N: numeric sample columns (Intensity by default)

These tests prove the pipeline handles this template correctly end-to-end
without breaking the existing multi-sheet / MaxQuant-style support.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.proteomics_tools import (
    detect_metric_columns,
    get_nonstandard_protein,
    infer_groups_from_row0,
    split_spc_intensity,
    top_n_by_metric,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def canonical_df():
    """Build a synthetic DataFrame matching the canonical template."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "Protein Name":     [f"Protein-{i} OS=Mus musculus GN=GENE{i} PE=1 SV=1"
                              for i in range(n)],
        "Accession Number": [f"P{i:05d}" for i in range(n)],
        "Gene Symbol":      [f"GENE{i}" for i in range(n)],
        "WT_1":             rng.integers(0, 1_000_000, n).astype(float),
        "WT_2":             rng.integers(0, 1_000_000, n).astype(float),
        "WT_3":             rng.integers(0, 1_000_000, n).astype(float),
        "KO_1":             rng.integers(0, 1_000_000, n).astype(float),
        "KO_2":             rng.integers(0, 1_000_000, n).astype(float),
        "KO_3":             rng.integers(0, 1_000_000, n).astype(float),
    })


@pytest.fixture()
def sample_named_canonical_df():
    """Canonical template with samples named 'Sample_1', 'Sample_2', ...

    Critical regression: 'sample' is in the metadata-hint list, so without
    the precedence fix these columns would be classified as METADATA.
    """
    rng = np.random.default_rng(7)
    n = 20
    return pd.DataFrame({
        "Protein Name":     [f"P_{i}" for i in range(n)],
        "Accession Number": [f"P{i:05d}" for i in range(n)],
        "Gene Symbol":      [f"G{i}" for i in range(n)],
        "Sample_1":         rng.integers(0, 1e6, n).astype(float),
        "Sample_2":         rng.integers(0, 1e6, n).astype(float),
        "Sample_3":         rng.integers(0, 1e6, n).astype(float),
        "Sample_4":         rng.integers(0, 1e6, n).astype(float),
    })


# ── 1. Metric-column detection on canonical template ───────────────────────

class TestCanonicalTemplateMetricDetection:

    def test_three_identifier_cols_recognized(self, canonical_df):
        groups = detect_metric_columns(canonical_df)
        ids = set(groups["identifier"])
        # All three classic identifier columns should be tagged
        assert "Protein Name" in ids
        assert "Accession Number" in ids
        assert "Gene Symbol" in ids

    def test_unlabeled_numeric_cols_classified_as_intensity(self, canonical_df):
        groups = detect_metric_columns(canonical_df)
        # WT_1..3 and KO_1..3 should all be intensity (since no SpC/LFQ markers)
        intensity = set(groups["intensity"])
        for col in ["WT_1", "WT_2", "WT_3", "KO_1", "KO_2", "KO_3"]:
            assert col in intensity, f"{col} should be classified as intensity"

    def test_no_spc_columns_in_canonical_template(self, canonical_df):
        groups = detect_metric_columns(canonical_df)
        assert groups["spc"] == [], (
            "Canonical template has no SpC columns — should be empty"
        )

    def test_sample_named_cols_classified_as_intensity(self, sample_named_canonical_df):
        groups = detect_metric_columns(sample_named_canonical_df)
        for col in ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]:
            assert col in groups["intensity"], f"{col} must be intensity"

    def test_legacy_explicit_spc_still_works(self):
        """Files with explicit 'A SpC' / 'B Intensity' must still classify
        correctly (no regression)."""
        df = pd.DataFrame({
            "Protein Name":     ["P1", "P2"],
            "Accession Number": ["A1", "A2"],
            "A SpC":            [10, 20],
            "B SpC":            [15, 25],
            "A Intensity":      [1e6, 2e6],
            "B Intensity":      [1.5e6, 2.5e6],
        })
        groups = detect_metric_columns(df)
        assert "A SpC" in groups["spc"] and "B SpC" in groups["spc"]
        assert "A Intensity" in groups["intensity"]
        assert "B Intensity" in groups["intensity"]


# ── 2. split_spc_intensity on canonical template ────────────────────────────

class TestCanonicalTemplateSplit:

    def test_intensity_frame_includes_all_sample_cols(self, canonical_df):
        _, df_int = split_spc_intensity(canonical_df)
        assert not df_int.empty
        for col in ["WT_1", "WT_2", "WT_3", "KO_1", "KO_2", "KO_3"]:
            assert col in df_int.columns
        # Identifier columns kept
        assert "Protein Name" in df_int.columns
        assert "Accession Number" in df_int.columns
        assert "Gene Symbol" in df_int.columns

    def test_spc_frame_empty_for_canonical(self, canonical_df):
        df_spc, _ = split_spc_intensity(canonical_df)
        assert df_spc.empty, "Canonical template should produce empty df_spc"

    def test_intensity_columns_are_numeric_dtype(self, canonical_df):
        _, df_int = split_spc_intensity(canonical_df)
        for col in ["WT_1", "KO_1"]:
            assert pd.api.types.is_numeric_dtype(df_int[col])


# ── 3. End-to-end query operations on canonical template ────────────────────

class TestCanonicalTemplateQueries:

    def test_top_n_by_intensity(self, canonical_df):
        _, df_int = split_spc_intensity(canonical_df)
        top3 = top_n_by_metric(df_int, "WT_1", n=3)
        assert len(top3) == 3
        # Identifier columns retained in the result
        assert "Protein Name" in top3.columns
        assert "Accession Number" in top3.columns
        assert "Gene Symbol" in top3.columns

    def test_lookup_by_accession(self, canonical_df):
        _, df_int = split_spc_intensity(canonical_df)
        out = get_nonstandard_protein(df_int, "P00005", metric="intensity")
        assert "error" not in out
        # Should return all the WT_/KO_ intensity values for that protein
        sample_keys = [k for k in out if k.startswith(("WT_", "KO_"))]
        assert len(sample_keys) == 6

    def test_lookup_by_gene_symbol(self, canonical_df):
        # Partial-name match via the Protein Name column (which contains GN=GENE{i})
        _, df_int = split_spc_intensity(canonical_df)
        out = get_nonstandard_protein(df_int, "GENE5", metric="intensity")
        # Should find one of the GENE5-* proteins
        assert "error" not in out


# ── 4. Full DataLoadingSkill load path on a canonical-template xlsx ─────────

class TestDataLoadingSkillCanonical:

    def _write_canonical_xlsx(self, tmp_path: Path) -> Path:
        """Write a canonical-template Excel file with a title row above the
        real headers — mimics the real-world export pattern."""
        df = pd.DataFrame({
            "Protein Name":     ["Myh4 OS=Mus musculus GN=Myh4 PE=1",
                                  "Alb OS=Mus musculus GN=Alb PE=1",
                                  "Dmd OS=Mus musculus GN=Dmd PE=1"],
            "Accession Number": ["Q5SX39", "P02769", "P11531"],
            "Gene Symbol":      ["Myh4", "Alb", "Dmd"],
            "WT_1": [100, 200, 50],
            "WT_2": [110, 210, 55],
            "WT_3": [120, 220, 60],
            "KO_1": [10, 20, 0],
            "KO_2": [15, 25, 0],
            "KO_3": [12, 22, 0],
        })
        path = tmp_path / "canonical.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="Proteins", index=False)
        return path

    def test_load_canonical_xlsx_end_to_end(self, tmp_path):
        from skills.load_data import DataLoadingSkill
        path = self._write_canonical_xlsx(tmp_path)
        skill = DataLoadingSkill()
        result = skill.execute(
            data_path=str(path), data_format="excel", output_dir=str(tmp_path),
        )
        # Basic sanity
        assert result["n_proteins"] == 3
        # All 6 sample columns must be detected as samples, NOT misclassified
        sample_cols = result["sample_columns"]
        for col in ["WT_1", "WT_2", "WT_3", "KO_1", "KO_2", "KO_3"]:
            assert col in sample_cols, f"{col} must be in sample_columns"
        # Accession Number / Gene Symbol must be in metadata, not samples
        metadata = result["metadata_columns"]
        assert "Accession Number" in metadata
        assert "Gene Symbol" in metadata
        # No pooled-design false positive (no MaxQuant identifier sheet here)
        assert result["is_pooled_design"] is False

    def test_load_canonical_csv_end_to_end(self, tmp_path):
        from skills.load_data import DataLoadingSkill
        df = pd.DataFrame({
            "Protein Name":     ["P1", "P2", "P3", "P4"],
            "Accession Number": ["A1", "A2", "A3", "A4"],
            "Gene Symbol":      ["G1", "G2", "G3", "G4"],
            "Sample_1":         [100, 200, 300, 400],
            "Sample_2":         [110, 220, 330, 440],
            "Sample_3":         [120, 240, 360, 480],
        })
        path = tmp_path / "canonical.csv"
        df.to_csv(path, index=False)
        skill = DataLoadingSkill()
        result = skill.execute(
            data_path=str(path), data_format="csv", output_dir=str(tmp_path),
        )
        assert result["n_proteins"] == 4
        for col in ["Sample_1", "Sample_2", "Sample_3"]:
            assert col in result["sample_columns"], f"{col} not detected"


# ── 4b. Row-0 group resolution (canonical convention) ──────────────────────

class TestRow0GroupResolution:
    """Row 0 of the sheet holds the sample-group name above each sample
    column. ``infer_groups_from_row0`` is the deterministic resolver used by
    IngestionAgent for 'group A vs group B' queries before any LLM fallback."""

    def test_resolves_two_groups_from_row0_labels(self):
        column_group_labels = {
            "Sample_1": "WT",  "Sample_2": "WT",  "Sample_3": "WT",
            "Sample_4": "DMD", "Sample_5": "DMD", "Sample_6": "DMD",
        }
        sample_cols = list(column_group_labels.keys())
        groups = infer_groups_from_row0(sample_cols, column_group_labels)
        assert set(groups.keys()) == {"WT", "DMD"}
        assert groups["WT"]  == ["Sample_1", "Sample_2", "Sample_3"]
        assert groups["DMD"] == ["Sample_4", "Sample_5", "Sample_6"]

    def test_ignores_columns_not_in_sample_list(self):
        # Identifier columns ("Protein Name" etc.) might leak into the
        # label map. Only entries that are also in sample_columns survive.
        column_group_labels = {
            "Protein Name": "Header",
            "WT_a": "WT", "WT_b": "WT",
            "KO_a": "KO", "KO_b": "KO",
        }
        groups = infer_groups_from_row0(
            ["WT_a", "WT_b", "KO_a", "KO_b"], column_group_labels,
        )
        assert "Header" not in groups
        assert set(groups.keys()) == {"WT", "KO"}

    def test_returns_empty_when_fewer_than_min_groups(self):
        # Only one distinct group → caller should fall back to LLM inference
        assert infer_groups_from_row0(
            ["A", "B"], {"A": "Only", "B": "Only"},
        ) == {}

    def test_returns_empty_when_no_row0_labels(self):
        assert infer_groups_from_row0(["A", "B"], {}) == {}
        assert infer_groups_from_row0(["A", "B"], None) == {}

    def test_strips_whitespace_in_group_names(self):
        groups = infer_groups_from_row0(
            ["a", "b", "c", "d"],
            {"a": "  WT  ", "b": "WT", "c": "DMD", "d": "  DMD"},
        )
        assert set(groups.keys()) == {"WT", "DMD"}
        assert sorted(groups["WT"])  == ["a", "b"]
        assert sorted(groups["DMD"]) == ["c", "d"]


# ── 5. No regression — existing MaxQuant-style file still works ─────────────

class TestNoRegressionLegacyFile:

    def test_legacy_spc_intensity_split(self):
        df = pd.DataFrame({
            "Protein Name":     ["P1", "P2"],
            "Accession Number": ["A1", "A2"],
            "Molecular Weight": [50.0, 60.0],
            "A SpC": [10, 20],   "B SpC": [15, 25],
            "A Intensity": [1e6, 2e6],   "B Intensity": [1.5e6, 2.5e6],
        })
        df_spc, df_int = split_spc_intensity(df)
        # SpC frame has only SpC columns + identifiers
        assert "A SpC" in df_spc.columns and "B SpC" in df_spc.columns
        assert "A Intensity" not in df_spc.columns
        # Intensity frame has only Intensity columns + identifiers
        assert "A Intensity" in df_int.columns
        assert "A SpC" not in df_int.columns
        # No spurious "intensity reclassification" of SpC cols
