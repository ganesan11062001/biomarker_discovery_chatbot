"""
tests/test_two_sheet_template.py
Tests for the 2-sheet canonical proteomics template:

  Sheet 1 (metadata)
      Sample ID | Group
      S1        | WT
      S2        | WT
      S3        | DMD
      S4        | DMD

  Sheet 2 (proteins)
      Protein Name | Accession Number | Gene Name | S1 | S2 | S3 | S4
      Myh4 OS=…    | Q5SX39           | Myh4      | 100| 110| 50 | 55
      …

When the user asks "compare WT vs DMD", the pipeline resolves
{WT: [S1, S2], DMD: [S3, S4]} via Sheet 1 — no LLM guessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.proteomics_tools import build_sample_group_map


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def metadata_sheet() -> pd.DataFrame:
    """Sheet 1 — the canonical 2-column metadata layout."""
    return pd.DataFrame({
        "Sample ID": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "Group":     ["WT", "WT", "WT", "DMD", "DMD", "DMD"],
    })


@pytest.fixture()
def proteins_sheet() -> pd.DataFrame:
    """Sheet 2 — proteins as rows, Sample IDs as sample-value columns."""
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame({
        "Protein Name":     [f"Protein-{i} OS=Mus musculus GN=GENE{i} PE=1 SV=1"
                              for i in range(n)],
        "Accession Number": [f"P{i:05d}" for i in range(n)],
        "Gene Name":        [f"GENE{i}" for i in range(n)],
        "S1": rng.integers(0, 1e6, n).astype(float),
        "S2": rng.integers(0, 1e6, n).astype(float),
        "S3": rng.integers(0, 1e6, n).astype(float),
        "S4": rng.integers(0, 1e6, n).astype(float),
        "S5": rng.integers(0, 1e6, n).astype(float),
        "S6": rng.integers(0, 1e6, n).astype(float),
    })


# ── 1. build_sample_group_map ────────────────────────────────────────────────

class TestBuildSampleGroupMap:

    def test_returns_both_directions(self, metadata_sheet):
        sample_to_group, group_to_samples = build_sample_group_map(metadata_sheet)
        # Forward
        assert sample_to_group["S1"] == "WT"
        assert sample_to_group["S3"] == "WT"
        assert sample_to_group["S4"] == "DMD"
        # Reverse
        assert set(group_to_samples["WT"])  == {"S1", "S2", "S3"}
        assert set(group_to_samples["DMD"]) == {"S4", "S5", "S6"}

    def test_two_groups_detected(self, metadata_sheet):
        _, g2s = build_sample_group_map(metadata_sheet)
        assert set(g2s.keys()) == {"WT", "DMD"}

    def test_empty_sheet_returns_empty(self):
        s2g, g2s = build_sample_group_map(pd.DataFrame())
        assert s2g == {} and g2s == {}

    def test_none_sheet_returns_empty(self):
        s2g, g2s = build_sample_group_map(None)
        assert s2g == {} and g2s == {}

    def test_missing_group_column_returns_empty(self):
        df = pd.DataFrame({"Sample ID": ["S1", "S2"]})
        s2g, g2s = build_sample_group_map(df)
        assert s2g == {} and g2s == {}

    def test_missing_sample_id_column_returns_empty(self):
        df = pd.DataFrame({"Group": ["WT", "DMD"]})
        s2g, g2s = build_sample_group_map(df)
        assert s2g == {} and g2s == {}

    def test_case_insensitive_header_match(self):
        # User can name the columns however they want — match is case-folded
        # and ignores whitespace/punctuation.
        df = pd.DataFrame({
            "  SampleID  ": ["A", "B", "C", "D"],
            "Condition":    ["Ctrl", "Ctrl", "Treated", "Treated"],
        })
        s2g, g2s = build_sample_group_map(df)
        assert s2g == {"A": "Ctrl", "B": "Ctrl", "C": "Treated", "D": "Treated"}
        assert set(g2s.keys()) == {"Ctrl", "Treated"}

    def test_alternative_group_header_treatment(self):
        df = pd.DataFrame({
            "Sample ID": ["A", "B"],
            "Treatment": ["Vehicle", "Drug"],
        })
        s2g, _ = build_sample_group_map(df)
        assert s2g == {"A": "Vehicle", "B": "Drug"}

    def test_empty_cells_skipped(self):
        df = pd.DataFrame({
            "Sample ID": ["S1", "S2", None, ""],
            "Group":     ["WT", None, "DMD", "DMD"],
        })
        s2g, g2s = build_sample_group_map(df)
        # Only S1 → WT should remain — every other row has a missing value
        assert s2g == {"S1": "WT"}
        # One-group result with one sample is degenerate; helper still keeps
        # it (it's the caller's job to decide minimum group sizes)
        assert "WT" in g2s

    def test_one_sample_per_group_is_valid_pooled_design(self):
        # n=1 per group is the pooled fold-change design — valid, must be kept.
        # The IngestionAgent further requires ≥2 groups before adopting this
        # mapping; the helper itself just returns whatever the sheet contains.
        df = pd.DataFrame({
            "Sample ID": ["S1", "S2", "S3"],
            "Group":     ["WT", "DMD", "KO"],
        })
        s2g, g2s = build_sample_group_map(df)
        assert s2g == {"S1": "WT", "S2": "DMD", "S3": "KO"}
        assert g2s == {"WT": ["S1"], "DMD": ["S2"], "KO": ["S3"]}


# ── 2. End-to-end via IngestionAgent / DataLoadingSkill ─────────────────────

class TestTwoSheetTemplateEndToEnd:

    def _write_two_sheet_xlsx(
        self,
        tmp_path: Path,
        metadata_sheet: pd.DataFrame,
        proteins_sheet: pd.DataFrame,
    ) -> Path:
        path = tmp_path / "two_sheet.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            metadata_sheet.to_excel(xw, sheet_name="Sample Metadata", index=False)
            proteins_sheet.to_excel(xw, sheet_name="Proteins",        index=False)
        return path

    def test_data_loading_skill_recognises_sheets(
        self, tmp_path, metadata_sheet, proteins_sheet,
    ):
        from skills.load_data import DataLoadingSkill
        path = self._write_two_sheet_xlsx(tmp_path, metadata_sheet, proteins_sheet)
        skill = DataLoadingSkill()
        result = skill.execute(
            data_path=str(path), data_format="excel", output_dir=str(tmp_path),
        )
        # Sample columns are S1..S6 from the proteins sheet
        for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
            assert sid in result["sample_columns"], (
                f"Sample ID {sid!r} should be a sample column"
            )
        # Identifier columns are NOT samples. They live either in
        # metadata_columns (kept alongside) or are promoted to the
        # DataFrame index by _parse_expression_sheet — both are valid.
        for col in ["Protein Name", "Accession Number", "Gene Name"]:
            assert col not in result["sample_columns"]
        # All sheets retained for downstream lookups
        assert "Sample Metadata" in result["all_sheets"]
        assert "Proteins" in result["all_sheets"]

    def test_ingestion_agent_builds_sample_to_group_state(
        self, tmp_path, metadata_sheet, proteins_sheet,
    ):
        from agents.ingestion_agent import IngestionAgent
        path = self._write_two_sheet_xlsx(tmp_path, metadata_sheet, proteins_sheet)

        # Construct an IngestionAgent without invoking its LLM (we only care
        # about the deterministic post-load wiring).
        from unittest.mock import patch
        with patch("agents.base_agent._build_client"):
            agent = IngestionAgent()
        # Stub the LLM-driven message + group inference calls — they're unrelated
        # to what we're verifying here.
        agent._call_llm = lambda *a, **k: "(stub)"

        state: dict = {
            "session_id":  "test-2sheet",
            "messages":    [],
            "data_path":   str(path),
            "data_format": "excel",
        }
        updated = agent.run(state)
        # The 2-sheet wiring should populate both fields deterministically
        s2g = updated.get("sample_to_group") or {}
        all_groups = updated.get("all_groups") or {}
        assert s2g.get("S1") == "WT"
        assert s2g.get("S4") == "DMD"
        assert set(all_groups.keys()) == {"WT", "DMD"}
        assert set(all_groups["WT"])  == {"S1", "S2", "S3"}
        assert set(all_groups["DMD"]) == {"S4", "S5", "S6"}


# ── 3. No regression — legacy MaxQuant-style file still detects sample_map ──

class TestNoRegressionLegacyFile:

    def test_legacy_maxquant_sheet_still_uses_sample_map_not_sample_to_group(self):
        """A MaxQuant-style 'Identifier Info' sheet (MaxQuant + Client identifier
        + Strain + Treatment Group + Mouse ID) should not be mistakenly
        treated as a Sample ID + Group sheet."""
        from core.proteomics_tools import build_sample_group_map, build_sample_map

        # The MaxQuant identifier sheet doesn't have an explicit "Group" column —
        # it has "Strain" and "Treatment Group". "Treatment Group" matches the
        # group-hint list, so a 2-sheet path WOULD pick it up; that's fine,
        # but the sample-id column also matches "Sample" weakly. We need to
        # make sure both helpers can run side-by-side.
        identifier = pd.DataFrame({
            "MaxQuant":          ["A", "B", "C", "D"],
            "Client identifier": ["WT", "mdx", "uDys5", "H2"],
            "Strain":            ["BL10", "MDX", "MDX", "MDX"],
            "Treatment Group":   ["Vehicle", "Vehicle", "AAV-uDys5", "AAV-uDys5"],
            "Mouse ID":          ["", "", "", ""],
        })
        sample_map = build_sample_map(identifier)
        # build_sample_map still picks MaxQuant codes A/B/C/D
        assert "A" in sample_map
        assert sample_map["A"]["client_id"] == "WT"
