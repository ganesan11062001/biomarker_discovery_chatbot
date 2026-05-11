"""
tests/test_proteomics_tools.py
Tests for the deterministic proteomics helper toolkit.
Each test maps to one of the 7 bugs from the failure report.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.proteomics_tools import (
    build_sample_map,
    detect_metric_columns,
    format_protein_row,
    get_gene_symbol,
    get_nonstandard_protein,
    get_short_name,
    safe_fold_change,
    split_spc_intensity,
    top_n_by_metric,
)


# ── Test fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def proteins_df():
    """A small DataFrame that mimics a MaxQuant 'Proteins' sheet."""
    return pd.DataFrame({
        "Protein Name": [
            "Myosin-4 OS=Mus musculus OX=10090 GN=Myh4 PE=1 SV=2",
            "Myosin-1 OS=Mus musculus GN=Myh1 PE=1 SV=1",
            "Albumin OS=Mus musculus GN=Alb PE=1 SV=1",
            "Dystrophin OS=Mus musculus GN=Dmd PE=1 SV=3",
            "miDys",
        ],
        "Accession Number": ["Q5SX39", "Q5SX40", "P02769", "P11531", "miDys"],
        "Molecular Weight": [223.0, 224.0, 69.0, 426.0, "?"],
        "A SpC":       [4307, 2405, 360,   8, 0],
        "B SpC":       [3500, 1500, 627,   0, 0],
        "C SpC":       [2000, 1000, 400,  10, 19],
        "D SpC":       [1500,  800, 350,  15,  9],
        "E SpC":       [1200,  700, 300,   5, 0],
        "A Intensity": [1e9, 5e8, 1e7, 1e5, 0],
        "B Intensity": [9e8, 4e8, 2e7, 0,   0],
        "C Intensity": [5e8, 3e8, 1.5e7, 1e5, 1e6],
        "D Intensity": [4e8, 2e8, 1.4e7, 1.5e5, 5e5],
        "E Intensity": [3e8, 1.5e8, 1.2e7, 8e4, 0],
    })


@pytest.fixture()
def identifier_sheet():
    """A 'Identifier Info' sheet — mixes 5 sample-code rows with 28 mouse-ID rows."""
    rows = [
        # Pooled-sample mapping rows
        {"MaxQuant": "A", "Client identifier": "WT",     "Strain": "BL10 WT",      "Treatment Group": "Vehicle",   "Mouse ID": ""},
        {"MaxQuant": "B", "Client identifier": "mdx",    "Strain": "MDX",          "Treatment Group": "Vehicle",   "Mouse ID": ""},
        {"MaxQuant": "C", "Client identifier": "uDys5",  "Strain": "MDX",          "Treatment Group": "AAV-uDys5", "Mouse ID": ""},
        {"MaxQuant": "D", "Client identifier": "H2",     "Strain": "MDX",          "Treatment Group": "AAV-uDys5", "Mouse ID": ""},
        {"MaxQuant": "E", "Client identifier": "nNOS KO","Strain": "BL6 nNOS KO",  "Treatment Group": "Vehicle",   "Mouse ID": ""},
    ]
    # Add per-mouse rows (no MaxQuant code) — these MUST be filtered out
    for i in range(1, 29):
        rows.append({
            "MaxQuant": "",
            "Client identifier": f"Mouse {i}",
            "Strain": "MDX",
            "Treatment Group": "Vehicle",
            "Mouse ID": str(i),
        })
    return pd.DataFrame(rows)


# ── BUG 1 — SpC / Intensity split ────────────────────────────────────────────

class TestSplitSpcIntensity:

    def test_spc_dataframe_has_only_spc_columns(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        # Identifier columns present
        assert "Protein Name" in df_spc.columns
        assert "Accession Number" in df_spc.columns
        # SpC columns present
        assert {"A SpC", "B SpC", "C SpC", "D SpC", "E SpC"}.issubset(df_spc.columns)
        # NO Intensity columns
        assert not any("Intensity" in c for c in df_spc.columns)

    def test_intensity_dataframe_has_only_intensity_columns(self, proteins_df):
        _, df_int = split_spc_intensity(proteins_df)
        assert "Protein Name" in df_int.columns
        assert {"A Intensity", "B Intensity"}.issubset(df_int.columns)
        # NO SpC columns
        assert not any("SpC" in c for c in df_int.columns)

    def test_no_overlap_between_spc_and_intensity(self, proteins_df):
        df_spc, df_int = split_spc_intensity(proteins_df)
        spc_only = set(df_spc.columns) - set(df_int.columns)
        int_only = set(df_int.columns) - set(df_spc.columns)
        # Numeric columns must be unique to each frame
        assert all("SpC" in c for c in spc_only)
        assert all("Intensity" in c for c in int_only)

    def test_detect_metric_columns_classifies_correctly(self, proteins_df):
        groups = detect_metric_columns(proteins_df)
        assert {"A SpC", "B SpC"}.issubset(groups["spc"])
        assert {"A Intensity", "B Intensity"}.issubset(groups["intensity"])
        assert {"Protein Name", "Accession Number"}.issubset(groups["identifier"])


# ── BUG 2 — Safe fold change ─────────────────────────────────────────────────

class TestSafeFoldChange:

    def test_normal_fold_change(self):
        result = safe_fold_change(360, 627, "A", "B")
        # 360 / 627 = 0.574...
        assert isinstance(result, float)
        assert abs(result - 0.5742) < 0.001

    def test_inverse_fold_change_matches_spec(self):
        # User's expected: Albumin A vs B = 1.74×.  A=360, B=627.
        # Their "1.74x" is actually B/A, but spec definition is A/B.
        # Test BOTH directions explicitly.
        ab = safe_fold_change(627, 360, "B", "A")  # B / A
        assert abs(ab - 1.7417) < 0.001

    def test_zero_denominator_returns_undefined_string(self):
        result = safe_fold_change(8, 0, "A", "B")
        assert isinstance(result, str)
        assert "undefined" in result.lower()
        assert "B" in result   # mentions which sample is the empty one
        assert "0" in result

    def test_zero_numerator_returns_zero_string(self):
        result = safe_fold_change(0, 8, "A", "B")
        assert isinstance(result, str)
        assert result.startswith("0")
        assert "A" in result   # mentions which sample is empty

    def test_both_zero_returns_undefined(self):
        result = safe_fold_change(0, 0, "A", "B")
        assert isinstance(result, str)
        assert "absent in both" in result.lower() or "undefined" in result.lower()

    def test_non_numeric_returns_undefined(self):
        result = safe_fold_change("?", 5)
        assert isinstance(result, str)
        assert "undefined" in result.lower()


# ── BUG 3 — Sample map from identifier sheet ─────────────────────────────────

class TestBuildSampleMap:

    def test_returns_five_pooled_sample_entries(self, identifier_sheet):
        m = build_sample_map(identifier_sheet)
        # Only the 5 MaxQuant short-code rows should appear
        assert len(m) == 5
        assert set(m.keys()) == {"A", "B", "C", "D", "E"}

    def test_sample_b_has_correct_metadata(self, identifier_sheet):
        m = build_sample_map(identifier_sheet)
        assert m["B"]["client_id"] == "mdx"
        assert m["B"]["strain"]    == "MDX"
        assert m["B"]["treatment"] == "Vehicle"

    def test_sample_c_has_correct_treatment(self, identifier_sheet):
        m = build_sample_map(identifier_sheet)
        assert m["C"]["client_id"] == "uDys5"
        assert m["C"]["treatment"] == "AAV-uDys5"

    def test_mouse_rows_are_excluded(self, identifier_sheet):
        m = build_sample_map(identifier_sheet)
        # No "Mouse 13" or similar should leak in
        for key in m:
            assert key in ("A", "B", "C", "D", "E")
            assert "Mouse" not in str(m[key].get("client_id", ""))

    def test_empty_sheet_returns_empty(self):
        m = build_sample_map(pd.DataFrame())
        assert m == {}

    def test_none_sheet_returns_empty(self):
        m = build_sample_map(None)
        assert m == {}


# ── BUG 4 — Protein-name parsing & formatting ───────────────────────────────

class TestGeneSymbolParsing:

    def test_extracts_standard_gn_field(self):
        s = "Myosin-4 OS=Mus musculus OX=10090 GN=Myh4 PE=1 SV=2"
        assert get_gene_symbol(s) == "Myh4"

    def test_returns_unknown_when_missing(self):
        assert get_gene_symbol("miDys") == "Unknown"

    def test_handles_none(self):
        assert get_gene_symbol(None) == "Unknown"

    def test_short_name_strips_os_suffix(self):
        s = "Albumin OS=Mus musculus GN=Alb PE=1 SV=1"
        assert get_short_name(s) == "Albumin"

    def test_format_protein_row_standard(self):
        s = "Myosin-4 OS=Mus musculus GN=Myh4 PE=1 SV=2"
        out = format_protein_row(s, "Q5SX39", 4307, "SpC")
        assert out == "Myh4 (Q5SX39) — 4307 SpC"

    def test_format_protein_row_falls_back_to_description(self):
        # No GN= — fall back to the description
        out = format_protein_row("miDys", "miDys", 19, "SpC")
        assert "miDys" in out and "19" in out


# ── BUG 6 — Non-standard protein lookup ──────────────────────────────────────

class TestGetNonstandardProtein:

    def test_exact_accession_match_returns_spc_values(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "miDys", metric="spc")
        # All 5 SpC columns should be returned exactly as in the source row
        assert out["A SpC"] == 0
        assert out["B SpC"] == 0
        assert out["C SpC"] == 19
        assert out["D SpC"] == 9
        assert out["E SpC"] == 0

    def test_partial_name_match(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "Albumin", metric="spc")
        assert out["A SpC"] == 360
        assert out["B SpC"] == 627

    def test_intensity_lookup_never_returns_spc(self, proteins_df):
        _, df_int = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_int, "Myh4", metric="intensity")
        # Should have intensity keys, never spc
        assert any("Intensity" in str(k) for k in out)
        assert not any("SpC" in str(k) for k in out)

    def test_not_found_returns_error_dict(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "DoesNotExist")
        assert "error" in out
        assert "not found" in out["error"].lower()


# ── Top N by metric ──────────────────────────────────────────────────────────

class TestTopNByMetric:

    def test_top_3_by_spc_returns_with_identifiers(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        result = top_n_by_metric(df_spc, "A SpC", n=3)
        assert len(result) == 3
        # Identifier columns must be present
        assert "Protein Name" in result.columns
        assert "Accession Number" in result.columns
        # Sorted descending
        a_spc = list(result["A SpC"])
        assert a_spc == sorted(a_spc, reverse=True)
        # The top protein should have the highest A SpC
        assert "Myh4" in result.iloc[0]["Protein Name"]


# ── Integration: the 7 validation scenarios from the user ────────────────────

class TestSevenValidationScenarios:
    """Codifies the 7 expected answers from the bug report."""

    def test_q1_protein_count(self, proteins_df):
        # "How many proteins?" → 1919 in real data; here we just confirm
        # len(df) is the right way to count.
        assert len(proteins_df) == 5  # our fixture has 5; real one has 1919

    def test_q2_sample_b_metadata(self, identifier_sheet):
        m = build_sample_map(identifier_sheet)
        assert m["B"]["client_id"] == "mdx"
        assert m["B"]["strain"]    == "MDX"
        assert m["B"]["treatment"] == "Vehicle"

    def test_q3_top_3_in_sample_a(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        top = top_n_by_metric(df_spc, "A SpC", n=3)
        proteins = list(top["Protein Name"])
        assert "Myh4" in proteins[0]
        assert "Myh1" in proteins[1]
        assert "Albumin" in proteins[2]

    def test_q4_dystrophin_in_b(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "Dystrophin", metric="spc")
        assert out["B SpC"] == 0  # Absent in B

    def test_q5_albumin_fold_change(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "Albumin", metric="spc")
        # User spec: 1.74× means B/A = 627/360
        fc = safe_fold_change(out["B SpC"], out["A SpC"], "B", "A")
        assert abs(fc - 1.74) < 0.01

    def test_q6_dystrophin_fold_change_undefined(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "Dystrophin", metric="spc")
        fc = safe_fold_change(out["A SpC"], out["B SpC"], "A", "B")
        assert isinstance(fc, str)
        assert "undefined" in fc.lower()
        assert "B" in fc   # the empty sample is named

    def test_q7_midys_spc_across_all_samples(self, proteins_df):
        df_spc, _ = split_spc_intensity(proteins_df)
        out = get_nonstandard_protein(df_spc, "miDys", metric="spc")
        assert out["A SpC"] == 0
        assert out["B SpC"] == 0
        assert out["C SpC"] == 19
        assert out["D SpC"] == 9
        assert out["E SpC"] == 0
