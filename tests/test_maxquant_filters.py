"""
tests/test_maxquant_filters.py
Tests for the MaxQuant cleanup primitives in core.maxquant_filters.

Maps each function to the open-source project it was ported from:
  - remove_columns_containing / remove_reverse / remove_potential_contaminants /
    remove_only_identified_by_site  ← pymaxquant
  - flag_maxquant_contaminants                                  ← alphapeptstats
  - normalise_column_names / fix_locale_decimals / coerce_metric_columns_numeric
                                                                ← mspypeline
  - assert_columns_present                                      ← alphapeptstats
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.maxquant_filters import (
    RequiredColumnsMissing,
    apply_standard_cleanup,
    assert_columns_present,
    coerce_metric_columns_numeric,
    fix_locale_decimals,
    flag_maxquant_contaminants,
    normalise_column_names,
    remove_all_contaminants,
    remove_columns_containing,
    remove_only_identified_by_site,
    remove_potential_contaminants,
    remove_reverse,
)


# ── pymaxquant-style filters ─────────────────────────────────────────────────

class TestPymaxquantFilters:

    def test_remove_columns_containing_drops_plus_rows(self):
        df = pd.DataFrame({"flag": ["+", "", "+", "x"], "v": [1, 2, 3, 4]})
        out = remove_columns_containing(df, "flag", "+")
        assert list(out["v"]) == [2, 4]

    def test_remove_reverse_uses_reverse_column(self):
        df = pd.DataFrame({
            "Protein IDs": ["P1", "P2", "P3"],
            "Reverse":     ["",   "+",  ""],
        })
        out = remove_reverse(df)
        assert list(out["Protein IDs"]) == ["P1", "P3"]

    def test_remove_potential_contaminants(self):
        df = pd.DataFrame({
            "Protein IDs":           ["P1", "P2", "P3"],
            "Potential contaminant": ["",   "+",  ""],
        })
        out = remove_potential_contaminants(df)
        assert list(out["Protein IDs"]) == ["P1", "P3"]

    def test_remove_only_identified_by_site(self):
        df = pd.DataFrame({
            "Protein IDs":                 ["P1", "P2"],
            "Only identified by site":     ["",   "+"],
        })
        out = remove_only_identified_by_site(df)
        assert list(out["Protein IDs"]) == ["P1"]

    def test_filters_are_no_op_when_marker_column_absent(self):
        df = pd.DataFrame({"Protein IDs": ["P1", "P2"]})
        assert len(remove_reverse(df)) == 2
        assert len(remove_potential_contaminants(df)) == 2
        assert len(remove_only_identified_by_site(df)) == 2

    def test_remove_all_contaminants_chains_filters(self):
        df = pd.DataFrame({
            "Protein IDs":                 ["P1", "P2", "P3", "P4"],
            "Reverse":                     ["",   "+",  "",   ""],
            "Potential contaminant":       ["",   "",   "+",  ""],
            "Only identified by site":     ["",   "",   "",   "+"],
        })
        out = remove_all_contaminants(df)
        assert list(out["Protein IDs"]) == ["P1"]


# ── alphapeptstats-style flag (don't delete) ────────────────────────────────

class TestFlagContaminants:

    def test_flag_adds_boolean_columns(self):
        df = pd.DataFrame({
            "Protein IDs":           ["P1", "CON__BSA", "REV__P3"],
            "Reverse":               ["",   "",          "+"],
            "Potential contaminant": ["",   "+",         ""],
        })
        out = flag_maxquant_contaminants(df)
        # Source rows are NEVER deleted — flag-and-keep pattern
        assert len(out) == 3
        assert out["is_reverse"].tolist()                 == [False, False, True]
        assert out["is_potential_contaminant"].tolist()   == [False, True,  False]
        assert out["is_contaminant_accession"].tolist()   == [False, True,  True]
        assert out["is_contaminant"].tolist()             == [False, True,  True]

    def test_flag_works_when_only_some_markers_exist(self):
        # Workbook has only Reverse column; CON_ accessions still get flagged
        df = pd.DataFrame({
            "Protein IDs": ["P1", "CON__X"],
            "Reverse":     ["",   ""],
        })
        out = flag_maxquant_contaminants(df)
        assert out["is_reverse"].tolist()              == [False, False]
        assert out["is_contaminant_accession"].tolist() == [False, True]
        assert out["is_contaminant"].tolist()           == [False, True]

    def test_flag_when_no_markers_present(self):
        df = pd.DataFrame({
            "Gene": ["A", "B"],
            "x":    [1, 2],
        })
        out = flag_maxquant_contaminants(df)
        # All boolean columns added but all False
        assert all(out["is_contaminant"] == False)
        assert all(out["is_contaminant_accession"] == False)


# ── mspypeline-style normalisation ───────────────────────────────────────────

class TestNormaliseColumnNames:

    def test_strips_whitespace_and_casts_to_str(self):
        df = pd.DataFrame({"  Gene  ": [1], 123: [2]})
        out = normalise_column_names(df)
        assert "Gene" in out.columns
        assert "123" in out.columns


class TestFixLocaleDecimals:

    def test_european_decimals_to_float(self):
        df = pd.DataFrame({
            "Intensity": ["1,5", "2,75", "3,0", "10,25"],
        })
        out = fix_locale_decimals(df, ["Intensity"])
        assert out["Intensity"].tolist() == [1.5, 2.75, 3.0, 10.25]

    def test_mixed_garbage_falls_through(self):
        # Less than 70% of cells parse as numeric — leave column alone
        df = pd.DataFrame({"Note": ["hello", "world", "1,5", "x"]})
        out = fix_locale_decimals(df, ["Note"])
        # Should still be strings because parse-rate < 70%
        assert out["Note"].dtype == object or out["Note"].dtype == "str"

    def test_auto_detect_columns_when_none_passed(self):
        df = pd.DataFrame({
            "Gene": ["A", "B", "C"],
            "MW":   ["1,0", "2,5", "3,75"],
        })
        out = fix_locale_decimals(df)
        assert out["MW"].tolist() == [1.0, 2.5, 3.75]
        # Gene column unchanged (no numbers in it)
        assert out["Gene"].tolist() == ["A", "B", "C"]


class TestCoerceMetricColumnsNumeric:

    def test_unparseable_becomes_nan(self):
        import math
        df = pd.DataFrame({
            "A SpC": ["10", "?", "20", "n/a"],
            "Gene":  ["X", "Y", "Z", "W"],
        })
        out = coerce_metric_columns_numeric(df, ["A SpC"])
        assert out["A SpC"].iloc[0] == 10
        assert math.isnan(out["A SpC"].iloc[1])
        assert out["A SpC"].iloc[2] == 20

    def test_missing_columns_silently_skipped(self):
        df = pd.DataFrame({"x": [1, 2]})
        out = coerce_metric_columns_numeric(df, ["does_not_exist"])
        assert list(out.columns) == ["x"]


# ── Required-column assertion ────────────────────────────────────────────────

class TestAssertColumnsPresent:

    def test_all_present_passes(self):
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        assert_columns_present(df, ["A", "C"])  # must not raise

    def test_case_insensitive_match(self):
        df = pd.DataFrame({"Gene Name": [1], "Accession": [2]})
        assert_columns_present(df, ["gene name", "accession"])

    def test_missing_raises(self):
        df = pd.DataFrame({"A": [1]})
        with pytest.raises(RequiredColumnsMissing):
            assert_columns_present(df, ["A", "B"])


# ── End-to-end cleanup pipeline ──────────────────────────────────────────────

class TestApplyStandardCleanup:

    def test_flag_mode_keeps_all_rows(self):
        df = pd.DataFrame({
            "Protein IDs":           ["P1", "CON__X", "REV__P3"],
            "Reverse":               ["",   "",        "+"],
            "Potential contaminant": ["",   "+",       ""],
            "A Intensity":           ["1,5", "2,5",    "3,0"],
        })
        out = apply_standard_cleanup(df)
        # No rows dropped
        assert len(out) == 3
        # Contaminants flagged
        assert out["is_contaminant"].sum() == 2
        # Decimal commas fixed
        assert out["A Intensity"].iloc[0] == 1.5

    def test_drop_mode_removes_contaminants(self):
        df = pd.DataFrame({
            "Protein IDs":           ["P1", "P2"],
            "Reverse":               ["",   "+"],
            "Potential contaminant": ["",   ""],
        })
        out = apply_standard_cleanup(df, drop_contaminants=True,
                                      flag_contaminants=False)
        assert len(out) == 1
        assert out["Protein IDs"].iloc[0] == "P1"

    def test_no_op_for_clean_data(self):
        df = pd.DataFrame({"Gene": ["A", "B"], "Value": [1.0, 2.0]})
        out = apply_standard_cleanup(df)
        # Flag columns added but all False (no markers)
        assert "is_contaminant" in out.columns
        assert out["is_contaminant"].sum() == 0
        # Original data intact
        assert out["Value"].tolist() == [1.0, 2.0]
