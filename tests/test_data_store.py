"""
tests/test_data_store.py
Tests for the per-session DuckDB data store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from core import data_store as ds


@pytest.fixture(autouse=True)
def clear_stores():
    """Make sure every test starts with no registered sessions."""
    yield
    for sid in list(ds._stores.keys()):
        ds.clear(sid)


@pytest.fixture()
def example_sheets():
    return {
        "Proteins": pd.DataFrame({
            "Protein IDs": ["P001", "P002", "P003"],
            "Gene":        ["Myh4", "Tpm3", "Dmd"],
            "A SpC":       [12, 5, 0],
            "B SpC":       [11, 6, 1],
        }),
        "Identifier Info": pd.DataFrame({
            "Label":  ["A", "B"],
            "Group":  ["WT", "DMD"],
        }),
    }


class TestRegisterSheets:

    def test_returns_store_with_tables(self, example_sheets):
        store = ds.register_sheets("sess1", example_sheets)
        assert store is not None
        assert "Proteins" in store.table_names
        assert "Identifier Info" in store.table_names
        # Sanitised names
        assert store.table_names["Proteins"] == "proteins"
        assert store.table_names["Identifier Info"] == "identifier_info"

    def test_re_registration_replaces_tables(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        ds.register_sheets("sess1", {
            "Proteins": pd.DataFrame({"P": ["X"]}),
        })
        store = ds.get_store("sess1")
        assert len(store.table_names) == 1

    def test_empty_dict_returns_none(self):
        result = ds.register_sheets("sess1", {})
        assert result is None

    def test_skips_empty_dataframes(self):
        store = ds.register_sheets("sess1", {
            "Real":   pd.DataFrame({"a": [1, 2]}),
            "Empty":  pd.DataFrame(),
        })
        assert "Real" in store.table_names
        assert "Empty" not in store.table_names


class TestSchemaText:

    def test_renders_each_table(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        text = ds.schema_text("sess1")
        assert "proteins" in text
        assert "identifier_info" in text
        assert "Protein IDs" in text
        assert "shape: 3 ×" in text

    def test_empty_session(self):
        text = ds.schema_text("nonexistent")
        assert "not initialised" in text.lower() or "no tables" in text.lower()


class TestQuery:

    def test_simple_select(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        df, err = ds.query("sess1", 'SELECT "Gene" FROM "proteins"')
        assert err is None
        assert list(df["Gene"]) == ["Myh4", "Tpm3", "Dmd"]

    def test_count(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        df, err = ds.query("sess1", 'SELECT COUNT(*) AS n FROM "proteins"')
        assert err is None
        assert int(df["n"].iloc[0]) == 3

    def test_spaces_in_column_names(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        df, err = ds.query("sess1", 'SELECT "A SpC", "B SpC" FROM "proteins"')
        assert err is None
        assert list(df["A SpC"]) == [12, 5, 0]

    def test_ilike_for_protein_lookup(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        df, err = ds.query("sess1",
            'SELECT "Gene", "A SpC" FROM "proteins" WHERE "Gene" ILIKE \'myh%\'')
        assert err is None
        assert len(df) == 1
        assert df["Gene"].iloc[0] == "Myh4"

    def test_invalid_sql_returns_error(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        df, err = ds.query("sess1", "SELECT * FROM nonexistent_table")
        assert df is None
        assert err is not None
        assert "nonexistent_table" in err.lower() or "catalog" in err.lower()

    def test_query_no_store_returns_error(self):
        df, err = ds.query("missing-session", "SELECT 1")
        assert df is None
        assert "not initialised" in err.lower()


class TestSessionIsolation:

    def test_two_sessions_isolated(self):
        ds.register_sheets("sess_a", {
            "T1": pd.DataFrame({"x": [1, 2, 3]}),
        })
        ds.register_sheets("sess_b", {
            "T1": pd.DataFrame({"x": [10, 20, 30]}),
        })
        df_a, _ = ds.query("sess_a", 'SELECT SUM(x) AS s FROM "t1"')
        df_b, _ = ds.query("sess_b", 'SELECT SUM(x) AS s FROM "t1"')
        assert int(df_a["s"].iloc[0]) == 6
        assert int(df_b["s"].iloc[0]) == 60

    def test_clear_drops_session(self, example_sheets):
        ds.register_sheets("sess1", example_sheets)
        assert ds.get_store("sess1") is not None
        ds.clear("sess1")
        assert ds.get_store("sess1") is None
