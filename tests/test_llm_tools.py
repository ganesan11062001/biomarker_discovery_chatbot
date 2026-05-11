"""
tests/test_llm_tools.py
Tests for the OpenAI tool-calling layer in core/llm_tools.py:
  - get_openai_tool_specs()
  - execute_tool_call() dispatching
  - tool executors (load_preview_data / complex_duckdb_query /
    simple_dataframe_query)

No real LLM calls — every test stubs Azure / OpenAI.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from core import data_store as ds
from core import llm_tools as lt


@pytest.fixture(autouse=True)
def isolated_data_store():
    """Each test starts with no registered sessions."""
    yield
    for sid in list(ds._stores.keys()):
        ds.clear(sid)


@pytest.fixture()
def session_with_sheets():
    """Register a tiny workbook and return its session id."""
    sid = "test-session-llm-tools"
    sheets = {
        "Proteins": pd.DataFrame({
            "Protein Name":     ["Albumin", "Myosin-4", "miDys"],
            "Accession Number": ["P02769",  "Q5SX39",   "miDys"],
            "A SpC":            [360, 4307, 0],
            "B SpC":            [627, 3500, 0],
        }),
        "Identifier Info": pd.DataFrame({
            "MaxQuant":          ["A",     "B"],
            "Client identifier": ["WT",    "mdx"],
            "Strain":            ["BL10",  "MDX"],
            "Treatment Group":   ["Veh",   "Veh"],
        }),
    }
    ds.register_sheets(sid, sheets)
    return sid


# ── Tool spec generation ─────────────────────────────────────────────────────

class TestGetOpenaiToolSpecs:

    def test_returns_three_tools(self):
        specs = lt.get_openai_tool_specs()
        names = {s["function"]["name"] for s in specs}
        assert names == {
            "load_preview_data",
            "complex_duckdb_query",
            "simple_dataframe_query",
        }

    def test_each_spec_is_openai_function_format(self):
        for spec in lt.get_openai_tool_specs():
            assert spec["type"] == "function"
            fn = spec["function"]
            assert "name"        in fn
            assert "description" in fn
            assert "parameters"  in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties"   in params

    def test_duckdb_query_requires_query_arg(self):
        spec = next(s for s in lt.get_openai_tool_specs()
                    if s["function"]["name"] == "complex_duckdb_query")
        assert "query" in spec["function"]["parameters"]["required"]


# ── load_preview_data ────────────────────────────────────────────────────────

class TestLoadPreviewData:

    def test_returns_columns_and_samples(self, session_with_sheets):
        result_json = lt.execute_tool_call(
            "load_preview_data", "{}", {"session_id": session_with_sheets},
        )
        result = json.loads(result_json)
        assert "proteins" in result
        assert "Protein Name" in result["proteins"]["columns"]
        assert "A SpC" in result["proteins"]["columns"]
        # Schema dtypes also present
        assert "Protein Name" in result["proteins"]["dtypes"]
        # First 3 sample rows
        assert len(result["proteins"]["sample_rows"]) == 3

    def test_returns_error_for_unknown_session(self):
        result_json = lt.execute_tool_call(
            "load_preview_data", "{}", {"session_id": "nonexistent"},
        )
        result = json.loads(result_json)
        assert "error" in result


# ── complex_duckdb_query ─────────────────────────────────────────────────────

class TestComplexDuckdbQuery:

    def test_simple_count(self, session_with_sheets):
        args = json.dumps({"query": 'SELECT COUNT(*) AS n FROM "proteins"'})
        result_json = lt.execute_tool_call(
            "complex_duckdb_query", args, {"session_id": session_with_sheets},
        )
        result = json.loads(result_json)
        assert "result" in result
        assert result["result"]["rows"][0]["n"] == 3

    def test_select_with_spaces(self, session_with_sheets):
        args = json.dumps({
            "query": 'SELECT "Protein Name", "A SpC" FROM "proteins" ORDER BY "A SpC" DESC LIMIT 2'
        })
        result_json = lt.execute_tool_call(
            "complex_duckdb_query", args, {"session_id": session_with_sheets},
        )
        result = json.loads(result_json)
        rows = result["result"]["rows"]
        assert rows[0]["Protein Name"] == "Myosin-4"
        assert rows[0]["A SpC"]        == 4307

    def test_invalid_sql_returns_error(self, session_with_sheets):
        args = json.dumps({"query": "SELECT * FROM doesnt_exist"})
        result_json = lt.execute_tool_call(
            "complex_duckdb_query", args, {"session_id": session_with_sheets},
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "doesnt_exist" in result["error"].lower() or "catalog" in result["error"].lower()

    def test_empty_query_returns_error(self, session_with_sheets):
        args = json.dumps({"query": ""})
        result_json = lt.execute_tool_call(
            "complex_duckdb_query", args, {"session_id": session_with_sheets},
        )
        result = json.loads(result_json)
        assert "error" in result


# ── simple_dataframe_query ───────────────────────────────────────────────────

class TestSimpleDataframeQuery:

    def test_pandas_snippet_with_namespace(self):
        ns = {
            "df": pd.DataFrame({"x": [1, 2, 3]}),
            "pd": pd,
        }
        args = json.dumps({"query": "answer = int(df['x'].sum())"})
        result_json = lt.execute_tool_call(
            "simple_dataframe_query", args, {"namespace": ns},
        )
        result = json.loads(result_json)
        assert result.get("result") == 6

    def test_unsafe_code_rejected(self):
        ns = {"pd": pd}
        args = json.dumps({"query": "import os; answer = os.getcwd()"})
        result_json = lt.execute_tool_call(
            "simple_dataframe_query", args, {"namespace": ns},
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "unsafe" in result["error"].lower()

    def test_missing_answer_variable_returns_error(self):
        ns = {"df": pd.DataFrame({"x": [1, 2]}), "pd": pd}
        args = json.dumps({"query": "z = df.shape"})
        result_json = lt.execute_tool_call(
            "simple_dataframe_query", args, {"namespace": ns},
        )
        result = json.loads(result_json)
        assert "error" in result

    def test_dataframe_answer_serialised(self):
        ns = {
            "df": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "pd": pd,
        }
        args = json.dumps({"query": "answer = df.head(2)"})
        result_json = lt.execute_tool_call(
            "simple_dataframe_query", args, {"namespace": ns},
        )
        result = json.loads(result_json)
        ans = result["result"]
        assert ans["type"]    == "dataframe"
        assert ans["columns"] == ["a", "b"]
        assert len(ans["rows"]) == 2


# ── Dispatcher behaviour ─────────────────────────────────────────────────────

class TestExecuteToolCall:

    def test_unknown_tool_returns_error(self):
        result_json = lt.execute_tool_call("bogus_tool", "{}", {})
        result = json.loads(result_json)
        assert "unknown tool" in result["error"].lower()

    def test_invalid_json_args_returns_error(self):
        result_json = lt.execute_tool_call(
            "complex_duckdb_query", "{not json", {"session_id": "x"},
        )
        result = json.loads(result_json)
        assert "invalid tool arguments" in result["error"].lower()
