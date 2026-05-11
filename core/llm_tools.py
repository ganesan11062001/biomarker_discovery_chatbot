"""
core/llm_tools.py
OpenAI function-calling tools, modelled on ExcelWorkerLLMToolCallAgent
(jenyss/ExcelWorkerLLMToolCallAgent).

Three tools the LLM can call to answer cell-level questions about the
user's uploaded workbook:

  1. load_preview_data        — schema + 3-row preview per registered sheet
  2. complex_duckdb_query     — run SQL against the session's DuckDB store
  3. simple_dataframe_query   — run safe pandas code against the primary sheet

Each tool's executor reads from session-scoped state (DuckDB connection +
per-sheet DataFrames). The session_id is injected by the dispatcher; the
LLM never sees or sends it.

This module is intentionally framework-agnostic — it builds OpenAI-spec
JSON tool definitions and a string-name → callable dispatch table. The
caller (LearningAgent._query_data_via_tools) drives the tool-call loop.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """A single LLM-callable tool: OpenAI spec + Python executor."""
    name:        str
    description: str
    parameters:  Dict[str, Any]                       # JSON Schema for OpenAI
    executor:    Callable[[Dict[str, Any]], Any]      # takes merged args+context


# ── Tool executors ────────────────────────────────────────────────────────────
# Each executor takes a single dict combining the LLM-provided arguments and
# the dispatcher-injected context (session_id, primary sheet DataFrame, etc.).

def _exec_load_preview_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return {sheet_sql_name: {columns, dtypes, sample_rows}} for every
    DuckDB-registered table in this session."""
    session_id = args.get("session_id", "")
    try:
        from core import data_store
    except ImportError:
        return {"error": "data_store unavailable"}
    if not data_store.is_available():
        return {"error": "DuckDB not installed in this environment"}

    store = data_store.get_store(session_id)
    if store is None or not store.table_names:
        return {"error": "no sheets registered — upload a file first"}

    preview: Dict[str, Any] = {}
    for sheet_name, sql_name in store.table_names.items():
        try:
            cols_meta = store.con.execute(f'DESCRIBE "{sql_name}"').fetchall()
            preview_df = store.con.execute(
                f'SELECT * FROM "{sql_name}" LIMIT 3'
            ).fetchdf()
            preview[sql_name] = {
                "sheet_name": sheet_name,
                "columns":    [c[0] for c in cols_meta],
                "dtypes":     {c[0]: c[1] for c in cols_meta},
                "sample_rows": preview_df.replace(
                    {float("inf"): None, float("-inf"): None}
                ).where(preview_df.notna(), None).to_dict(orient="records"),
            }
        except Exception as exc:
            preview[sql_name] = {"error": str(exc)}
    return preview


def _exec_complex_duckdb_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute SQL against the session DuckDB store. Returns capped rows."""
    session_id = args.get("session_id", "")
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "empty query"}

    try:
        from core import data_store
    except ImportError:
        return {"error": "data_store unavailable"}

    df, err = data_store.query(session_id, query, max_rows=200)
    if err is not None:
        return {"error": err, "query": query}
    if df is None:
        return {"result": None, "query": query}

    # Null-normalise to JSON-friendly None (ExcelWorker pattern)
    import pandas as pd
    df = df.replace({float("inf"): None, float("-inf"): None})
    df = df.where(pd.notna(df), None)
    return {
        "result": {
            "columns": list(df.columns),
            "rows":    df.to_dict(orient="records"),
            "row_count": len(df),
        },
        "query": query,
    }


def _exec_simple_dataframe_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute LLM-generated pandas code against the primary sheet via the
    safe-exec sandbox. The snippet should set a variable named `answer`.
    """
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "empty pandas snippet"}

    namespace = args.get("namespace") or {}
    if not namespace:
        return {"error": "no execution namespace"}

    try:
        from core.safe_exec import safe_exec, UnsafeCodeError, CodeTimeoutError
    except ImportError:
        return {"error": "safe_exec unavailable"}

    try:
        safe_exec(query, namespace, timeout=15)
    except UnsafeCodeError as exc:
        return {"error": f"unsafe code rejected: {exc}"}
    except CodeTimeoutError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    answer = namespace.get("answer", None)
    if answer is None:
        return {"error": "no `answer` variable set by the snippet"}

    return {"result": _to_jsonable(answer)}


def _to_jsonable(value: Any) -> Any:
    """Convert pandas / numpy objects into JSON-serialisable forms."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return value if isinstance(value, (str, int, float, bool, list, dict)) \
                     else str(value)

    if isinstance(value, pd.DataFrame):
        df = value.head(50).replace({float("inf"): None, float("-inf"): None})
        df = df.where(pd.notna(df), None)
        return {
            "type":    "dataframe",
            "columns": list(df.columns),
            "rows":    df.to_dict(orient="records"),
            "row_count": len(df),
        }
    if isinstance(value, pd.Series):
        s = value.head(50)
        return {"type": "series", "name": s.name, "values": s.tolist()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value[:50].tolist()
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(x) for x in list(value)[:50]]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in list(value.items())[:50]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS: Dict[str, ToolSpec] = {
    "load_preview_data": ToolSpec(
        name="load_preview_data",
        description=(
            "Examine the structure of the user's uploaded workbook. Returns "
            "a per-sheet preview with column names, DuckDB dtypes, and the "
            "first 3 rows. Call this FIRST before constructing any query so "
            "you know the exact table and column names available."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        executor=_exec_load_preview_data,
    ),

    "complex_duckdb_query": ToolSpec(
        name="complex_duckdb_query",
        description=(
            "Execute a DuckDB SQL query against the session's registered "
            "tables. Use this for filtering, joining across sheets, "
            "aggregating, ranking, and any operation that's natural in SQL. "
            "Always double-quote table and column names that contain spaces "
            "or mixed case, e.g. SELECT \"A SpC\" FROM \"proteins\"."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A complete DuckDB SQL statement (no semicolons mid-query)."
                }
            },
            "required": ["query"],
        },
        executor=_exec_complex_duckdb_query,
    ),

    "simple_dataframe_query": ToolSpec(
        name="simple_dataframe_query",
        description=(
            "Run a small pandas snippet against the primary protein sheet "
            "for operations awkward in SQL (custom string parsing, regex, "
            "complex multi-step transforms). Variables available: `df`, "
            "`df_spc`, `df_intensity`, `sheets`, `sample_map`, `pd`, `np`, "
            "and these deterministic helpers: safe_fold_change, "
            "get_gene_symbol, format_protein_row, get_nonstandard_protein, "
            "top_n_by_metric, detect_metric_columns. Assign the final "
            "answer to a variable named `answer`."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Pandas/numpy snippet that ends by setting `answer`."
                }
            },
            "required": ["query"],
        },
        executor=_exec_simple_dataframe_query,
    ),
}


def get_openai_tool_specs() -> List[Dict[str, Any]]:
    """Return the tool registry as a list of OpenAI function-call specs."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOLS.values()
    ]


def execute_tool_call(
    name: str,
    arguments_json: str,
    context: Dict[str, Any],
) -> str:
    """Dispatch a tool call by name. Returns a JSON string the LLM can read.

    `arguments_json` is the raw arguments string from the LLM (already JSON).
    `context` carries system fields (session_id, namespace) that the LLM
    never sees — merged into args before the executor is called.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool: {name!r}"})
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid tool arguments JSON: {exc}"})
    merged = {**args, **context}
    try:
        result = tool.executor(merged)
    except Exception as exc:
        logger.warning("Tool %s executor raised: %s", name, exc)
        result = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        return json.dumps(result, default=str)
    except Exception:
        return json.dumps({"error": "tool result not JSON-serialisable",
                           "repr": str(result)[:500]})
