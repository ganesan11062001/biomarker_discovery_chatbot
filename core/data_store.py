"""
core/data_store.py
Per-session DuckDB data store, inspired by ExcelWorkerLLMToolCallAgent
(jenyss/ExcelWorkerLLMToolCallAgent) and adapted for our session manager.

When a user uploads a workbook, every sheet (already parsed into a pandas
DataFrame by DataLoadingSkill) is registered as a DuckDB table named after
the sheet. The LLM can then issue SQL queries against those tables instead
of writing imperative pandas. SQL has three big wins here:

  1. Column names with spaces / mixed case are handled with quoting —
     SQL `"some col name"` never raises a NameError the way pandas attribute
     access does.
  2. The LLM gets a stable schema dump (`SHOW COLUMNS`, `DESCRIBE`) to
     ground its query against — no more "is the column called 'Spectral
     Count' or 'A SpC'?" guesswork.
  3. Joins across the Identifier-Info / Proteins / Modifications sheets
     come for free.

We keep the pandas-based query_data path as a fallback (and for cell-level
operations SQL can't express well). This module just provides a stable,
schema-introspectable layer on top of `all_sheets`.

Public API
----------
get_store(session_id)             — per-session DuckDB connection + tables
register_sheets(session_id, …)    — register a {name: DataFrame} dict
schema_text(session_id)           — render CREATE TABLE-like schema for prompts
query(session_id, sql)            — execute SQL, return DataFrame or error
clear(session_id)                 — drop all tables for the session
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy DuckDB import — keep core lightweight, fail gracefully when missing.
try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore
    _DUCKDB_AVAILABLE = False
    logger.warning("duckdb not installed — SQL queries will be unavailable.")


def is_available() -> bool:
    """True when DuckDB is importable. Callers should check before using SQL."""
    return _DUCKDB_AVAILABLE


_VALID_TABLE_NAME = re.compile(r"[^A-Za-z0-9_]+")


def _sanitize_table_name(name: str) -> str:
    """Map any sheet name to a valid SQL identifier.
    'Identifier Info' → 'identifier_info'; '__Proteins!' → 'proteins'."""
    cleaned = _VALID_TABLE_NAME.sub("_", name.strip()).strip("_").lower()
    if not cleaned:
        cleaned = "sheet"
    # Avoid SQL keywords as table names by prefixing if needed
    if cleaned in {"select", "from", "where", "table", "order", "group"}:
        cleaned = f"t_{cleaned}"
    return cleaned


@dataclass
class _SessionStore:
    """Holds the DuckDB connection + the table-name mapping for one session."""
    con:               object                    # duckdb.DuckDBPyConnection
    table_names:       Dict[str, str]            = field(default_factory=dict)
    # Sanitised SQL identifier  →  original sheet name (for prompt clarity)
    original_names:    Dict[str, str]            = field(default_factory=dict)


_stores: Dict[str, _SessionStore] = {}
_lock = threading.RLock()


# ── Public API ────────────────────────────────────────────────────────────────

def get_store(session_id: str) -> Optional[_SessionStore]:
    """Return the existing store for a session, or None if not yet built."""
    if not _DUCKDB_AVAILABLE:
        return None
    with _lock:
        return _stores.get(session_id)


def register_sheets(
    session_id: str, sheets: Dict[str, "pd.DataFrame"],  # noqa: F821
) -> Optional[_SessionStore]:
    """
    Register a session's sheets as DuckDB tables.

    Returns the SessionStore (with `table_names` populated) or None if DuckDB
    is unavailable. Safe to call multiple times — re-registration drops
    existing tables first.
    """
    if not _DUCKDB_AVAILABLE:
        logger.debug("register_sheets: DuckDB not available; skipping.")
        return None

    import pandas as pd

    if not sheets:
        return None

    with _lock:
        store = _stores.get(session_id)
        if store is None:
            store = _SessionStore(con=duckdb.connect(":memory:"))
            _stores[session_id] = store
        else:
            # Drop existing tables so we re-register fresh data
            for sql_name in list(store.table_names.values()):
                try:
                    store.con.execute(f'DROP TABLE IF EXISTS "{sql_name}"')
                except Exception as exc:
                    logger.debug("Drop %s failed: %s", sql_name, exc)
            store.table_names.clear()
            store.original_names.clear()

        # Register each sheet
        for sheet_name, df in sheets.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            sql_name = _sanitize_table_name(sheet_name)

            # Ensure uniqueness if two sheets sanitise to the same name
            base = sql_name
            counter = 2
            while sql_name in store.table_names.values():
                sql_name = f"{base}_{counter}"
                counter += 1

            try:
                # Normalise: empty strings → NULL so SQL aggregates behave
                df_clean = df.replace(r"^\s*$", None, regex=True)
                store.con.register(f"_tmp_{sql_name}", df_clean)
                store.con.execute(
                    f'CREATE OR REPLACE TABLE "{sql_name}" AS '
                    f'SELECT * FROM _tmp_{sql_name}'
                )
                store.con.unregister(f"_tmp_{sql_name}")
                store.table_names[sheet_name] = sql_name
                store.original_names[sql_name] = sheet_name
            except Exception as exc:
                logger.warning("Could not register sheet %r as DuckDB table: %s",
                               sheet_name, exc)

        logger.info(
            "DuckDB registered %d table(s) for session %s: %s",
            len(store.table_names), session_id[:8], list(store.table_names.values()),
        )
        return store


def schema_text(session_id: str, sample_rows: int = 3) -> str:
    """
    Render a markdown schema description suitable for LLM prompts.

    Looks like:

      ### Table: `proteins`  (original sheet: 'Proteins')
      shape: 1919 × 24
      columns:
        - "Protein IDs" (VARCHAR)
        - "A SpC" (DOUBLE)
        - ...
      first 3 rows:
        | Protein IDs | A SpC | ... |
    """
    store = get_store(session_id)
    if store is None:
        return "_(DuckDB not initialised for this session — no SQL schema available.)_"

    blocks: List[str] = []
    for sql_name in store.table_names.values():
        try:
            orig    = store.original_names.get(sql_name, sql_name)
            row     = store.con.execute(f'SELECT COUNT(*) FROM "{sql_name}"').fetchone()
            n_rows  = int(row[0]) if row else 0
            cols    = store.con.execute(f'DESCRIBE "{sql_name}"').fetchall()
            preview = store.con.execute(
                f'SELECT * FROM "{sql_name}" LIMIT {sample_rows}'
            ).fetchdf()

            cols_block = "\n".join(f'  - "{c[0]}" ({c[1]})' for c in cols[:50])
            if len(cols) > 50:
                cols_block += f"\n  …(+{len(cols)-50} more columns)"
            # CSV avoids the tabulate dependency that to_markdown needs
            preview_csv = preview.to_csv(index=False)
            if len(preview_csv) > 800:
                preview_csv = preview_csv[:800] + "\n…[truncated]"

            blocks.append(
                f"### Table: `{sql_name}`  (sheet: {orig!r})\n"
                f"shape: {n_rows} × {len(cols)}\n"
                f"columns:\n{cols_block}\n"
                f"first {sample_rows} rows (CSV):\n{preview_csv}"
            )
        except Exception as exc:
            logger.debug("schema_text failed for %s: %s", sql_name, exc)
    return "\n\n".join(blocks) if blocks else "_(no tables registered)_"


def query(session_id: str, sql: str, max_rows: int = 200) -> Tuple[object, Optional[str]]:
    """
    Execute SQL against the session's DuckDB store.

    Returns (result_dataframe, error_or_none). The result is capped at
    `max_rows` to keep responses bounded. The caller decides how to
    serialise it for the LLM.
    """
    store = get_store(session_id)
    if store is None:
        return None, "DuckDB store not initialised for this session."

    try:
        df = store.con.execute(sql).fetchdf()
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        return df, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def clear(session_id: str) -> None:
    """Drop the session's DuckDB connection and tables."""
    with _lock:
        store = _stores.pop(session_id, None)
        if store is not None:
            try:
                store.con.close()
            except Exception:
                pass
