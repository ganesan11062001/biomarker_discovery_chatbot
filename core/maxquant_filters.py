"""
core/maxquant_filters.py
MaxQuant / proteomics data-cleaning primitives.

Combines the proven patterns from four open-source projects:
  • pymaxquant      — boolean '+' column filters (Reverse, Potential
                       contaminant, Only identified by site)
  • mspypeline      — locale-aware decimal handling, NaN-to-zero
                       normalisation, column-name string cast
  • alphapeptstats  — explicit metric-prefix detection (LFQ / iBAQ /
                       Intensity); flag-as-column rather than drop
  • autoprot        — accession-prefix flagging (CON__ / REV__)

Design choices:
  * We never DELETE contaminant rows — instead we ADD boolean flag columns
    (is_reverse, is_potential_contaminant, is_only_identified_by_site,
    is_contaminant_accession, is_contaminant). Downstream analysis skills
    can `df[~df.is_contaminant]`, and ad-hoc user queries can still ask
    "how many contaminants are in the file?".
  * Every helper is data-agnostic — no specific protein names, accessions,
    or sample codes baked in.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── pymaxquant-style flag columns ────────────────────────────────────────────
# MaxQuant marks special rows with '+' in dedicated columns. These names are
# the canonical MaxQuant headers; we match case-insensitively.

_REVERSE_COL_HINTS         = ("reverse",)
_CONTAMINANT_COL_HINTS     = ("potential contaminant", "contaminant")
_SITE_ONLY_COL_HINTS       = ("only identified by site",)

# Common contaminant / decoy accession prefixes used by MaxQuant, FragPipe,
# Proteome Discoverer, etc. Anything starting with these is flagged.
_CONTAMINANT_ACC_PREFIXES  = ("CON__", "CON_", "REV__", "REV_", "##", "DECOY_")


def _find_column(columns: Sequence[str], hints: Sequence[str]) -> Optional[str]:
    """First column whose case-folded name contains any of the hint substrings."""
    for col in columns:
        cl = str(col).lower()
        for h in hints:
            if h in cl:
                return col
    return None


# ── pymaxquant-style: filter rows where a marker column contains '+' ─────────

def remove_columns_containing(df, column: str, match: str = "+"):
    """Return a copy of `df` with rows where `column` contains `match` removed.

    Direct port of pymaxquant.filters.remove_columns_containing.
    """
    df = df.copy()
    mask = df[column].astype(str).str.contains(re.escape(match), na=False)
    return df.loc[~mask].reset_index(drop=True)


def remove_reverse(df):
    """Drop rows flagged as reverse decoys (Reverse column == '+')."""
    col = _find_column(df.columns, _REVERSE_COL_HINTS)
    if col is None:
        return df
    return remove_columns_containing(df, col)


def remove_potential_contaminants(df):
    """Drop rows flagged as contaminants ('Potential contaminant' == '+')."""
    col = _find_column(df.columns, _CONTAMINANT_COL_HINTS)
    if col is None:
        return df
    return remove_columns_containing(df, col)


def remove_only_identified_by_site(df):
    """Drop rows flagged as site-only IDs ('Only identified by site' == '+')."""
    col = _find_column(df.columns, _SITE_ONLY_COL_HINTS)
    if col is None:
        return df
    return remove_columns_containing(df, col)


# ── Flag (don't delete) — alphapeptstats-style ───────────────────────────────

def flag_maxquant_contaminants(df):
    """
    Add boolean flag columns to `df` instead of dropping rows. Returns a new
    DataFrame with these extra columns when the source markers exist:

        is_reverse                   — Reverse == '+'
        is_potential_contaminant     — Potential contaminant == '+'
        is_only_identified_by_site   — Only identified by site == '+'
        is_contaminant_accession     — accession starts with CON__/REV__/##/DECOY_
        is_contaminant               — logical OR of the above

    Missing markers don't raise — the corresponding flag column is simply
    skipped. Downstream code should check column presence before filtering.
    """
    import pandas as pd

    out = df.copy()

    def _mark(col_hint_tokens: Tuple[str, ...], flag_name: str) -> None:
        src = _find_column(out.columns, col_hint_tokens)
        if src is not None:
            out[flag_name] = out[src].astype(str).str.contains(r"\+", na=False)
        else:
            out[flag_name] = False

    _mark(_REVERSE_COL_HINTS,     "is_reverse")
    _mark(_CONTAMINANT_COL_HINTS, "is_potential_contaminant")
    _mark(_SITE_ONLY_COL_HINTS,   "is_only_identified_by_site")

    # Accession-prefix-based contaminant flagging (autoprot pattern)
    acc_col = _find_column(out.columns, ("accession", "protein id", "protein ids", "uniprot"))
    if acc_col is not None:
        prefixes = tuple(_CONTAMINANT_ACC_PREFIXES)
        out["is_contaminant_accession"] = out[acc_col].astype(str).str.startswith(prefixes)
    else:
        out["is_contaminant_accession"] = False

    out["is_contaminant"] = (
        out["is_reverse"]
        | out["is_potential_contaminant"]
        | out["is_only_identified_by_site"]
        | out["is_contaminant_accession"]
    )
    return out


def remove_all_contaminants(df):
    """Apply all three pymaxquant filters in one call.

    Returns a DataFrame with reverse decoys, potential contaminants, and
    site-only IDs dropped. Use `flag_maxquant_contaminants` instead when
    you want to KEEP these rows for queryability but mark them.
    """
    df = remove_reverse(df)
    df = remove_potential_contaminants(df)
    df = remove_only_identified_by_site(df)
    return df


# ── mspypeline-style: numeric & locale normalisation ────────────────────────

def normalise_column_names(df):
    """Cast every column name to a stripped string. Returns a new DataFrame.

    Excel and CSV exports occasionally produce mixed-type columns (numbers,
    bytes, NaN); SQL / ILIKE / regex matching all break unless headers are
    plain strings.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fix_locale_decimals(df, columns: Optional[Iterable[str]] = None):
    """
    Replace ',' with '.' in numeric-looking string columns and coerce to
    float. European MaxQuant exports often write '0,42' instead of '0.42'.

    If `columns` is None, every column that's NOT already numeric is
    attempted (best-effort coercion via pd.to_numeric).
    """
    import pandas as pd

    df = df.copy()
    target_cols = list(columns) if columns is not None else [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in target_cols:
        if col not in df.columns:
            continue
        try:
            as_str = df[col].astype(str).str.replace(",", ".", regex=False)
            coerced = pd.to_numeric(as_str, errors="coerce")
            # Only adopt the coerced version if MOST cells parsed (>=70%)
            if coerced.notna().sum() >= 0.7 * len(coerced):
                df[col] = coerced
        except Exception as exc:
            logger.debug("fix_locale_decimals skipped %r: %s", col, exc)
    return df


def coerce_metric_columns_numeric(df, metric_cols: Iterable[str]):
    """Force the named metric columns to float, replacing un-parseable values
    with NaN. Use this on SpC / Intensity / LFQ / iBAQ columns immediately
    after load."""
    import pandas as pd

    df = df.copy()
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── Schema validation — alphapeptstats _check_if_columns_are_present ────────

class RequiredColumnsMissing(KeyError):
    """Raised when a workbook lacks columns required for downstream analysis."""


def assert_columns_present(df, required: Iterable[str]) -> None:
    """
    Raise RequiredColumnsMissing if any required column is absent.

    Required column names are matched case- and whitespace-insensitively so
    the caller doesn't have to know the exact header capitalisation.
    """
    have = {str(c).strip().lower() for c in df.columns}
    missing = [c for c in required
               if str(c).strip().lower() not in have]
    if missing:
        raise RequiredColumnsMissing(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)[:20]}…"
        )


# ── End-to-end MaxQuant-style cleanup pipeline ──────────────────────────────

def apply_standard_cleanup(
    df,
    drop_contaminants: bool = False,
    fix_decimals: bool = True,
    flag_contaminants: bool = True,
):
    """
    Run the standard MaxQuant cleanup pipeline in one call.

    Pipeline order:
      1. Normalise column names to stripped strings.
      2. Optionally fix locale decimal commas → periods.
      3. Either FLAG contaminants (is_contaminant column) or DROP them.

    Returns the cleaned DataFrame. The caller decides whether to drop or
    flag — flagging is recommended so users can still ask "how many
    contaminants are in the file?".
    """
    df = normalise_column_names(df)
    if fix_decimals:
        df = fix_locale_decimals(df)
    if drop_contaminants:
        df = remove_all_contaminants(df)
    elif flag_contaminants:
        df = flag_maxquant_contaminants(df)
    return df
