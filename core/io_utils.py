"""
core/io_utils.py
Encoding-robust file I/O helpers used by skills and agents.

Why this exists
---------------
Proteomics CSVs from MaxQuant / FragPipe / Excel-on-Windows commonly contain
non-ASCII characters in headers — most often µ (0xb5, micro) but also ≥, °,
β, α, Greek letters, etc. They are usually encoded in cp1252 (Windows-1252)
or latin-1, not utf-8. A bare ``pd.read_csv(path)`` defaults to utf-8 and
crashes with ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb5``
on the first such character.

This module centralises the encoding-fallback chain so every place that
re-loads a user CSV (analysis skill, visualisation skill, dual-engine,
enrichment background) gets the same robust behaviour.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Order matters:
#   utf-8      — the modern default; pass through with no surprises
#   utf-8-sig  — utf-8 with BOM (Excel sometimes writes this)
#   cp1252     — Windows-1252; what Excel-on-Windows writes by default
#   latin-1    — last-resort superset that always decodes (no errors)
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_csv_safe(path: str, **kwargs: Any) -> pd.DataFrame:
    """``pd.read_csv`` with automatic encoding fallback.

    If the caller passes ``encoding=...`` explicitly we honour it. Otherwise
    we try utf-8 → utf-8-sig → cp1252 → latin-1 in order. latin-1 can decode
    any byte string so this chain is guaranteed to succeed eventually.

    All other read_csv kwargs (index_col, header, usecols, nrows, etc.) are
    forwarded unchanged.
    """
    if "encoding" in kwargs:
        return pd.read_csv(path, **kwargs)

    last_exc: Exception | None = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            if enc != "utf-8":
                logger.info("read_csv_safe: '%s' decoded as %s", path, enc)
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue

    # Should be unreachable because latin-1 never raises UnicodeDecodeError,
    # but keep a defensive fallback that won't lose the original error.
    raise last_exc if last_exc is not None else RuntimeError(
        f"Could not decode CSV '{path}' with any encoding."
    )
