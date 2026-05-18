"""
skills/canonical_loader.py
Canonical 2-sheet proteomics loader — the ONE supported format going forward.

  Sheet 1 (metadata):
    Sample ID | Group
    S1        | WT
    S2        | WT
    S3        | DMD
    S4        | DMD

  Sheet 2 (proteins):
    Protein Name | Accession Number | Gene Name | S1   | S2   | S3   | S4
    Myh4 OS=…    | Q5SX39           | Myh4      | 100  | 110  | 50   | 55
    …

What this loader does NOT do (intentionally):
  • No SpC / spectral-count handling.
  • No MaxQuant pooled-design auto-routing.
  • No SILAC ratio detection.
  • No locale-decimal acrobatics (we expect well-formed numerics).
  • No header-row guessing (header is row 0).

It produces a single clean ``LoadResult`` the BiomarkerAgent can consume.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Header-name hints. Case-insensitive, whitespace/underscore-tolerant.
_SAMPLE_ID_HINTS = ("sample id", "sampleid", "sample_id", "sample-id", "sample")
_GROUP_HINTS     = ("group", "condition", "treatment", "class", "phenotype", "cohort")
_PROTEIN_HINTS   = ("protein name", "protein names", "protein", "description")
_ACCESSION_HINTS = ("accession number", "accession", "uniprot", "protein id", "protein ids")
_GENE_HINTS      = ("gene name", "gene names", "gene symbol", "gene")


@dataclass
class LoadResult:
    """The single output of the canonical loader, consumed by analysis skills."""
    proteins_df:       pd.DataFrame                # rows = proteins; index = Accession
    metadata_df:       pd.DataFrame                # 2-col DataFrame: Sample_ID + Group
    sample_to_group:   Dict[str, str]              # column-name → group label
    group_to_samples:  Dict[str, List[str]]        # group label → list of column names
    protein_name_col:  str                         # actual column name in proteins_df
    accession_col:     str                         # actual column name (also the index)
    gene_col:          Optional[str]               # gene-symbol column when present
    sample_columns:    List[str]                   # all sample-value column names
    n_proteins:        int
    n_samples:         int
    groups:            List[str]                   # ordered list of unique groups
    organism:          Optional[str] = None        # auto-detected from OS= tags

    # Diagnostics + raw sheets (kept for question-answering / "show data")
    metadata_sheet_name:  str = ""
    proteins_sheet_name:  str = ""
    raw_sheets:           Dict[str, pd.DataFrame] = field(default_factory=dict)


# ── Header picking helpers ───────────────────────────────────────────────────

def _norm(s: object) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


def _pick(columns: List[str], hints: Tuple[str, ...]) -> Optional[str]:
    norm = {c: _norm(c) for c in columns}
    for c, n in norm.items():
        for h in hints:
            if _norm(h) in n:
                return c
    return None


# ── Public API ──────────────────────────────────────────────────────────────

class CanonicalLoaderError(ValueError):
    """Raised when the workbook does not match the canonical 2-sheet template."""


def load_canonical_workbook(path: str) -> LoadResult:
    """Load a 2-sheet proteomics workbook in the canonical template.

    Accepts:
      - .xlsx / .xls workbooks with at least 2 sheets
      - .csv files where the proteins sheet is given inline (no metadata sheet
        path → raises; users must provide both)

    Raises CanonicalLoaderError with a clear message if the template doesn't match.
    """
    p = Path(path)
    if not p.exists():
        raise CanonicalLoaderError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix not in (".xlsx", ".xls"):
        raise CanonicalLoaderError(
            f"Canonical template requires an Excel workbook (.xlsx/.xls) "
            f"with two sheets — got {suffix!r}."
        )

    raw_sheets = pd.read_excel(p, sheet_name=None)
    if len(raw_sheets) < 2:
        raise CanonicalLoaderError(
            f"Workbook has {len(raw_sheets)} sheet(s); the canonical "
            f"template requires two: metadata + proteins."
        )

    # Identify metadata sheet (has Sample ID + Group columns)
    meta_name, meta_df = _find_metadata_sheet(raw_sheets)
    if meta_df is None:
        raise CanonicalLoaderError(
            "Could not find a metadata sheet with both a 'Sample ID' and "
            "'Group' column. Sheets seen: "
            f"{list(raw_sheets.keys())}."
        )

    # Identify proteins sheet (has Protein Name + Accession + Gene)
    proteins_name, proteins_df = _find_proteins_sheet(raw_sheets, exclude=meta_name)
    if proteins_df is None:
        raise CanonicalLoaderError(
            "Could not find a proteins sheet with 'Protein Name', "
            "'Accession Number', and 'Gene Name' columns."
        )

    # Build the sample → group map from metadata
    sample_col   = _pick(list(meta_df.columns), _SAMPLE_ID_HINTS)
    group_col    = _pick(list(meta_df.columns), _GROUP_HINTS)
    # Ensure the picker didn't return the same column for both slots
    if sample_col is not None and sample_col == group_col:
        alt = [c for c in meta_df.columns if c != sample_col]
        group_col = _pick(alt, _GROUP_HINTS)
    if sample_col is None or group_col is None:
        raise CanonicalLoaderError(
            "Metadata sheet must have 'Sample ID' + 'Group' columns."
        )

    sample_to_group: Dict[str, str] = {}
    group_to_samples: Dict[str, List[str]] = {}
    for _, row in meta_df.iterrows():
        sid = row.get(sample_col)
        grp = row.get(group_col)
        if pd.isna(sid) or pd.isna(grp):
            continue
        sid_str = str(sid).strip()
        grp_str = str(grp).strip()
        if not sid_str or not grp_str:
            continue
        sample_to_group[sid_str] = grp_str
        group_to_samples.setdefault(grp_str, []).append(sid_str)

    if not group_to_samples:
        raise CanonicalLoaderError(
            "Metadata sheet has no usable Sample ID / Group rows."
        )

    # Resolve identifier columns in the proteins sheet
    protein_col = _pick(list(proteins_df.columns), _PROTEIN_HINTS)
    acc_col     = _pick(list(proteins_df.columns), _ACCESSION_HINTS)
    gene_col    = _pick(list(proteins_df.columns), _GENE_HINTS)

    if acc_col is None:
        raise CanonicalLoaderError(
            "Proteins sheet must include an 'Accession Number' column."
        )
    if protein_col is None:
        raise CanonicalLoaderError(
            "Proteins sheet must include a 'Protein Name' column."
        )

    # Identify which proteins-sheet columns correspond to the metadata's
    # Sample IDs. We match by exact string equality after stripping.
    proteins_cols      = [str(c).strip() for c in proteins_df.columns]
    proteins_df.columns = proteins_cols
    expected_sample_ids = set(sample_to_group.keys())
    sample_columns      = [c for c in proteins_cols if c in expected_sample_ids]

    if not sample_columns:
        raise CanonicalLoaderError(
            "No proteins-sheet columns match the Sample IDs from the metadata "
            f"sheet. Metadata samples: {sorted(expected_sample_ids)[:10]}…  "
            f"Proteins columns: {proteins_cols[:10]}…"
        )

    missing_samples = expected_sample_ids - set(sample_columns)
    if missing_samples:
        logger.warning(
            "Metadata sheet declares %d sample(s) not present in the proteins "
            "sheet: %s — those rows will be dropped from the group map.",
            len(missing_samples), sorted(missing_samples),
        )
        # Drop the unresolvable samples from both maps so analysis sees only
        # what actually exists.
        for s in missing_samples:
            g = sample_to_group.pop(s, None)
            if g and s in group_to_samples.get(g, []):
                group_to_samples[g].remove(s)
        # Drop groups that became empty
        group_to_samples = {g: v for g, v in group_to_samples.items() if v}

    # Coerce sample columns to numeric — drop any stray strings as NaN
    for c in sample_columns:
        proteins_df[c] = pd.to_numeric(proteins_df[c], errors="coerce")

    # Set Accession as the row index for fast lookups + Plotly labels
    proteins_df = proteins_df.set_index(acc_col, drop=False)
    proteins_df.index.name = acc_col

    # Detect organism from the OS= tag in the protein-name column
    organism = _detect_organism(proteins_df[protein_col])

    return LoadResult(
        proteins_df       = proteins_df,
        metadata_df       = meta_df[[sample_col, group_col]].rename(
                                columns={sample_col: "Sample ID", group_col: "Group"}),
        sample_to_group   = sample_to_group,
        group_to_samples  = group_to_samples,
        protein_name_col  = protein_col,
        accession_col     = acc_col,
        gene_col          = gene_col,
        sample_columns    = sample_columns,
        n_proteins        = len(proteins_df),
        n_samples         = len(sample_columns),
        groups            = sorted(group_to_samples.keys()),
        organism          = organism,
        metadata_sheet_name = meta_name,
        proteins_sheet_name = proteins_name,
        raw_sheets        = raw_sheets,
    )


# ── Internals ───────────────────────────────────────────────────────────────

def _find_metadata_sheet(
    sheets: Dict[str, pd.DataFrame],
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Scan every sheet for one that has BOTH a Sample-ID column and a Group column."""
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        cols = [str(c).strip() for c in df.columns]
        s = _pick(cols, _SAMPLE_ID_HINTS)
        g = _pick(cols, _GROUP_HINTS)
        if s and g and s != g:
            df = df.copy()
            df.columns = cols
            return name, df
    return "", None


def _find_proteins_sheet(
    sheets: Dict[str, pd.DataFrame],
    exclude: str,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Among the remaining sheets, find one with both a Protein column and an Accession column."""
    for name, df in sheets.items():
        if name == exclude or df is None or df.empty:
            continue
        cols = [str(c).strip() for c in df.columns]
        if _pick(cols, _PROTEIN_HINTS) and _pick(cols, _ACCESSION_HINTS):
            df = df.copy()
            df.columns = cols
            return name, df
    return "", None


_OS_RE = re.compile(r"OS=([A-Z][a-z]+(?:\s+[a-z]+)+?)(?=\s+(?:OX|GN|PE|SV|=)|$)")


def _detect_organism(name_series: pd.Series) -> Optional[str]:
    """Scan the first ~200 protein names for `OS=<Genus species>` and return the modal hit."""
    from collections import Counter
    counts: Counter = Counter()
    for v in name_series.dropna().astype(str).head(200):
        m = _OS_RE.search(v)
        if m:
            counts[m.group(1).strip()] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
