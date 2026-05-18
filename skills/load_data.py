"""
skills/load_data.py
Data Layer — DataLoadingSkill

Reads every sheet in an uploaded Excel workbook (or a single CSV), classifies
each sheet automatically, and returns:

  • The primary expression matrix (proteins × samples) as a processed CSV.
  • A dict of ALL parsed sheets so no data is discarded.
  • A label_map and identifier_info extracted from any metadata sheet.

Sheet classification (applied to every sheet)
----------------------------------------------
  EXPRESSION  — many rows (≥ 20), majority of columns are numeric.
                Used as the protein-abundance matrix.
  METADATA    — fewer rows or mostly text.  Contains sample / group info.

The first EXPRESSION sheet found becomes the primary data source.
All METADATA sheets are parsed, cleaned, and stored under ``all_sheets``.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_PROTEIN_ID_PATTERNS = [
    re.compile(r"^(P|Q|O)\d{5}", re.I),
    re.compile(r"\w{2,}_HUMAN$", re.I),
    re.compile(r"^ENSP\d+", re.I),
    re.compile(r"^sp\|", re.I),
]

_PROTEIN_ID_COLUMNS = {
    "majority protein ids", "protein ids", "uniprot",
    "accession", "accession number",
    "gene name", "gene names", "protein name", "protein names",
    "identified proteins", "id", "ids",
}

_METADATA_HINTS = {
    "group", "condition", "sample", "label", "type", "class",
    "patient", "disease", "treatment", "control", "status",
    "gender", "sex", "age", "batch", "time", "timepoint",
    "subject", "donor", "cohort", "replicate",
}

# Tokens that suggest a row is the real column-header row in a proteomics sheet.
# Includes identifier columns, metric prefixes, and platform-specific names.
_HEADER_ROW_TOKENS = (
    # Identifiers
    "protein", "accession", "uniprot", "gene", "description", "fasta",
    "name", "id", "ids",
    # Metrics (MaxQuant / FragPipe / Spectronaut / DIA-NN / TMT / SILAC)
    "intensity", "spc", "spectral", "lfq", "ibaq", "ratio", "abundance",
    "quantity", "channel", "plex",
    # Physical properties
    "mw", "weight", "molecular", "pi",
    # Platform-specific
    "npx", "assay", "panel", "olink",
    # Statistics
    "fdr", "qvalue", "q-value", "pep", "score", "p-value", "p_value", "pvalue",
    # Sample-axis tokens
    "sample", "subject", "donor", "patient",
)


def _extract_column_group_labels(
    raw: pd.DataFrame, header_idx: int,
) -> Dict[str, str]:
    """When a proteomics sheet has group labels in the row ABOVE the real
    header (e.g. row 0 = 'BL6 Quad', 'DMD Quad', ... above row 1 = 'SpC A',
    'SpC B', ...), pair them up.

    Returns {real_column_name: group_label} for every column where the
    immediately-preceding row has a distinct non-empty label. Empty when no
    such pattern exists (most files won't have it).
    """
    if header_idx <= 0 or header_idx >= len(raw):
        return {}
    label_row  = raw.iloc[header_idx - 1]
    header_row = raw.iloc[header_idx]
    mapping: Dict[str, str] = {}
    for i, header_val in enumerate(header_row.values):
        if header_val is None or (isinstance(header_val, float) and pd.isna(header_val)):
            continue
        label_val = label_row.iloc[i] if i < len(label_row) else None
        if label_val is None or (isinstance(label_val, float) and pd.isna(label_val)):
            continue
        h, l = str(header_val).strip(), str(label_val).strip()
        if not h or not l or h == l:
            continue
        # Skip identifier columns whose label is just a section heading (e.g.
        # column "Accession Number" with no real "group" above it). We only
        # care when both header and label look like distinct meaningful strings.
        mapping[h] = l
    return mapping


def _detect_header_row(
    raw: pd.DataFrame,
    max_scan: int = 12,
    min_non_empty: int = 2,
) -> int:
    """Find the most-likely header row by scanning the first ``max_scan`` rows.

    Many proteomics exports put a title row above the real headers
    (e.g. ``Identified Proteins (1919)`` followed by ``Protein Name,
    Accession Number, A SpC, ...``). pandas' default ``header=0`` then
    treats the real headers as data.

    Algorithm: for each candidate row, count cells whose lowercased text
    contains one of ``_HEADER_ROW_TOKENS``. The row with the highest hit
    count wins. Falls back to 0 when no row matches.
    """
    if raw is None or raw.empty:
        return 0

    best_idx, best_score = 0, 0
    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i]
        non_empty = 0
        score     = 0
        for val in row.values:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val).strip().lower()
            if not s:
                continue
            non_empty += 1
            if any(tok in s for tok in _HEADER_ROW_TOKENS):
                score += 1
        # A header row must have at least 2 non-empty cells AND at least one
        # header-token match. Title rows (e.g. "Identified Proteins (1919)")
        # have non_empty == 1 and won't beat a multi-cell header row.
        if non_empty >= min_non_empty and score > best_score:
            best_idx, best_score = i, score
    return best_idx

# A sheet is EXPRESSION if it has ≥ this many rows
_EXPRESSION_MIN_ROWS = 20
# … and ≥ this fraction of its columns are numeric
_EXPRESSION_MIN_NUMERIC_FRAC = 0.4


# ── Sheet classification ───────────────────────────────────────────────────────

def _classify_sheet(df: pd.DataFrame) -> str:
    """Return 'expression' or 'metadata' for a raw sheet DataFrame."""
    if len(df) < _EXPRESSION_MIN_ROWS:
        return "metadata"
    numeric_cols = sum(
        1 for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.6
    )
    frac = numeric_cols / max(len(df.columns), 1)
    return "expression" if frac >= _EXPRESSION_MIN_NUMERIC_FRAC else "metadata"


# ── Column-name normalisation ─────────────────────────────────────────────────

_TITLE_COL_PATTERN = re.compile(r"^Identified\s+Proteins?\s*(\(\s*\d+\s*\))?\s*$",
                                re.IGNORECASE)


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename / strip whitespace from columns so the LLM sees stable names.

    - Trims leading/trailing whitespace on every column (`'Protein Name '` →
      `'Protein Name'`).
    - Renames title-style identifier columns (`Identified Proteins (2217)`)
      to the canonical `'Protein Name'` so the LLM doesn't have to guess
      which column holds the UniProt-style description.
    """
    new_columns: Dict[str, str] = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        clean = c.strip()
        if _TITLE_COL_PATTERN.match(clean):
            clean = "Protein Name"
        if clean != c:
            new_columns[c] = clean
    if new_columns:
        df = df.rename(columns=new_columns)
        logger.debug("Canonicalised %d column name(s): %s",
                      len(new_columns), new_columns)
    return df


# ── Column helpers ─────────────────────────────────────────────────────────────

def _is_protein_id_column(col_name: str) -> bool:
    name = col_name.strip().lower()
    # Exact match
    if name in _PROTEIN_ID_COLUMNS:
        return True
    # Partial match for columns like "Identified Proteins (1919)"
    return any(kw in name for kw in ("majority protein", "accession", "identified protein"))


def _looks_like_protein_index(index) -> bool:
    sample = [str(x) for x in list(index)[:20]]
    hits = sum(1 for s in sample if any(p.search(s) for p in _PROTEIN_ID_PATTERNS))
    return hits / max(len(sample), 1) >= 0.3


def _detect_data_type(df: pd.DataFrame, sample_cols: List[str]) -> str:
    combined = [c.lower() for c in df.columns] + [str(i).lower() for i in df.index]
    if any("npx" in s or "olink" in s for s in combined):
        return "olink_npx"
    if _is_spectronaut(df):
        return "dia_spectronaut"
    if _is_silac(df):
        return "silac_ratio"
    if any(kw in s for s in combined
           for kw in ("intensity", "lfq", "tmt", "itraq", "ms1", "spectral", "spc")):
        return "ms_lfq"
    if not sample_cols:
        return "generic"
    num = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    mx, mn = num.max().max(), num.min().min()
    if pd.notna(mx):
        if -5 <= float(mn) and float(mx) <= 20:
            return "olink_npx"
        if float(mx) > 1_000:
            return "ms_lfq"
    return "generic"


def _is_spectronaut(df: pd.DataFrame) -> bool:
    """Return True when the DataFrame looks like a Spectronaut wide-format export."""
    col_str = " ".join(str(c) for c in df.columns)
    spectronaut_markers = ("PG.Genes", "PG.ProteinAccessions", "PG.UniProtIds",
                           "PG.Quantity", "EG.Sequence", "R.Condition")
    return (
        any(m in col_str for m in spectronaut_markers)
        or sum(1 for c in df.columns if "PG." in str(c) or "EG." in str(c)) >= 2
    )


def _is_silac(df: pd.DataFrame) -> bool:
    """Return True when column names suggest SILAC ratio data."""
    _kw = ("ratio h/l", "ratio h/m", "ratio m/l", "h/l ratio", "silac ratio",
           "normalized ratio", "ratio hl", "ratio hm")
    col_lower = " ".join(str(c).lower() for c in df.columns)
    return any(kw in col_lower for kw in _kw)


def _parse_spectronaut(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape a Spectronaut wide-format export into a standard proteins × samples matrix.

    Spectronaut typically has:
      • PG.Genes / PG.ProteinAccessions / PG.UniProtIds — protein identifier columns
      • [SampleName].PG.Quantity — per-sample intensity columns
      • Other metadata columns (PG.Cscore, EG.*, R.*)
    """
    # Find protein ID column
    id_candidates = [c for c in df_raw.columns
                     if any(m in str(c) for m in
                            ("PG.Genes", "PG.ProteinAccessions", "PG.UniProtIds",
                             "Protein.Ids", "Protein IDs"))]
    id_col = id_candidates[0] if id_candidates else df_raw.columns[0]

    # Find quantity / expression columns
    qty_cols = [
        c for c in df_raw.columns
        if c != id_col and (
            "PG.Quantity" in str(c)
            or "Quantity" in str(c)
            or (pd.to_numeric(df_raw[c], errors="coerce").notna().mean() >= 0.6
                and "PG." not in str(c) and "EG." not in str(c) and "R." != str(c)[:2])
        )
    ]
    if not qty_cols:
        qty_cols = [c for c in df_raw.columns if c != id_col]

    df_expr = df_raw[[id_col] + qty_cols].copy()
    df_expr = df_expr.set_index(id_col)
    df_expr.index.name = "Protein"

    # Clean column names: strip ".PG.Quantity" suffix and brackets
    df_expr.columns = [
        str(c).replace(".PG.Quantity", "").replace("[", "").replace("]", "").strip()
        for c in df_expr.columns
    ]
    logger.info(
        "Spectronaut: parsed %d proteins × %d samples from wide-format export.",
        len(df_expr), len(df_expr.columns),
    )
    return df_expr


def _detect_tmt_batches(
    sample_cols: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Detect multi-plex TMT structure from column names.

    Recognised patterns
    -------------------
    • Plex1_ChannelName, TMT2_Sample, Batch3_Ref
    • P1_Sample, B2_Reference
    Reference channels are identified by keywords: ref, pool, reference, standard.

    Returns {plex_name: {"samples": [...], "reference": col_or_None}} or None.
    """
    batch_pat = re.compile(
        r"^((?:Plex|TMT|Batch|Plx|plex|tmt|batch|BN)\s*\d+)[_\-\s](.+)$",
        re.IGNORECASE,
    )
    batches: Dict[str, List[str]] = {}
    for col in sample_cols:
        m = batch_pat.match(col)
        if m:
            batches.setdefault(m.group(1), []).append(col)

    if len(batches) < 2:
        return None

    ref_kw = ("ref", "pool", "reference", "ctrl", "standard", "iref")
    result: Dict[str, Any] = {}
    for bname, cols in batches.items():
        ref_col = next((c for c in cols if any(kw in c.lower() for kw in ref_kw)), None)
        result[bname] = {"samples": cols, "reference": ref_col}

    logger.info("TMT batch structure detected: %s", list(result.keys()))
    return result


def _separate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into sample-value columns and metadata columns.

    Order of checks:
      1. Numeric content first — a column with ≥70% numeric values is a
         sample column regardless of its name. This is critical for the
         canonical single-sheet template where users name columns
         ``Sample_1``, ``Sample_2``, etc. ("sample" matches the metadata
         hint list but the column is clearly a sample-value column).
      2. Otherwise, metadata-hint name match → metadata.
      3. Otherwise, low-numeric → metadata.
    """
    sample_cols, metadata_cols = [], []
    for col in df.columns:
        col_lower = str(col).lower()
        frac = pd.to_numeric(df[col], errors="coerce").notna().sum() / max(len(df), 1)
        if frac >= 0.7:
            sample_cols.append(col)
        elif any(hint in col_lower for hint in _METADATA_HINTS):
            metadata_cols.append(col)
        else:
            metadata_cols.append(col)
    return sample_cols, metadata_cols


# ── Expression sheet parser ────────────────────────────────────────────────────

def _parse_expression_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw expression sheet (already read without index_col),
    find the protein-ID column, set it as index, and return the matrix.
    """
    id_col: Optional[str] = None

    # Strategy 1: column name matches known protein-ID names
    for col in df_raw.columns:
        if _is_protein_id_column(str(col)):
            id_col = col
            break

    # Strategy 2: values in first few columns resemble UniProt IDs
    if id_col is None:
        for col in df_raw.columns[:5]:
            vals = df_raw[col].dropna().astype(str).head(10)
            hits = sum(1 for v in vals if any(p.search(v) for p in _PROTEIN_ID_PATTERNS))
            if hits >= 2:
                id_col = col
                break

    if id_col is not None:
        logger.info("Expression sheet: using '%s' as protein index.", id_col)
        df = df_raw.set_index(id_col)
    else:
        logger.warning(
            "No protein-ID column found; using first column '%s' as index.",
            df_raw.columns[0],
        )
        df = df_raw.set_index(df_raw.columns[0])

    df.index = df.index.astype(str).str.strip()
    return df


# ── Metadata sheet parser ──────────────────────────────────────────────────────

def _clean_metadata_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Generic cleaning for any metadata / annotation sheet:
      • Drop all-NaN columns and rows.
      • Standardise common column names.
      • Coerce numeric columns.
    """
    df = df_raw.copy()

    # Drop completely empty axes
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.reset_index(drop=True)

    # Standardise column names (strip whitespace, title-case)
    col_rename = {}
    std_map = {
        "mouse id": "Mouse ID",
        "strain": "Strain",
        "treatment group": "Treatment Group",
        "[mg/ml]": "Concentration_mg_mL",
        "mg/ml": "Concentration_mg_mL",
        "molecular weight": "Molecular Weight",
        "gene names": "Gene Names",
        "gene name": "Gene Names",
        "protein names": "Protein Names",
        "protein name": "Protein Names",
    }
    for col in df.columns:
        key = str(col).strip().lower()
        if key in std_map:
            col_rename[col] = std_map[key]
        elif key.startswith("client identifier"):
            col_rename[col] = "Pooled Group"
        elif key.startswith("unnamed"):
            # Keep unnamed cols only if they have data; mark empty ones for drop
            if df[col].notna().sum() == 0:
                col_rename[col] = f"__drop_{col}"

    df = df.rename(columns=col_rename)
    df = df[[c for c in df.columns if not str(c).startswith("__drop_")]]

    # Detect and coerce numeric columns
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.7:
            df[col] = converted

    logger.info(
        "Metadata sheet '%s' cleaned: %d rows × %d cols — %s",
        sheet_name, len(df), len(df.columns), list(df.columns),
    )
    return df


# ── Label-map extraction ───────────────────────────────────────────────────────

def _extract_label_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Scan any sheet for single uppercase letter codes paired with group names.
    Returns {letter: group_name} or empty dict.
    """
    letter_pat = re.compile(r"^[A-Z]$")

    def _s(val) -> str:
        if val is None:
            return ""
        s = str(val).strip()
        return "" if s.lower() == "nan" else s

    # Scan every column; for each cell that is a single letter, look for a
    # neighbouring column with a non-numeric group name in the same row.
    for letter_col_idx in range(len(df.columns)):
        candidate: Dict[str, str] = {}
        for row_idx in range(min(30, len(df))):
            cell = _s(df.iloc[row_idx, letter_col_idx])
            if not letter_pat.match(cell):
                continue
            for name_col_idx in range(len(df.columns)):
                if name_col_idx == letter_col_idx:
                    continue
                name_val = _s(df.iloc[row_idx, name_col_idx])
                if name_val and not re.match(r"^\d+\.?\d*$", name_val):
                    candidate[cell] = name_val
                    break
        if len(candidate) >= 2:
            logger.info("Label map found: %s", candidate)
            return candidate

    return {}


# ── Main skill class ───────────────────────────────────────────────────────────

class DataLoadingSkill:
    """
    Loads any CSV or multi-sheet Excel proteomics file.

    For every Excel workbook:
      1. Opens the file and reads EVERY sheet.
      2. Classifies each sheet as 'expression' or 'metadata'.
      3. Parses the first expression sheet as the protein-abundance matrix.
      4. Cleans and stores ALL metadata sheets.
      5. Tries to extract a label_map from any sheet.
      6. Sets is_pooled_design=True when a label_map is found.
    """

    def execute(
        self,
        data_path: str,
        data_format: str = "csv",
        output_dir: str = "data/processed",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data_format = data_format.lower()

        path     = Path(data_path)
        suffix   = path.suffix.lower()
        is_excel = suffix in (".xlsx", ".xls") or data_format in ("xlsx", "xls", "excel")

        label_map:       Optional[Dict[str, str]] = None
        identifier_info: Optional[pd.DataFrame]   = None
        all_sheets:      Dict[str, pd.DataFrame]  = {}
        is_pooled = False
        column_group_labels: Dict[str, str] = {}

        # ── 1. Load ──────────────────────────────────────────────────────────
        if is_excel:
            (df, label_map, is_pooled, identifier_info,
             all_sheets, column_group_labels) = self._load_all_sheets(path, suffix)
            data_format = "excel"
        else:
            df = self._load_csv(str(path))
            data_format = "csv"

        logger.info("Primary matrix: %d rows × %d cols", len(df), len(df.columns))

        # ── 2. Clean primary matrix ───────────────────────────────────────────
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        df.index = df.index.astype(str).str.strip()

        # ── 2b. Spectronaut — reshape before column classification ────────────
        is_spectronaut = _is_spectronaut(df)
        if is_spectronaut:
            logger.info("Spectronaut DIA export detected — reshaping to proteins × samples.")
            df = _parse_spectronaut(df)

        if not is_pooled:
            df = self._ensure_proteins_are_rows(df)

        # ── 3. Separate sample vs metadata columns ────────────────────────────
        sample_cols, metadata_cols = _separate_columns(df)
        logger.info(
            "%d sample cols, %d metadata cols. Preview: %s",
            len(sample_cols), len(metadata_cols), sample_cols[:5],
        )

        # ── 4. Detect data type ───────────────────────────────────────────────
        data_type = _detect_data_type(df, sample_cols)
        logger.info("Data type: %s", data_type)

        # ── 4b. TMT multi-batch detection ─────────────────────────────────────
        tmt_batches = _detect_tmt_batches(sample_cols)

        # ── 4c. SILAC omic type ───────────────────────────────────────────────
        omic_type_hint: Optional[str] = None
        if data_type == "silac_ratio":
            omic_type_hint = "proteomics_silac"
        elif data_type == "dia_spectronaut":
            omic_type_hint = "proteomics"   # DIA uses the standard proteomics pipeline

        # ── 5. Persist processed CSV ──────────────────────────────────────────
        out_name = f"{path.stem}_processed_{uuid.uuid4().hex[:8]}.csv"
        out_path = str(Path(output_dir) / out_name)
        df.to_csv(out_path)
        logger.info("Saved → %s", out_path)

        result: dict = {
            "processed_path":       out_path,
            "data_type":            data_type,
            "data_format":          data_format,
            "n_proteins":           len(df),
            "n_samples":            len(sample_cols),
            "sample_columns":       sample_cols,
            "metadata_columns":     metadata_cols,
            "is_pooled_design":     is_pooled,
            "all_sheets":           all_sheets,
            "column_group_labels":  column_group_labels,
        }
        if label_map:
            result["label_map"] = label_map
        if identifier_info is not None:
            result["identifier_info"] = identifier_info
        if tmt_batches:
            result["tmt_batches"] = tmt_batches
            logger.info("TMT batches stored: %s", list(tmt_batches.keys()))
        if omic_type_hint:
            result["omic_type"] = omic_type_hint
        return result

    # ── Core multi-sheet loader ───────────────────────────────────────────────

    def _load_all_sheets(
        self, path: Path, suffix: str
    ) -> Tuple[
        pd.DataFrame,
        Optional[Dict[str, str]],
        bool,
        Optional[pd.DataFrame],
        Dict[str, pd.DataFrame],
    ]:
        """
        Open the workbook, read ALL sheets, classify each one, and return:
          (primary_df, label_map, is_pooled, identifier_info, all_sheets_dict)
        """
        engines = ["openpyxl", "xlrd"] if suffix != ".xls" else ["xlrd", "openpyxl"]
        xl: Optional[pd.ExcelFile] = None
        last_err: Exception = RuntimeError("no engine tried")

        for engine in engines:
            try:
                xl = pd.ExcelFile(str(path), engine=engine)
                logger.info(
                    "Opened '%s' with engine='%s'. Sheets: %s",
                    path.name, engine, xl.sheet_names,
                )
                break
            except Exception as exc:
                last_err = exc

        if xl is None:
            raise ValueError(
                f"Cannot open '{path.name}'. Engines tried: {engines}. "
                f"Last error: {last_err}."
            )

        expression_sheets: List[Tuple[str, pd.DataFrame]] = []
        metadata_sheets:   List[Tuple[str, pd.DataFrame]] = []

        # ── Read and classify every sheet ─────────────────────────────────────
        # Header detection: many proteomics exports put a title row (e.g.
        # "Identified Proteins (1919)") above the real column names. We read
        # the sheet with header=None first, find the row that matches the most
        # proteomics-header tokens, then re-parse with the detected header row.
        # column_group_labels: real_column_name → group_label (e.g. "SpC J" → "DMD Soleus")
        column_group_labels: Dict[str, str] = {}

        for sheet_name in xl.sheet_names:
            try:
                no_header = xl.parse(sheet_name, header=None)
                header_idx = _detect_header_row(no_header)
                if header_idx > 0:
                    logger.info(
                        "Sheet '%s': auto-detected header row at index %d "
                        "(skipped %d title row(s))",
                        sheet_name, header_idx, header_idx,
                    )
                # ── Extract group labels from the row above the header ──────
                # If row 0 of the raw sheet is ["BL6 Quad", "DMD Quad", ...]
                # above row 1 ["SpC A", "SpC B", ...], pair them so downstream
                # query agents can resolve "DMD Soleus" to the "SpC J" column.
                labels = _extract_column_group_labels(no_header, header_idx)
                # Re-parse with the correct header. xl.parse rewinds the file.
                raw = xl.parse(sheet_name, header=header_idx)
                # ── Canonicalise column names (strip whitespace; rename
                # "Identified Proteins (N)" → "Protein Name") so downstream
                # queries hit stable identifiers across files.
                raw = _canonicalise_columns(raw)
                # Re-key the extracted labels under canonical column names
                if labels:
                    labels = {
                        ("Protein Name" if _TITLE_COL_PATTERN.match(k.strip())
                         else k.strip()): v
                        for k, v in labels.items()
                    }
                    column_group_labels.update(labels)
                    logger.info(
                        "Sheet '%s': extracted %d column→group label pair(s); "
                        "sample: %s",
                        sheet_name, len(labels),
                        {k: labels[k] for k in list(labels)[:3]},
                    )
                kind = _classify_sheet(raw)
                logger.info(
                    "Sheet '%s': %d rows × %d cols → %s",
                    sheet_name, len(raw), len(raw.columns), kind,
                )
                if kind == "expression":
                    expression_sheets.append((sheet_name, raw))
                else:
                    metadata_sheets.append((sheet_name, raw))
            except Exception as exc:
                logger.warning("Could not read sheet '%s': %s", sheet_name, exc)

        if not expression_sheets and not metadata_sheets:
            raise ValueError(f"No readable sheets found in '{path.name}'.")

        # ── Parse primary expression sheet ────────────────────────────────────
        if expression_sheets:
            primary_name, primary_raw = expression_sheets[0]
            logger.info("Primary expression sheet: '%s'", primary_name)
            primary_df = _parse_expression_sheet(primary_raw)
        else:
            # Fallback: largest metadata sheet (shouldn't happen in practice)
            primary_name, primary_raw = max(metadata_sheets, key=lambda t: len(t[1]))
            logger.warning(
                "No expression sheet found — using largest metadata sheet '%s'.",
                primary_name,
            )
            primary_df = _parse_expression_sheet(primary_raw)

        # ── Clean all metadata sheets & try to extract label map ──────────────
        all_sheets: Dict[str, pd.DataFrame] = {}
        label_map: Optional[Dict[str, str]] = None
        identifier_info: Optional[pd.DataFrame] = None

        # Include the primary sheet too
        all_sheets[primary_name] = primary_df

        for sheet_name, raw in metadata_sheets:
            cleaned = _clean_metadata_sheet(raw, sheet_name)
            all_sheets[sheet_name] = cleaned

            # Try to extract label map from this sheet
            if label_map is None:
                extracted = _extract_label_map(raw)
                if extracted:
                    label_map = extracted
                    identifier_info = cleaned
                    logger.info(
                        "Label map extracted from sheet '%s': %s",
                        sheet_name, label_map,
                    )

        # Only use identifier_info from the first metadata sheet (no fake label_map)
        if label_map is None and metadata_sheets:
            identifier_info = all_sheets[metadata_sheets[0][0]]

        # is_pooled is retained on state for legacy callers but the canonical
        # template never sets it — downstream analysis is always run by the
        # dual-engine intensity pipeline regardless of this flag.
        is_pooled = label_map is not None
        return (primary_df, label_map, is_pooled, identifier_info,
                all_sheets, column_group_labels)

    # ── CSV loader ────────────────────────────────────────────────────────────

    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load a CSV, auto-detecting the header row to handle title rows.

        First reads the file with ``header=None`` to find the most likely
        header row (same heuristic as the Excel loader), then re-reads with
        ``skiprows=header_idx``.
        """
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                # Peek at the first ~15 rows to detect the header
                peek = pd.read_csv(path, header=None, encoding=enc, nrows=15)
                header_idx = _detect_header_row(peek)
                if header_idx > 0:
                    logger.info(
                        "CSV '%s': auto-detected header row at index %d "
                        "(skipped %d title row(s))",
                        path, header_idx, header_idx,
                    )
                return pd.read_csv(path, index_col=0, encoding=enc,
                                    skiprows=header_idx)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, index_col=0)

    def _ensure_proteins_are_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_prot = _looks_like_protein_index(df.index)
        cols_prot = _looks_like_protein_index(df.columns)
        if cols_prot and not rows_prot:
            logger.info("Transposing: columns look like proteins.")
            return df.T
        if len(df.columns) > len(df) * 3 and not rows_prot:
            logger.info("Transposing: many more columns than rows.")
            return df.T
        return df
