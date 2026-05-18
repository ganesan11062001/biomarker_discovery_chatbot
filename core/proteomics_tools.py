"""
core/proteomics_tools.py
Deterministic helper functions exposed to LLM-generated pandas code so common
proteomics queries are guaranteed correct (no LLM hallucination of metric
type, no /0 errors, no missing protein-name columns).

Every function here is data-agnostic — they detect schema patterns rather
than assuming specific column names or values. A caller (DataLoadingSkill,
IngestionAgent, or the safe-exec namespace) hands in the actual DataFrames
and the functions infer everything else.

These functions are also injected into the safe_exec sandbox so the LLM can
just call them by name instead of re-implementing fragile logic.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ── Column-family detection ───────────────────────────────────────────────────

_SPC_TOKENS       = ("spc", "spectral", "spectralcount", "spectral_count")
_INTENSITY_TOKENS = ("intensity", "lfq", "ibaq")
_RATIO_TOKENS     = ("ratio", "log2_ratio", "h/l", "m/l", "h/m")
_IDENTIFIER_TOKENS = ("protein name", "protein", "description", "gene", "accession", "uniprot")


def _matches_any(name: str, tokens: tuple) -> bool:
    n = re.sub(r"[\s_\-]+", "", name.lower())
    return any(re.sub(r"[\s_\-]+", "", t.lower()) in n for t in tokens)


def detect_metric_columns(df) -> Dict[str, List[str]]:
    """
    Group a proteins DataFrame's columns by metric type.

    Returns a dict like:
        {
            "identifier": ["Protein Name", "Accession Number", "Molecular Weight"],
            "spc":        ["A SpC", "B SpC", ...],
            "intensity":  ["A Intensity", "B Intensity", ...],
            "ratio":      [...],
            "other":      [...],
        }

    Two paths:

    1. **Explicit token match** (legacy / MaxQuant-style files):
       Columns whose name contains "SpC" / "Intensity" / "LFQ" / "iBAQ" /
       "Ratio" / "NPX" are classified directly. Identifier-looking columns
       go to "identifier".

    2. **Canonical template** (our standardised single-sheet format):
       If NO SpC / Intensity / Ratio columns are detected but the sheet
       has 2+ identifier columns followed by numeric columns, those
       numeric columns are treated as **intensity by default**. This
       lets users hand us a clean template like
           Protein Name | Accession | Gene | Sample_1 | Sample_2 | ...
       without having to put "Intensity" in every sample header.

    The grouping is purely schema-driven — no hardcoded sample codes or
    biology assumptions.
    """
    import pandas as pd

    out: Dict[str, List[str]] = {
        "identifier": [],
        "spc":        [],
        "intensity":  [],
        "ratio":      [],
        "other":      [],
    }
    unlabeled: List[str] = []          # columns with no metric token in the name

    for col in df.columns:
        c  = str(col)
        cl = c.lower()
        # Skip flag columns added by maxquant_filters — they go to "other"
        # but specifically as booleans, never as a metric.
        if c.startswith("is_") and (cl in {"is_reverse", "is_potential_contaminant",
                                              "is_only_identified_by_site",
                                              "is_contaminant_accession",
                                              "is_contaminant"}):
            out["other"].append(c)
            continue
        if _matches_any(c, _SPC_TOKENS):
            out["spc"].append(c)
        elif _matches_any(c, _INTENSITY_TOKENS):
            out["intensity"].append(c)
        elif _matches_any(c, _RATIO_TOKENS):
            out["ratio"].append(c)
        elif any(t in cl for t in _IDENTIFIER_TOKENS):
            out["identifier"].append(c)
        else:
            unlabeled.append(c)

    # ── Canonical-template fallback ────────────────────────────────────────
    # If we found no explicit metric tokens AND we have enough identifier
    # context, treat all the remaining unlabeled NUMERIC columns as
    # intensity samples.
    no_explicit_metrics = (not out["spc"] and not out["intensity"]
                           and not out["ratio"])
    if no_explicit_metrics and len(out["identifier"]) >= 1:
        for c in unlabeled:
            try:
                series = df[c]
            except Exception:
                out["other"].append(c)
                continue
            # Treat as intensity if the column is mostly numeric
            try:
                numeric = pd.to_numeric(series, errors="coerce")
                non_null_frac = numeric.notna().mean() if len(numeric) else 0
            except Exception:
                non_null_frac = 0
            if non_null_frac >= 0.5:
                out["intensity"].append(c)
            else:
                out["other"].append(c)
    else:
        # We found explicit metric columns — any unlabeled cols stay in "other"
        out["other"].extend(unlabeled)

    return out


# Replicate suffixes to strip when computing the base group name.
# Order matters: try longest / most specific first.
_REPLICATE_SUFFIX_RE = re.compile(
    r"(?:"
    r"\.\d+"             # pandas auto-rename for duplicate headers: WT, WT.1, WT.2
    r"|[_\-]\d+"         # underscore / dash + number:  WT_1, WT-1
    r"|\s+\d+"           # whitespace + number:         'WT 1'
    r"|\d+"              # trailing digits with no separator: WT1, WT12
    r")$",
)


def infer_groups_from_row0(
    sample_columns: List[str],
    *,
    min_groups: int = 2,
) -> Dict[str, List[str]]:
    """
    Deterministic group inference for the canonical single-sheet template
    where the column header IS the group name and pandas auto-renames
    duplicate headers as ``WT``, ``WT.1``, ``WT.2``.

    Algorithm:
      1. For each column, strip the replicate suffix (``.N``, ``_N``,
         ``-N``, `` N``, or trailing digits) to get the *base group name*.
      2. Group columns by base name.
      3. Return the mapping only when at least ``min_groups`` distinct
         base names exist; otherwise return ``{}`` so the caller falls
         back to the LLM-based inference.

    Examples (using the user's canonical-template convention):

        >>> infer_groups_from_row0(["WT", "WT.1", "DMD", "DMD.1"])
        {"WT": ["WT", "WT.1"], "DMD": ["DMD", "DMD.1"]}

        >>> infer_groups_from_row0(["WT_1", "WT_2", "KO_1", "KO_2"])
        {"WT": ["WT_1", "WT_2"], "KO": ["KO_1", "KO_2"]}

        >>> infer_groups_from_row0(["Sample_1", "Sample_2", "Sample_3"])
        {}   # only one base name → caller falls back to LLM
    """
    if not sample_columns:
        return {}

    groups: Dict[str, List[str]] = {}
    for col in sample_columns:
        base = _REPLICATE_SUFFIX_RE.sub("", str(col)).strip()
        if not base:
            continue
        groups.setdefault(base, []).append(col)

    if len(groups) < min_groups:
        return {}
    # Sanity: every group must have at least 1 column (trivially true) and
    # the grouping must not be 1:1 with columns (i.e. every column its own
    # group means no aggregation actually happened).
    if all(len(v) == 1 for v in groups.values()) and len(groups) == len(sample_columns):
        return {}
    return groups


def infer_groups_from_row0(
    sample_columns: List[str],
    column_group_labels: Optional[Dict[str, str]],
    *,
    min_groups: int = 2,
) -> Dict[str, List[str]]:
    """Resolve sample groups from the row-0 labels extracted at ingest.

    In the canonical sheet layout, row 0 of the workbook contains the
    biological group name for each sample column (cols 4+). ``DataLoadingSkill``
    pairs each real column with the row-0 cell above it and stores the result
    as ``column_group_labels`` (``real_column → group_name``). This function
    inverts that mapping, keeping only entries whose key is in
    ``sample_columns``, and returns ``{group_name: [cols, ...]}``.

    Returns ``{}`` when fewer than ``min_groups`` distinct groups can be
    resolved — so the caller (IngestionAgent) can fall back to the LLM-based
    column-name inference for files without row-0 labels.
    """
    if not column_group_labels or not sample_columns:
        return {}
    sample_set = set(sample_columns)
    groups: Dict[str, List[str]] = {}
    for col, label in column_group_labels.items():
        if col not in sample_set:
            continue
        if label is None:
            continue
        name = str(label).strip()
        if not name:
            continue
        groups.setdefault(name, []).append(col)
    if len(groups) < min_groups:
        return {}
    return groups


def split_spc_intensity(df) -> Tuple[Any, Any]:
    """
    BUG-1 FIX. Return two DataFrames — one with identifier columns + SpC
    columns, the other with identifier columns + intensity columns. Each
    caller must explicitly pick which to query; nothing mixes them.

    Both DataFrames also carry any contaminant-flag columns (``is_*``) that
    were added by the MaxQuant cleanup pipeline — so LLM-generated pandas
    code can write ``df_spc[~df_spc['is_contaminant']]`` without hitting
    KeyError. Without this, the flag columns get classified as "other"
    and dropped from the split frames.

    If a metric family isn't present, the corresponding DataFrame is empty.
    """
    import pandas as pd

    groups   = detect_metric_columns(df)
    id_cols  = groups["identifier"]
    spc_cols = groups["spc"]
    int_cols = groups["intensity"]
    # Contaminant / decoy flag columns added by maxquant_filters.flag_*
    flag_cols = [c for c in df.columns
                 if c.startswith("is_") and df[c].dtype == bool]

    df_spc = df[id_cols + spc_cols + flag_cols].copy() if spc_cols else pd.DataFrame()
    df_intensity = df[id_cols + int_cols + flag_cols].copy() if int_cols else pd.DataFrame()

    # Coerce metric columns to numeric so LLM-generated sorts / fold changes
    # don't fail with "TypeError: '<' not supported between instances of
    # 'float' and 'str'" when stray strings (NaN, '?', '' ) sneak in.
    for c in spc_cols:
        if c in df_spc.columns:
            df_spc[c] = pd.to_numeric(df_spc[c], errors="coerce")
    for c in int_cols:
        if c in df_intensity.columns:
            df_intensity[c] = pd.to_numeric(df_intensity[c], errors="coerce")

    return df_spc, df_intensity


# ── Safe arithmetic ───────────────────────────────────────────────────────────

def safe_fold_change(
    numerator:     float,
    denominator:   float,
    sample_num:    str = "numerator",
    sample_den:    str = "denominator",
) -> Union[float, str]:
    """
    BUG-2 FIX. Compute a fold change with explicit absent-value semantics.

    Returns a float when both samples have non-zero signal; otherwise
    returns a clear string explaining the situation. Treats 0 as "absent",
    never as a real intensity — so a 0/0 ratio is "undefined", not NaN.
    """
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return "undefined — non-numeric values"

    if d == 0 and n == 0:
        return f"undefined — protein absent in both samples ({sample_num} and {sample_den})"
    if d == 0:
        return (f"undefined — protein absent in sample {sample_den} (value=0), "
                f"cannot divide by zero")
    if n == 0:
        return f"0 — protein absent in sample {sample_num} (value=0)"
    return round(n / d, 4)


# ── Dataset-level dynamic detection ───────────────────────────────────────────
# All three functions read from the actual file at runtime — no hardcoded
# answers. The disease-model map below is generic bioinformatics knowledge
# (well-published mouse / human models), not user-specific data.

# Known disease-model tokens. Matching is case-insensitive and considers whole
# words OR explicit gene-symbol mentions in strain / treatment / sample names.
# Extend this map by adding new entries — it's intentionally generic biology.
_DISEASE_MODEL_TOKENS: Dict[str, List[str]] = {
    # canonical disease label → list of tokens that, if seen in metadata, suggest it
    "Duchenne Muscular Dystrophy (DMD)":  ["mdx", "dmd", "dystrophin", "dys"],
    "Spinal Muscular Atrophy (SMA)":      ["sma", "smn1", "smn"],
    "Amyotrophic Lateral Sclerosis (ALS)":["sod1", "tdp-43", "tdp43", "fus", "c9orf72"],
    "Alzheimer's Disease (AD)":           ["app", "ps1", "ps2", "presenilin", "tau", "mapt"],
    "Huntington's Disease (HD)":          ["htt", "huntingtin"],
    "Parkinson's Disease (PD)":           ["snca", "alpha-synuclein", "parkin", "park2", "lrrk2"],
    "Friedreich's Ataxia (FA)":           ["fxn", "frataxin"],
}


# Software signatures — column/sheet names that uniquely identify a vendor.
_SOFTWARE_SIGNATURES: Dict[str, List[str]] = {
    "MaxQuant":            ["maxquant", "max quant", "lfq intensity", "ibaq",
                            "identified proteins"],
    "FragPipe":            ["fragpipe", "philosopher", "msstats annotation"],
    "Proteome Discoverer": ["proteome discoverer", "proteomediscoverer", "pd report"],
    "Spectronaut":         ["spectronaut", "pg.quantity", "eg.quantity"],
    "DIA-NN":              ["diann", "dia-nn", "precursor.normalised"],
    "Skyline":             ["skyline", "skylinequantitative"],
}


_GN_RE = re.compile(r"GN=([A-Za-z0-9._-]+)")
_OS_RE = re.compile(r"OS=([A-Z][a-z]+(?:\s+[a-z]+)+?)(?=\s+(?:OX|GN|PE|SV|=)|$)")


def get_gene_symbol(protein_name: Optional[str]) -> str:
    """Extract the gene symbol from a UniProt-style protein name.

    The standard MaxQuant / UniProt description format is:
        "<protein description> OS=<species> OX=<taxid> GN=<gene_symbol> PE=<n> SV=<n>"
    Returns the value after ``GN=``, or 'Unknown' if no ``GN=`` is present.
    """
    if not protein_name:
        return "Unknown"
    m = _GN_RE.search(str(protein_name))
    return m.group(1) if m else "Unknown"


def get_short_name(protein_name: Optional[str]) -> str:
    """Return the bare description (everything before the first ` OS=`)."""
    if not protein_name:
        return ""
    return str(protein_name).split(" OS=", 1)[0].strip()


def format_protein_row(
    protein_name: str,
    accession:    Optional[str],
    value:        Any,
    unit:         str = "",
) -> str:
    """BUG-4 FIX. Render a protein row as 'Gene (Accession) — Value Unit'.

    If gene symbol can't be parsed, fall back to the protein description.
    """
    gene = get_gene_symbol(protein_name)
    if gene == "Unknown":
        gene = get_short_name(protein_name) or "Unknown"
    acc = accession or "?"
    val_str = f"{value} {unit}".strip() if unit else f"{value}"
    return f"{gene} ({acc}) — {val_str}"


# ── Identifier-sheet → sample map ─────────────────────────────────────────────

# Common header tokens used by MaxQuant-style workbooks. We match
# case-insensitively and tolerate spaces/underscores.
_SAMPLE_CODE_HINTS = ("maxquant", "max quant", "samplecode", "sample code",
                      "samplelabel", "sample label", "label", "channel")
_CLIENT_ID_HINTS   = ("clientidentifier", "client identifier", "client id",
                      "clientid", "sample id", "subjectid", "subject id",
                      "name", "alias", "condition")
_STRAIN_HINTS      = ("strain", "genotype", "background")
_TREATMENT_HINTS   = ("treatment", "treatmentgroup", "treatment group",
                      "condition", "intervention")
_MOUSE_ID_HINTS    = ("mouseid", "mouse id", "subject", "animal id", "animal")


def _pick_column(columns: List[str], hints: tuple) -> Optional[str]:
    """Return the first column whose name (case-folded, no whitespace) matches any hint."""
    norm = {c: re.sub(r"[\s_\-]+", "", str(c).lower()) for c in columns}
    for c, n in norm.items():
        for h in hints:
            h_norm = re.sub(r"[\s_\-]+", "", h.lower())
            if h_norm in n:
                return c
    return None


# Hints for the new 2-sheet canonical template (Sample ID + Group sheet).
_SAMPLE_ID_HINTS  = ("sample id", "sampleid", "sample_id", "sample-id", "sample")
_GROUP_HINTS_V2   = ("group", "condition", "treatment", "class", "phenotype", "cohort")


def build_sample_group_map(
    metadata_sheet,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Read a Sample ID + Group metadata sheet and produce both directions of
    the mapping used by every group-vs-group analysis.

    Returns ``(sample_to_group, group_to_samples)``:
      * ``sample_to_group``  — ``{"S1": "WT", "S2": "WT", "S3": "DMD", …}``
      * ``group_to_samples`` — ``{"WT": ["S1", "S2"], "DMD": ["S3"], …}``

    The Sample ID values here will match the *column names* in the proteins
    sheet, so the orchestrator can translate "compare WT vs DMD" directly
    into lists of column names without any LLM inference.

    Returns ``({}, {})`` when the sheet doesn't expose a usable Sample ID
    and Group column — callers should then fall back to other strategies.
    """
    import pandas as pd
    if metadata_sheet is None or not isinstance(metadata_sheet, pd.DataFrame) \
            or metadata_sheet.empty:
        return {}, {}

    columns = list(metadata_sheet.columns)
    sample_col = _pick_column(columns, _SAMPLE_ID_HINTS)
    group_col  = _pick_column(columns, _GROUP_HINTS_V2)
    # Don't let one column win both slots — that happens when the sheet
    # only has e.g. a "Sample" column. Force them to be different.
    if sample_col is not None and sample_col == group_col:
        alt = [c for c in columns if c != sample_col]
        group_col = _pick_column(alt, _GROUP_HINTS_V2)

    if sample_col is None or group_col is None:
        return {}, {}

    sample_to_group: Dict[str, str] = {}
    group_to_samples: Dict[str, List[str]] = {}
    for _, row in metadata_sheet.iterrows():
        sid = _clean(row.get(sample_col))
        grp = _clean(row.get(group_col))
        if not sid or not grp:
            continue
        sid_str = str(sid).strip()
        grp_str = str(grp).strip()
        sample_to_group[sid_str] = grp_str
        group_to_samples.setdefault(grp_str, []).append(sid_str)

    # Drop only the truly useless case: zero groups. A single-group result
    # (e.g. only WT samples) is still informative for downstream Q&A even
    # though no comparison is possible. n=1-per-group pooled designs are
    # also valid — they route to PooledFoldChangeSkill.
    if not group_to_samples:
        return {}, {}

    return sample_to_group, group_to_samples


def build_sample_map(identifier_sheet) -> Dict[str, Dict[str, Any]]:
    """
    BUG-3 FIX. Build a clean sample_code → {client_id, strain, treatment,
    mouse_id} mapping from an identifier sheet.

    Algorithm (no hardcoded codes / labels):
      1. Pick the column that looks like the sample-code column (MaxQuant /
         Channel / Sample Label / etc.).
      2. Keep ONLY rows where that column is a non-empty short code (1–4
         characters) — this filters out subject-level rows like 'Mouse 13'.
      3. For every kept row, harvest the candidate metadata columns by name.

    Returns an empty dict if no recognisable code column or no short-code
    rows exist (e.g. the sheet is purely per-subject).
    """
    import pandas as pd
    if identifier_sheet is None or not isinstance(identifier_sheet, pd.DataFrame) \
            or identifier_sheet.empty:
        return {}

    columns = list(identifier_sheet.columns)
    code_col = _pick_column(columns, _SAMPLE_CODE_HINTS)
    if code_col is None:
        # No explicit MaxQuant/Channel/Label column — try heuristic: find
        # a column whose non-null values are mostly 1–4 char strings.
        code_col = _heuristic_code_column(identifier_sheet)
    if code_col is None:
        return {}

    client_col    = _pick_column(columns, _CLIENT_ID_HINTS)
    strain_col    = _pick_column(columns, _STRAIN_HINTS)
    treatment_col = _pick_column(columns, _TREATMENT_HINTS)
    mouse_col     = _pick_column(columns, _MOUSE_ID_HINTS)

    # Avoid the same column being picked for both MaxQuant code AND client_id —
    # use a different column for client if the picker chose the code column.
    if client_col == code_col:
        alt_columns = [c for c in columns if c != code_col]
        client_col  = _pick_column(alt_columns, _CLIENT_ID_HINTS)

    sample_map: Dict[str, Dict[str, Any]] = {}
    for _, row in identifier_sheet.iterrows():
        raw_code = row.get(code_col)
        if raw_code is None:
            continue
        code = str(raw_code).strip()
        # Keep only short codes (typical MaxQuant labels are 1–4 chars)
        if not code or len(code) > 4 or code.lower() in ("nan", "none", ""):
            continue
        entry: Dict[str, Any] = {"code": code}
        if client_col is not None:
            entry["client_id"] = _clean(row.get(client_col))
        if strain_col is not None:
            entry["strain"] = _clean(row.get(strain_col))
        if treatment_col is not None:
            entry["treatment"] = _clean(row.get(treatment_col))
        if mouse_col is not None:
            entry["mouse_id"] = _clean(row.get(mouse_col))
        sample_map[code] = entry
    return sample_map


def _heuristic_code_column(df) -> Optional[str]:
    """Find a column whose values look like short sample codes."""
    import pandas as pd
    best, best_score = None, 0
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        if len(vals) == 0:
            continue
        short = vals[vals.str.len().between(1, 4)]
        if len(short) >= 2:
            score = len(short) - 0.1 * len(vals - short)
            if score > best_score:
                best, best_score = col, score
    return best


def _clean(value: Any) -> Any:
    """Normalise NaN / empty string → None so downstream consumers see a clear absence."""
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    return s


# ── Non-standard protein lookup ──────────────────────────────────────────────

def get_nonstandard_protein(
    df,
    accession_or_name: str,
    metric: str = "spc",
) -> Dict[str, Any]:
    """
    BUG-6 FIX. Look up a protein by exact accession first, then by partial
    name match. Return the requested metric's per-sample values (or 'NOT
    FOUND' message). Never sums across metric types.
    """
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "no proteins table available"}

    groups = detect_metric_columns(df)
    ident_cols = groups["identifier"]
    if not ident_cols:
        return {"error": "could not identify protein-name columns"}

    # Pick metric columns
    metric_key = metric.lower()
    metric_cols = groups.get(metric_key, [])
    if not metric_cols:
        return {"error": f"no '{metric}' columns in this table"}

    # Try exact accession match across all identifier columns
    target = str(accession_or_name).strip()
    target_lower = target.lower()

    accession_col = next(
        (c for c in ident_cols if "accession" in c.lower() or "uniprot" in c.lower()),
        None,
    )
    name_col = next(
        (c for c in ident_cols if "protein" in c.lower() and "name" in c.lower()),
        None,
    ) or next((c for c in ident_cols if "protein" in c.lower()), None)

    row = None
    if accession_col is not None:
        exact = df[df[accession_col].astype(str).str.strip().str.lower() == target_lower]
        if not exact.empty:
            row = exact.iloc[0]
    if row is None and name_col is not None:
        partial = df[df[name_col].astype(str).str.contains(
            re.escape(target), case=False, na=False)]
        if not partial.empty:
            row = partial.iloc[0]
    if row is None:
        return {"error": f"protein {accession_or_name!r} not found"}

    result: Dict[str, Any] = {}
    if name_col is not None:
        result["protein_name"] = row[name_col]
    if accession_col is not None:
        result["accession"]    = row[accession_col]
    for c in metric_cols:
        try:
            result[c] = float(row[c]) if pd.notna(row[c]) else 0
        except (TypeError, ValueError):
            result[c] = row[c]
    result["metric"] = metric_key
    return result


# ── Convenience: top N by a metric ───────────────────────────────────────────

def top_n_by_metric(
    df,
    metric_col: str,
    n: int = 10,
) -> "pd.DataFrame":  # noqa: F821
    """
    Return the top N rows of `df` sorted descending by `metric_col`, with
    identifier columns always included on the left so callers never lose
    the protein name / accession.
    """
    import pandas as pd
    if df is None or metric_col not in df.columns:
        return pd.DataFrame()
    groups = detect_metric_columns(df)
    keep   = [c for c in groups["identifier"] if c in df.columns] + [metric_col]
    return df[keep].sort_values(metric_col, ascending=False).head(n)


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic dataset-level detection
# ══════════════════════════════════════════════════════════════════════════════
# Each function reads from the actual workbook (`all_sheets` and / or
# `sample_map`) at runtime. None of them assume specific organisms, software,
# or disease programs — they only match against the generic biology /
# proteomics knowledge encoded in the constants above.


def detect_organism(all_sheets: Dict[str, Any]) -> Optional[str]:
    """
    Scan protein-description columns for `OS=<species>` (UniProt / MaxQuant
    FASTA descriptor format) and return the modal species observed.

    Returns None when no descriptor is found — never falls back to a default
    organism. The caller decides what to show when the value is None.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    if not all_sheets:
        return None

    from collections import Counter
    counts: Counter = Counter()

    def _scan_series(values_iter) -> None:
        for v in values_iter:
            m = _OS_RE.search(v)
            if m:
                counts[m.group(1).strip()] += 1

    for df in all_sheets.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # ── 1. Scan columns whose name looks descriptive ──────────────────
        for col in df.columns:
            cl = str(col).lower()
            if not any(t in cl for t in ("protein", "description", "fasta",
                                          "identifi", "name")):
                continue
            try:
                _scan_series(df[col].dropna().astype(str).head(200))
            except Exception:
                continue
            if sum(counts.values()) > 50:
                break
        if sum(counts.values()) > 50:
            break
        # ── 2. Fallback: also scan the DataFrame INDEX. After ingestion
        # the protein-description column is often promoted to the index by
        # _parse_expression_sheet, so it's invisible to the column scan
        # above. Trying the index catches that case. ───────────────────────
        idx_name = (df.index.name or "").lower()
        if idx_name and any(t in idx_name for t in
                             ("protein", "description", "fasta", "identifi", "name")):
            try:
                _scan_series(df.index.dropna().astype(str)[:200])
            except Exception:
                pass
        elif not isinstance(df.index, pd.RangeIndex):
            # Anonymous non-trivial index — sample it anyway as a last resort
            try:
                idx_values = df.index.astype(str)[:50]
                if any("OS=" in s for s in idx_values):
                    _scan_series(df.index.astype(str)[:200])
            except Exception:
                pass

    if not counts:
        return None
    return counts.most_common(1)[0][0]


def detect_software(all_sheets: Dict[str, Any]) -> Optional[str]:
    """
    Identify the quantification software by scanning sheet names and column
    names against known vendor signatures (`_SOFTWARE_SIGNATURES`).

    Returns the canonical software label (e.g. 'MaxQuant') or None when no
    signature is recognised.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    if not all_sheets:
        return None

    haystack_parts: List[str] = []
    for sheet_name, df in all_sheets.items():
        haystack_parts.append(str(sheet_name).lower())
        if isinstance(df, pd.DataFrame):
            haystack_parts.extend(str(c).lower() for c in df.columns)
    haystack = " ".join(haystack_parts)

    matches: Dict[str, int] = {}
    for software, tokens in _SOFTWARE_SIGNATURES.items():
        hits = sum(1 for t in tokens if t in haystack)
        if hits:
            matches[software] = hits

    if not matches:
        return None
    # Return the software with the most signature hits
    return max(matches.items(), key=lambda kv: kv[1])[0]


def detect_disease(
    sample_map:  Optional[Dict[str, Dict[str, Any]]] = None,
    all_sheets:  Optional[Dict[str, Any]]            = None,
    organism:    Optional[str]                       = None,
) -> Optional[str]:
    """
    Infer the disease program from values in the sample_map (strain,
    treatment, client_id) and, as a fallback, by scanning all sheet contents
    for known disease-model tokens.

    Matching is purely token-based against the `_DISEASE_MODEL_TOKENS` map
    — no specific user value is hardcoded. Returns None when no signal is
    found; callers should treat None as "unknown — ask the user".

    The `organism` arg is reserved for future organism-aware disambiguation
    (e.g. an HTT hit in a fly would be a model, not the disease itself); it's
    currently a hint only.
    """
    _ = organism   # reserved for future use
    haystack_tokens: List[str] = []

    # 1. Pull strain / treatment / client_id values out of the sample_map —
    #    these are the most reliable signal because they're per-sample metadata.
    if sample_map:
        for entry in sample_map.values():
            for key in ("strain", "treatment", "client_id"):
                val = entry.get(key)
                if val:
                    haystack_tokens.append(str(val).lower())

    # 2. Fall back: scan the identifier sheet (small, metadata-rich) for any
    #    cell containing a disease-model token. This catches files where
    #    the strain/treatment columns are named differently than expected.
    if not haystack_tokens and all_sheets:
        try:
            import pandas as pd
            # Only the smaller sheets — large protein tables would slow us down
            # and the same tokens would not appear there.
            for df in all_sheets.values():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if df.shape[0] > 200:
                    continue
                for col in df.columns:
                    try:
                        vals = df[col].dropna().astype(str).str.lower().tolist()
                    except Exception:
                        continue
                    haystack_tokens.extend(vals)
        except ImportError:
            pass

    if not haystack_tokens:
        return None

    haystack = " ".join(haystack_tokens)
    matches: Dict[str, int] = {}
    for disease, tokens in _DISEASE_MODEL_TOKENS.items():
        hits = 0
        for t in tokens:
            # Whole-word match to avoid false positives like 'dmd' inside 'dmdomain'
            if re.search(rf"\b{re.escape(t.lower())}\b", haystack):
                hits += 1
        if hits:
            matches[disease] = hits

    if not matches:
        return None
    return max(matches.items(), key=lambda kv: kv[1])[0]
