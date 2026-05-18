"""
skills/run_enrichment.py
Knowledge Layer — PathwaySkill

Pure-Python pathway enrichment via gseapy (Enrichr API).
No R / clusterProfiler required.

Design decisions:
  - Caller must pass only significant proteins (not all top_n) for gene set
  - Measured-protein background corrects for the "measured vs genome" bias
  - Up- and down-regulated proteins are enriched separately when available
  - MaxQuant sp|P12345|GENE_SPECIES format is parsed correctly
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from skills.protein_lookup import ProteinLookupSkill

logger = logging.getLogger(__name__)

_lookup_skill = ProteinLookupSkill()

_LIBRARIES: dict[str, list[str]] = {
    "human": [
        "KEGG_2021_Human",
        "GO_Biological_Process_2023",
        "Reactome_2022",
        "WikiPathway_2023_Human",
    ],
    "mouse": [
        "KEGG_2019_Mouse",
        "GO_Biological_Process_2023",
        "WikiPathways_2019_Mouse",
    ],
    "rat": [
        "KEGG_2019_Mouse",          # best available proxy
        "GO_Biological_Process_2023",
    ],
}

_GN_RE = re.compile(r'\bGN=(\w[\w\-]*)', re.IGNORECASE)
# MaxQuant/FASTA: sp|ACCESSION|GENENAME_SPECIES
_SP_RE = re.compile(r'(?:sp|tr)\|[A-Z0-9\-]+\|([A-Z0-9]+)_[A-Z]+', re.IGNORECASE)


# ── Blood / sample-prep contaminants ─────────────────────────────────────────
# Removed BEFORE submission to Enrichr. These proteins are nearly always
# present at high intensity in tissue lysates from incomplete blood washout
# (albumin, hemoglobin), skin contact (keratins), trypsin used in digestion,
# or immunoglobulins from serum. Leaving them in inflates immune / complement
# / coagulation pathways and produces misleading enrichment hits.
#
# All entries are upper-case gene symbols matched case-insensitively.
_CONTAMINANT_EXACT: set = {
    # Serum carrier proteins
    "ALB", "AFP",
    # Complement (commonly contaminating)
    "C3", "C4A", "C4B", "C5", "C6", "C7", "C8A", "C8B", "C8G", "C9",
    # Coagulation
    "F2", "FGA", "FGB", "FGG", "PLG", "SERPINC1", "SERPINA1",
    "APOA1", "APOA2", "APOB", "APOE", "APOC3",
    # Trypsin / digestion reagents
    "TRYP1", "PRSS1", "PRSS2",
    # Immunoglobulins
    "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHM", "IGHD", "IGHE",
    "IGKC", "IGLC1", "IGLC2", "IGLC3", "JCHAIN",
}
_CONTAMINANT_PREFIXES: tuple = (
    "HBA", "HBB", "HBD", "HBG", "HBE", "HBM", "HBZ",   # hemoglobins
    "KRT",                                              # keratins (skin/hair)
    "IGH", "IGK", "IGL",                                # immunoglobulin loci
)


def _filter_contaminants(symbols: List[str]) -> tuple[List[str], List[str]]:
    """Return (clean_symbols, dropped_symbols). Match is case-insensitive.

    Prefix rule: a symbol starting with one of the contaminant prefixes is
    flagged if either the prefix IS the whole symbol, or the next character
    is a digit, '-', or '_' (covers human "HBB1" / mouse "Hbb-b1" / "KRT_5").
    """
    clean: list[str] = []
    dropped: list[str] = []
    for s in symbols:
        u = str(s).strip().upper()
        if not u:
            continue
        is_contam = u in _CONTAMINANT_EXACT
        if not is_contam:
            for p in _CONTAMINANT_PREFIXES:
                if u.startswith(p) and (
                    len(u) == len(p)
                    or u[len(p)] in "-_"
                    or u[len(p)].isdigit()
                ):
                    is_contam = True
                    break
        (dropped if is_contam else clean).append(s)
    return clean, dropped


# ── Gene symbol extraction ────────────────────────────────────────────────────

def _extract_gene_symbols(protein_names: List[str]) -> List[str]:
    """
    Extract HGNC / MGI gene symbols from a list of protein name strings.

    Handles:
      - UniProt long names  "<description> OS=<species> GN=<symbol> PE=<n> SV=<n>"
      - MaxQuant entries    "sp|<accession>|<symbol>_<organism>"
      - MaxQuant groups     "sp|<acc1>|<sym1>_…;sp|<acc2>|<sym2>_…"  (first entry used)
      - Short names / aliases passed through if no parse rule matches.
    """
    symbols: list[str] = []

    for name in protein_names:
        name_str = str(name).strip()
        # MaxQuant protein groups are semicolon-separated — take the first entry
        first_entry = name_str.split(";")[0].strip()

        # 1. GN= tag (UniProt description lines, highest confidence)
        m = _GN_RE.search(name_str)
        if m:
            symbols.append(m.group(1))
            continue

        # 2. sp|ACCESSION|GENE_SPECIES (MaxQuant / FASTA header format)
        m = _SP_RE.search(first_entry)
        if m:
            symbols.append(m.group(1))
            continue

        # 3. Fallback: part before " OS=" or after last "|"
        short = first_entry.split(" OS=")[0].split("|")[-1].strip()
        if 1 < len(short) <= 30 and not any(c in short for c in "=/\\;"):
            symbols.append(short)

    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _resolve_symbols(
    protein_list: List[str],
    organism: str,
    output_dir: str,
    api: bool = True,
) -> List[str]:
    """
    Convert protein name strings to gene symbols.

    api=True  — try UniProt REST first, regex fallback (used for the main gene set)
    api=False — regex only, no network call (used for background and up/down subsets)
    """
    if not protein_list:
        return []
    if not api:
        return _extract_gene_symbols(protein_list)
    lookup = _lookup_skill.execute(
        protein_list=protein_list,
        organism=organism,
        output_dir=output_dir,
    )
    symbols = lookup["gene_symbols"]
    if not symbols:
        symbols = _extract_gene_symbols(protein_list)
    return symbols


# ── Main skill ────────────────────────────────────────────────────────────────

class PathwaySkill:
    """
    Runs KEGG / GO / Reactome / WikiPathway enrichment using gseapy / Enrichr.

    Key features vs. the naive version:
    - background_proteins sets Enrichr background to measured proteins only
      (not the 20 000-gene genome — avoids housekeeping-pathway inflation)
    - up_proteins / down_proteins get separate enrichment runs so the summary
      distinguishes activated from suppressed biology
    - MaxQuant sp|...|GENE_SPECIES parsed correctly
    """

    def execute(
        self,
        protein_list: List[str],
        background_proteins: Optional[List[str]] = None,
        up_proteins: Optional[List[str]] = None,
        down_proteins: Optional[List[str]] = None,
        dea_result_path: str = "",
        organism: str = "human",
        pval_cutoff: float = 0.05,
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        gene_symbols = _resolve_symbols(protein_list, organism, output_dir)
        logger.info(
            "Enrichment: %d sig proteins → %d gene symbols",
            len(protein_list), len(gene_symbols),
        )

        # ── Strip blood / sample-prep contaminants ───────────────────────────
        # Albumin, hemoglobins, keratins, immunoglobulins etc. inflate
        # immune / complement / coagulation pathways and are not biologically
        # meaningful in most tissue proteomics experiments.
        gene_symbols, dropped = _filter_contaminants(gene_symbols)
        if dropped:
            logger.info("Contaminants excluded from enrichment input: %s", dropped)

        if not gene_symbols:
            logger.warning("No gene symbols extracted — enrichment skipped.")
            return self._empty_result(output_dir)

        # Background: regex-only — no API needed, avoids slow batched calls for 2000+ proteins
        background_symbols: Optional[List[str]] = None
        if background_proteins:
            background_symbols = _extract_gene_symbols(background_proteins)
            logger.info(
                "Background: %d proteins → %d gene symbols (regex)",
                len(background_proteins), len(background_symbols),
            )

        # Up/down are subsets of protein_list — regex is consistent and avoids extra API calls
        up_symbols   = _extract_gene_symbols(up_proteins)   if up_proteins   else []
        down_symbols = _extract_gene_symbols(down_proteins) if down_proteins else []
        up_symbols,   _ = _filter_contaminants(up_symbols)
        down_symbols, _ = _filter_contaminants(down_symbols)

        try:
            import gseapy as gp
        except ImportError:
            raise RuntimeError("gseapy is not installed. Run: python3 -m pip install gseapy")

        organism_key = organism.lower()
        if organism_key == "rat":
            logger.warning("Rat Enrichr libraries unavailable; using mouse as proxy.")
        libraries    = _LIBRARIES.get(organism_key, _LIBRARIES["human"])
        enr_organism = "human" if organism_key == "human" else "mouse"

        # Build the list of (direction_label, symbols) to enrich
        runs: list[tuple[str, List[str]]] = []
        if up_symbols:
            runs.append(("up", up_symbols))
        if down_symbols:
            runs.append(("down", down_symbols))
        if not runs:
            runs.append(("all", gene_symbols))

        all_frames: list[pd.DataFrame] = []
        n_kegg = n_go = 0

        for direction, symbols in runs:
            for lib in libraries:
                try:
                    enr = gp.enrichr(
                        gene_list=symbols,
                        gene_sets=lib,
                        organism=enr_organism,
                        outdir=None,
                        background=background_symbols if background_symbols else 20000,
                        cutoff=pval_cutoff,
                        verbose=False,
                    )

                    df  = enr.results.copy()
                    sig = df[df["Adjusted P-value"] <= pval_cutoff].copy()
                    if sig.empty:
                        logger.info("Library %s [%s]: no significant terms", lib, direction)
                        continue
                    sig["library"]   = lib
                    sig["direction"] = direction
                    all_frames.append(sig)
                    if "KEGG" in lib:
                        n_kegg += len(sig)
                    if "GO_Biological" in lib:
                        n_go += len(sig)
                    logger.info("Library %s [%s]: %d significant terms", lib, direction, len(sig))
                except Exception as exc:
                    logger.warning("Enrichr failed — lib=%s direction=%s: %s", lib, direction, exc)

        if not all_frames:
            logger.warning("No enrichment results from any library.")
            return self._empty_result(output_dir, gene_symbols)

        combined = (
            pd.concat(all_frames, ignore_index=True)
            .sort_values("Adjusted P-value")
        )

        out_path = str(Path(output_dir) / "enrichment_results.csv")
        combined.to_csv(out_path, index=False)
        logger.info("Enrichment results saved → %s  (%d rows)", out_path, len(combined))

        top_pathways = [
            {
                "pathway":    row.get("Term", ""),
                "library":    row.get("library", ""),
                "direction":  row.get("direction", "all"),
                "p_value":    float(row.get("P-value", 1.0)),
                "p_adjust":   float(row.get("Adjusted P-value", 1.0)),
                "gene_count": len([g for g in re.split(r"[;,]", str(row.get("Genes", ""))) if g.strip()]),
                "genes":      str(row.get("Genes", "")),
                "overlap":    str(row.get("Overlap", "")),
            }
            for _, row in combined.head(20).iterrows()
        ]

        up_pathways   = [p for p in top_pathways if p["direction"] == "up"][:5]
        down_pathways = [p for p in top_pathways if p["direction"] == "down"][:5]

        return {
            "top_pathways":           top_pathways,
            "up_pathways":            up_pathways,
            "down_pathways":          down_pathways,
            "n_kegg_significant":     n_kegg,
            "n_go_significant":       n_go,
            "enrichment_result_path": out_path,
            "genes_submitted":        len(gene_symbols),
            "gene_symbols":           gene_symbols[:20],
            "background_size":        len(background_symbols) if background_symbols else None,
            "has_directional":        bool(up_symbols or down_symbols),
            "contaminants_excluded":  dropped,
        }

    @staticmethod
    def _empty_result(output_dir: str = "outputs", gene_symbols: list | None = None) -> dict:
        return {
            "top_pathways":           [],
            "up_pathways":            [],
            "down_pathways":          [],
            "n_kegg_significant":     0,
            "n_go_significant":       0,
            "enrichment_result_path": "",
            "genes_submitted":        len(gene_symbols) if gene_symbols else 0,
            "gene_symbols":           (gene_symbols or [])[:20],
            "background_size":        None,
            "has_directional":        False,
        }
