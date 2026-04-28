"""
skills/run_enrichment.py
Knowledge Layer — PathwaySkill

Pure-Python pathway enrichment via gseapy (Enrichr API).
No R / clusterProfiler required.

Gene symbols are extracted via the ProteinLookupSkill (UniProt REST API)
with a regex fallback for GN= tagged strings. This gives more accurate
gene sets than plain regex alone.
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

# Gene-set libraries queried per organism
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


# ── Gene symbol extraction ────────────────────────────────────────────────────

def _extract_gene_symbols(protein_names: List[str]) -> List[str]:
    """
    Extract HGNC / MGI gene symbols from a list of protein name strings.

    Handles both:
      - UniProt long names  "Myosin-6 OS=Mus musculus GN=Myh6 PE=1 SV=2"
      - Short names         "miDys", "Dystrophin"
    """
    symbols: list[str] = []
    _GN_RE = re.compile(r'\bGN=(\w[\w\-]*)', re.IGNORECASE)

    for name in protein_names:
        name_str = str(name).strip()
        m = _GN_RE.search(name_str)
        if m:
            symbols.append(m.group(1))
        else:
            # No GN= tag — use the part before " OS=" as the gene name
            short = name_str.split(" OS=")[0].strip()
            # Keep only if short enough to be a plausible gene/protein name
            if 1 < len(short) <= 30 and not any(c in short for c in "=|/\\"):
                symbols.append(short)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ── Main skill ────────────────────────────────────────────────────────────────

class PathwaySkill:
    """
    Runs KEGG and GO pathway enrichment using gseapy / Enrichr.

    Accepts raw protein name strings — gene symbols are extracted internally.
    Falls back gracefully if the Enrichr API is unreachable.
    """

    def execute(
        self,
        protein_list: List[str],
        dea_result_path: str = "",
        organism: str = "human",
        pval_cutoff: float = 0.05,
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Use UniProt REST API for reliable ID→gene conversion, then regex as fallback
        lookup_result = _lookup_skill.execute(
            protein_list=protein_list,
            organism=organism,
            output_dir=output_dir,
        )
        gene_symbols = lookup_result["gene_symbols"]
        if not gene_symbols:
            gene_symbols = _extract_gene_symbols(protein_list)
        logger.info(
            "Enrichment input: %d proteins → %d gene symbols (UniProt=%d, regex fallback)",
            len(protein_list), len(gene_symbols), lookup_result["n_resolved"],
        )

        if not gene_symbols:
            logger.warning("No gene symbols could be extracted — enrichment skipped.")
            return self._empty_result(output_dir)

        try:
            import gseapy as gp
        except ImportError:
            raise RuntimeError(
                "gseapy is not installed. Run: python3 -m pip install gseapy"
            )

        organism_key = organism.lower()
        if organism_key == "rat":
            logger.warning(
                "Rat-specific gene-set libraries are unavailable in Enrichr. "
                "Using mouse KEGG/GO libraries as a proxy — results may differ from rat biology."
            )
        libraries   = _LIBRARIES.get(organism_key, _LIBRARIES["human"])
        all_frames: list[pd.DataFrame] = []
        n_kegg = n_go = 0

        for lib in libraries:
            try:
                enr_organism = "human" if organism == "human" else "mouse"
                enr = gp.enrichr(
                    gene_list=gene_symbols,
                    gene_sets=lib,
                    organism=enr_organism,
                    outdir=None,
                    cutoff=pval_cutoff,
                    verbose=False,
                )
                df = enr.results.copy()
                sig = df[df["Adjusted P-value"] <= pval_cutoff].copy()
                if sig.empty:
                    logger.info("Library %s: no significant terms", lib)
                    continue
                sig["library"] = lib
                all_frames.append(sig)
                if "KEGG" in lib:
                    n_kegg += len(sig)
                if "GO_Biological" in lib:
                    n_go += len(sig)
                logger.info("Library %s: %d significant terms", lib, len(sig))
            except Exception as exc:
                logger.warning("Enrichr query failed for library '%s': %s", lib, exc)

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
                "p_value":    float(row.get("P-value", 1.0)),
                "p_adjust":   float(row.get("Adjusted P-value", 1.0)),
                "gene_count": len([g for g in re.split(r"[;,]", str(row.get("Genes", ""))) if g.strip()]),
                "genes":      str(row.get("Genes", "")),
                "overlap":    str(row.get("Overlap", "")),
            }
            for _, row in combined.head(20).iterrows()
        ]

        return {
            "top_pathways":           top_pathways,
            "n_kegg_significant":     n_kegg,
            "n_go_significant":       n_go,
            "enrichment_result_path": out_path,
            "genes_submitted":        len(gene_symbols),
            "gene_symbols":           gene_symbols[:20],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(output_dir: str = "outputs", gene_symbols: list | None = None) -> dict:
        return {
            "top_pathways":           [],
            "n_kegg_significant":     0,
            "n_go_significant":       0,
            "enrichment_result_path": "",
            "genes_submitted":        len(gene_symbols) if gene_symbols else 0,
            "gene_symbols":           (gene_symbols or [])[:20],
        }
