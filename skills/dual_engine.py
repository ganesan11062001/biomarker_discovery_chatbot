"""
skills/dual_engine.py
Combine Python (Welch / limma) and R (limma/Bioconductor) differential
expression results into a single intersected top-biomarker list.

Why intersect?
--------------
The two engines have different prior assumptions:
  • Python's Welch t-test makes no variance assumption (good for n≥5).
  • Python's eBayes uses our own moment-matched prior.
  • R's limma uses Bioconductor's mature ``eBayes(trend=TRUE)`` —
    the field-standard for small-n proteomics.

A protein significant in BOTH engines is far more likely to be a real
hit than one significant only in one. We intersect on the accession,
require adj_p < cutoff in both, and require the log2FC signs to match.

This is intentionally conservative — researchers can drill into the
Python-only or R-only "soft" hits via the per-engine result sheets in
the exported workbook.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DualEngineResult:
    intersected_df: pd.DataFrame                # accession-keyed, both engines significant
    python_df:      pd.DataFrame                # full Python results
    r_df:           pd.DataFrame                # full R results
    n_python_sig:   int          = 0            # significant in Python only
    n_r_sig:        int          = 0            # significant in R only
    n_intersection: int          = 0            # significant in BOTH
    qc:             Dict         = field(default_factory=dict)


def combine_engines(
    python_results: pd.DataFrame,
    r_results:      pd.DataFrame,
    adj_pval_cutoff: float = 0.05,
    log2fc_cutoff:   float = 1.0,
    accession_col_python: Optional[str] = None,
    accession_col_r:      str = "accession",
) -> DualEngineResult:
    """
    Intersect the two engines' significant hits.

    Parameters
    ----------
    python_results
        Output of ``ProteomicsAnalysisSkill._supervised`` /
        ``_limma_ebayes``: must include ``adj_p_value``,
        ``log2_fold_change`` and a protein identifier column.
    r_results
        Output of ``RAnalysisSkill.run()`` (``RLimmaResult.results_df``):
        must include ``accession``, ``adj_p_value``, ``log2_fold_change``.
    accession_col_python
        Name of the accession column in ``python_results``. If None, the
        function auto-picks ``"accession"``, ``"protein"``, or uses the
        DataFrame's index.
    """
    py = python_results.copy()
    r  = r_results.copy()

    # Resolve the Python accession column
    if accession_col_python and accession_col_python in py.columns:
        py = py.rename(columns={accession_col_python: "accession"})
    elif "accession" in py.columns:
        pass
    elif "protein" in py.columns:
        # The Python skill uses 'protein' to hold the accession (it's the
        # row index of proteins_df, which canonical_loader sets to the
        # Accession Number).
        py = py.rename(columns={"protein": "accession"})
    else:
        py = py.reset_index().rename(columns={py.index.name or "index": "accession"})

    py["accession"] = py["accession"].astype(str)
    r ["accession"] = r ["accession"].astype(str)

    py_sig_mask = (py["adj_p_value"] < adj_pval_cutoff) & \
                  (py["log2_fold_change"].abs() >= log2fc_cutoff)
    r_sig_mask  = (r ["adj_p_value"] < adj_pval_cutoff) & \
                  (r ["log2_fold_change"].abs() >= log2fc_cutoff)

    n_py = int(py_sig_mask.sum())
    n_r  = int(r_sig_mask.sum())

    py_hits = py.loc[py_sig_mask, ["accession", "log2_fold_change",
                                    "p_value", "adj_p_value"]].rename(
        columns={"log2_fold_change": "log2fc_python",
                 "p_value":          "p_python",
                 "adj_p_value":      "adj_p_python"})
    r_hits  = r.loc[r_sig_mask, ["accession", "log2_fold_change",
                                  "p_value", "adj_p_value"]].rename(
        columns={"log2_fold_change": "log2fc_r",
                 "p_value":          "p_r",
                 "adj_p_value":      "adj_p_r"})

    merged = py_hits.merge(r_hits, on="accession", how="inner")

    # Require sign agreement — if Python says +log2FC and R says –log2FC the
    # protein isn't a reliable hit even if both are "significant".
    sign_match = np.sign(merged["log2fc_python"]) == np.sign(merged["log2fc_r"])
    merged = merged.loc[sign_match].copy()

    # Mean log2FC + combined ranking: smaller adj_p means higher rank.
    merged["mean_log2fc"]    = (merged["log2fc_python"] + merged["log2fc_r"]) / 2.0
    merged["combined_adj_p"] = merged[["adj_p_python", "adj_p_r"]].max(axis=1)
    merged = merged.sort_values("combined_adj_p").reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    # Bring back the protein name + gene symbol if Python had them
    extras = [c for c in ("protein_name", "gene_name", "significance")
              if c in py.columns]
    if extras:
        merged = merged.merge(py[["accession"] + extras].drop_duplicates("accession"),
                              on="accession", how="left")

    # Canonical column aliases — every downstream consumer (enrichment agent,
    # visualisation, domain expert, chatbot summary) reads these names:
    #   protein           = accession
    #   log2_fold_change  = mean of Python & R log2FCs
    #   adj_p_value       = max of Python & R adj.p   (the conservative one)
    #   p_value           = max of Python & R raw p
    merged["protein"]          = merged["accession"]
    merged["log2_fold_change"] = merged["mean_log2fc"]
    merged["adj_p_value"]      = merged["combined_adj_p"]
    merged["p_value"]          = merged[["p_python", "p_r"]].max(axis=1)
    if "significance" not in merged.columns:
        merged["significance"] = "Significant"

    n_int = len(merged)
    qc = {
        "adj_pval_cutoff": adj_pval_cutoff,
        "log2fc_cutoff":   log2fc_cutoff,
        "python_sig":      n_py,
        "r_sig":           n_r,
        "intersection":    n_int,
        "agreement_rate":  round(n_int / max(min(n_py, n_r), 1), 3),
    }
    logger.info(
        "Dual-engine intersection: Python=%d, R=%d, ∩=%d (%.1f%% agreement on smaller set)",
        n_py, n_r, n_int, 100.0 * qc["agreement_rate"],
    )

    return DualEngineResult(
        intersected_df = merged,
        python_df      = py,
        r_df           = r,
        n_python_sig   = n_py,
        n_r_sig        = n_r,
        n_intersection = n_int,
        qc             = qc,
    )
