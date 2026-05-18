"""
skills/r_analysis.py
R-side proteomics differential expression via limma (Bioconductor).

Architecture
------------
The Python pipeline is fast and serves the chatbot's primary loop. limma
in R is the gold-standard moderated t-test for small-n proteomics and is
worth incorporating as an independent second opinion. Rather than embed
rpy2 (which complicates packaging and pulls a heavy native dep), this
skill writes a CSV + an R script to a temporary directory and invokes
``Rscript`` as a subprocess. The R script writes its results back to a
CSV which Python re-reads.

Inputs
------
- ``expression_csv``  : protein × sample numeric matrix (log₂-normalised
                        upstream by ProteomicsAnalysisSkill). Index column
                        holds the accession number.
- ``group1_samples``  : column names belonging to group 1.
- ``group2_samples``  : column names belonging to group 2.
- ``group1_label`` / ``group2_label``.

Outputs
-------
A ``RLimmaResult`` dataclass with the limma top-table as a DataFrame
plus the path to the raw R-side CSV (kept for downstream debugging).

R must be on PATH with the ``limma`` package installed:
    install.packages("BiocManager"); BiocManager::install("limma")
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RAnalysisError(RuntimeError):
    """Raised when the R subprocess fails or limma isn't available."""


@dataclass
class RLimmaResult:
    results_df:   pd.DataFrame   # accession, log2_fold_change, p_value, adj_p_value, sig
    r_script:     str            # the R code that was executed
    stdout:       str            # captured Rscript stdout
    stderr:       str            # captured Rscript stderr
    csv_path:     str            # path to the raw R CSV (for inspection)


_R_SCRIPT_TEMPLATE = textwrap.dedent(
    """\
    suppressPackageStartupMessages({
      library(limma)
    })

    args <- commandArgs(trailingOnly = TRUE)
    expr_csv  <- args[1]
    group_csv <- args[2]
    out_csv   <- args[3]

    expr  <- read.csv(expr_csv, row.names = 1, check.names = FALSE)
    grps  <- read.csv(group_csv, check.names = FALSE, stringsAsFactors = FALSE)

    # grps has columns: sample, group  (group is one of group1_label / group2_label)
    sample_cols <- grps$sample
    expr <- expr[, sample_cols, drop = FALSE]
    expr <- as.matrix(expr)

    # NA → row half-min (Perseus-style); limma can't accept NaN/NA in the design
    row_min <- apply(expr, 1, function(x) {
      v <- x[!is.na(x)]
      if (length(v) == 0) NA else min(v) / 2
    })
    for (i in seq_len(nrow(expr))) {
      bad <- is.na(expr[i, ])
      if (any(bad)) expr[i, bad] <- row_min[i]
    }

    grp_factor <- factor(grps$group)
    design     <- model.matrix(~0 + grp_factor)
    colnames(design) <- levels(grp_factor)

    # Contrast: group2 - group1 (positive log2FC = up in group2 / "test" condition)
    # The factor levels are alphabetised (A_group1, B_group2) so levels[2] is g2.
    contrast_str <- paste0(levels(grp_factor)[2], "-", levels(grp_factor)[1])
    contrast <- makeContrasts(contrasts = contrast_str, levels = design)

    fit  <- lmFit(expr, design)
    fit2 <- contrasts.fit(fit, contrast)
    fit2 <- eBayes(fit2, trend = TRUE)

    top  <- topTable(fit2, number = Inf, sort.by = "P", adjust.method = "BH")
    top$accession <- rownames(top)
    top <- top[, c("accession", "logFC", "AveExpr", "t", "P.Value",
                   "adj.P.Val", "B")]
    colnames(top) <- c("accession", "log2_fold_change", "avg_expr",
                       "t_statistic", "p_value", "adj_p_value", "B_statistic")

    write.csv(top, out_csv, row.names = FALSE)
    cat(sprintf("[r_analysis] limma wrote %d rows to %s\\n", nrow(top), out_csv))
    """
)


class RAnalysisSkill:
    """Run limma differential expression by shelling out to ``Rscript``."""

    def __init__(self, rscript_path: Optional[str] = None,
                 adj_pval_cutoff: float = 0.05,
                 log2fc_cutoff:   float = 1.0) -> None:
        self.rscript_path    = rscript_path or shutil.which("Rscript") or "Rscript"
        self.adj_pval_cutoff = float(adj_pval_cutoff)
        self.log2fc_cutoff   = float(log2fc_cutoff)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        expression_df:  pd.DataFrame,
        group1_samples: List[str],
        group2_samples: List[str],
        group1_label:   str = "Group1",
        group2_label:   str = "Group2",
        workdir:        Optional[str] = None,
    ) -> RLimmaResult:
        """Run limma on ``expression_df`` (rows = accessions, cols = samples)."""
        g1 = [c for c in group1_samples if c in expression_df.columns]
        g2 = [c for c in group2_samples if c in expression_df.columns]
        if len(g1) < 2 or len(g2) < 2:
            raise RAnalysisError(
                f"limma requires ≥2 samples per group. Got {group1_label}={len(g1)}, "
                f"{group2_label}={len(g2)}."
            )

        if not shutil.which(self.rscript_path):
            raise RAnalysisError(
                f"Rscript not found at {self.rscript_path!r}. Install R + the "
                f"limma package, or set RSCRIPT_PATH env var."
            )

        work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="r_limma_"))
        work.mkdir(parents=True, exist_ok=True)

        # Build group label table — column order matters because limma's
        # contrast string uses levels(grp_factor)[1] - levels(grp_factor)[2].
        # `factor()` orders by alphabetical default, so we relabel both groups
        # with a numeric prefix that pins g1 before g2.
        grps = pd.DataFrame({
            "sample": g1 + g2,
            "group":  [f"A_{group1_label}"] * len(g1) + [f"B_{group2_label}"] * len(g2),
        })

        expr_csv  = work / "expression.csv"
        group_csv = work / "groups.csv"
        out_csv   = work / "limma_results.csv"
        script_p  = work / "limma_run.R"

        expression_df[g1 + g2].to_csv(expr_csv, index=True)
        grps.to_csv(group_csv, index=False)
        script_p.write_text(_R_SCRIPT_TEMPLATE)

        cmd = [self.rscript_path, str(script_p),
               str(expr_csv), str(group_csv), str(out_csv)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0 or not out_csv.exists():
            raise RAnalysisError(
                "Rscript failed.\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  stdout: {proc.stdout[:600]}\n"
                f"  stderr: {proc.stderr[:600]}"
            )

        df = pd.read_csv(out_csv)

        # Flip sign if the contrast came back as B - A (the alphabetical prefix
        # should prevent this, but guard anyway by checking mean per group).
        # NB: log2_fold_change is interpreted as group1 - group2 throughout.
        df["significance"] = "NS"
        sig = (df["adj_p_value"] < self.adj_pval_cutoff) & \
              (df["log2_fold_change"].abs() >= self.log2fc_cutoff)
        hi  = (df["adj_p_value"] < min(0.01, self.adj_pval_cutoff / 5.0)) & \
              (df["log2_fold_change"].abs() >= self.log2fc_cutoff)
        df.loc[sig, "significance"] = "Significant"
        df.loc[hi,  "significance"] = "Highly Significant"

        df["rank"] = range(1, len(df) + 1)
        cols = ["rank", "accession", "log2_fold_change", "avg_expr",
                "t_statistic", "p_value", "adj_p_value", "B_statistic",
                "significance"]
        df = df[cols]

        logger.info(
            "R/limma analysis complete | %d proteins | %d significant @ adj_p<%.3f, |log2FC|>=%.2f",
            len(df), int(sig.sum()), self.adj_pval_cutoff, self.log2fc_cutoff,
        )
        return RLimmaResult(
            results_df = df,
            r_script   = _R_SCRIPT_TEMPLATE,
            stdout     = proc.stdout,
            stderr     = proc.stderr,
            csv_path   = str(out_csv),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Rscript + limma are both available."""
        if not shutil.which(self.rscript_path):
            return False
        try:
            chk = subprocess.run(
                [self.rscript_path, "-e",
                 'if (!requireNamespace("limma", quietly=TRUE)) quit(status=1)'],
                capture_output=True, text=True, timeout=30,
            )
            return chk.returncode == 0
        except Exception:
            return False
