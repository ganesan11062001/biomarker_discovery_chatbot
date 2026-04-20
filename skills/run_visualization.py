"""
skills/run_visualization.py
Output Layer — ReportingSkill

Pure-Python visualization using matplotlib and pandas.
No R / ggplot required.

Generates:
  • Volcano plot         (log2FC vs -log10 adj-p)  — supervised only
  • Rescue score bar     (pooled design)
  • Pathway dot plot     (if enrichment results available)
  • Biomarker ranking CSV
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportingSkill:
    """
    Generates visualizations and a ranked biomarker CSV from analysis outputs.

    Works with both supervised DEA results and pooled fold-change results.
    Falls back gracefully if individual plot generation fails.
    """

    def execute(
        self,
        dea_result_path: str,
        enrichment_result_path: Optional[str] = None,
        top_proteins: Optional[List[Dict[str, Any]]] = None,
        top_pathways: Optional[List[Dict[str, Any]]] = None,
        contrast_groups: Optional[List[str]] = None,
        disease_program: str = "",
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_paths: List[str] = []
        stem = Path(dea_result_path).stem if dea_result_path else "biomarker"

        top_proteins = top_proteins or []

        # ── Detect design type ────────────────────────────────────────────────
        is_pooled = top_proteins and "rescue_score" in top_proteins[0]

        # ── Plot 1: Volcano (supervised) or Rescue-score bar (pooled) ─────────
        if is_pooled:
            p = _plot_rescue_bar(top_proteins, stem, output_dir)
        else:
            p = _plot_volcano(top_proteins, contrast_groups, stem, output_dir)
        if p:
            plot_paths.append(p)

        # ── Plot 2: Top-N horizontal bar (works for both designs) ─────────────
        p = _plot_topn_bar(top_proteins, stem, output_dir, is_pooled=is_pooled)
        if p:
            plot_paths.append(p)

        # ── Plot 3: Pathway dot plot (optional) ───────────────────────────────
        if top_pathways:
            p = _plot_pathway_dot(top_pathways, stem, output_dir)
            if p:
                plot_paths.append(p)
        elif enrichment_result_path and Path(enrichment_result_path).exists():
            try:
                enr_df = pd.read_csv(enrichment_result_path)
                pathways_from_file = enr_df.head(15).to_dict("records")
                p = _plot_pathway_dot(pathways_from_file, stem, output_dir)
                if p:
                    plot_paths.append(p)
            except Exception as exc:
                logger.warning("Could not load enrichment file for dot plot: %s", exc)

        # ── Ranked CSV ────────────────────────────────────────────────────────
        report_path = str(Path(output_dir) / f"{stem}_biomarker_ranking.csv")
        if top_proteins:
            pd.DataFrame(top_proteins).to_csv(report_path, index=False)

        logger.info(
            "ReportingSkill done: %d plots, report=%s",
            len(plot_paths), report_path,
        )

        return {
            "plot_paths":  plot_paths,
            "report_path": report_path,
            "plain_language_summary": "",
        }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_volcano(
    top_proteins: List[Dict],
    contrast_groups: Optional[List[str]],
    stem: str,
    output_dir: str,
) -> str:
    """Volcano: log2FC on x-axis, -log10(adj-p) on y-axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.DataFrame(top_proteins)
        if "log2_fold_change" not in df.columns or "adj_p_value" not in df.columns:
            return ""

        df = df.dropna(subset=["log2_fold_change", "adj_p_value"])
        df["neg_log10_p"] = -np.log10(df["adj_p_value"].clip(lower=1e-300))

        sig_mask  = (df["adj_p_value"] < 0.05) & (df["log2_fold_change"].abs() >= 1.0)
        colors    = ["#C0392B" if s else "#AAAAAA" for s in sig_mask]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["log2_fold_change"], df["neg_log10_p"],
                   c=colors, alpha=0.7, s=25, linewidths=0)
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.axvline( 1.0,  color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(-1.0,  color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("log₂ Fold Change", fontsize=11)
        ax.set_ylabel("−log₁₀(adj. p-value)", fontsize=11)
        g1, g2 = (contrast_groups or ["Group1", "Group2"] )[:2] + ["Group1", "Group2"]
        ax.set_title(f"Volcano plot — {g1} vs {g2}", fontsize=12)

        # Label top 5 significant
        top5 = df[sig_mask].nlargest(5, "neg_log10_p")
        for _, row in top5.iterrows():
            label = str(row.get("protein", "")).split(" OS=")[0][:20]
            ax.annotate(label, (row["log2_fold_change"], row["neg_log10_p"]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_volcano.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Volcano plot failed: %s", exc)
        return ""


def _plot_rescue_bar(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 20,
) -> str:
    """Horizontal bar chart of rescue scores (pooled design)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.DataFrame(top_proteins).head(top_n)
        if "rescue_score" not in df.columns:
            return ""

        labels = [str(p).split(" OS=")[0][:35] for p in df["protein"]]
        scores = df["rescue_score"].values

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
        bars = ax.barh(labels[::-1], scores[::-1],
                       color="#2E86AB", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Rescue Score (sum of up-regulation in treatment vs mdx)", fontsize=10)
        ax.set_title(f"Top {top_n} Rescued Proteins", fontsize=12)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_rescue_bar.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Rescue bar plot failed: %s", exc)
        return ""


def _plot_topn_bar(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 20,
    is_pooled: bool = False,
) -> str:
    """Generic top-N ranking bar — works for both designs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.DataFrame(top_proteins).head(top_n)
        if df.empty:
            return ""

        value_col = "rescue_score" if is_pooled and "rescue_score" in df.columns \
                    else "log2_fold_change" if "log2_fold_change" in df.columns \
                    else None
        if value_col is None:
            return ""

        labels = [str(p).split(" OS=")[0][:35] for p in df["protein"]]
        values = df[value_col].values
        colors = ["#C0392B" if v > 0 else "#2980B9" for v in values]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
        ax.barh(labels[::-1], values[::-1],
                color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(value_col.replace("_", " ").title(), fontsize=10)
        ax.set_title(f"Top {top_n} Biomarkers", fontsize=12)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_topn_ranking.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Top-N bar chart failed: %s", exc)
        return ""


def _plot_pathway_dot(
    pathways: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 15,
) -> str:
    """Dot plot: pathway on y, -log10(p_adjust) on x, dot size = gene_count."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.DataFrame(pathways).head(top_n)
        if df.empty or "pathway" not in df.columns:
            return ""

        p_col   = "p_adjust" if "p_adjust" in df.columns else "p_value"
        g_col   = "gene_count" if "gene_count" in df.columns else None
        neg_log = -np.log10(df[p_col].clip(lower=1e-300).astype(float))
        sizes   = (df[g_col].astype(float) * 10).clip(lower=20) if g_col else [80] * len(df)
        labels  = [str(t)[:55] for t in df["pathway"]]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.42)))
        sc = ax.scatter(neg_log[::-1], labels[::-1],
                        s=list(sizes)[::-1], c=neg_log[::-1],
                        cmap="RdYlBu_r", alpha=0.85, edgecolors="grey", linewidths=0.4)
        plt.colorbar(sc, ax=ax, label="−log₁₀(adj. p-value)", shrink=0.6)
        ax.set_xlabel("−log₁₀(adj. p-value)", fontsize=10)
        ax.set_title("Pathway Enrichment", fontsize=12)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_pathway_dotplot.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Pathway dot plot failed: %s", exc)
        return ""
