"""
skills/run_visualization.py
Output Layer — ProteomicsPlotSuite

Standard proteomics visualisation library.

Standard suite (auto-run after analysis)
-----------------------------------------
Supervised (two-group DEA):
  1. volcano         — log2FC vs −log10(adj-p), coloured by significance
  2. ma_plot         — mean expression vs log2FC (MA / RA plot)
  3. heatmap         — clustered expression heatmap, top N proteins
  4. pca             — PCA of all samples, coloured by group
  5. boxplot         — top 10 proteins, one box per group
  6. sample_corr     — pairwise Pearson correlation heatmap across samples
  7. topn_bar        — horizontal bar of top N ranked proteins

Unsupervised (CV ranking):
  1. cv_distribution — histogram of CV% across all proteins
  2. heatmap         — top variable proteins
  3. pca             — PCA coloured by expression level
  4. topn_bar        — top variable proteins

Pooled fold-change:
  1. fc_heatmap      — proteins × contrasts fold-change grid
  2. rescue_bar      — rescue score bar chart
  3. topn_bar        — top proteins by rescue score

On-demand (user-requested):
  Any plot name above can be requested individually — the skill generates
  only that plot.  Requested via plot_types parameter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_SIG_COLOR   = "#C0392B"   # red — significant
_UP_COLOR    = "#E74C3C"   # up-regulated
_DOWN_COLOR  = "#2980B9"   # down-regulated
_NS_COLOR    = "#BDC3C7"   # grey — not significant
_G1_COLOR    = "#2ECC71"   # group 1
_G2_COLOR    = "#E67E22"   # group 2
_HEAT_CMAP   = "RdBu_r"


def _short_name(protein: str, max_len: int = 35) -> str:
    return str(protein).split(" OS=")[0][:max_len]


def _mpl():
    """Import matplotlib in non-interactive mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ═══════════════════════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_volcano(
    top_proteins: List[Dict],
    contrast_groups: List[str],
    stem: str,
    output_dir: str,
) -> str:
    try:
        plt = _mpl()
        df = pd.DataFrame(top_proteins)
        if "log2_fold_change" not in df.columns or "adj_p_value" not in df.columns:
            return ""
        df = df.dropna(subset=["log2_fold_change", "adj_p_value"])
        df["neg_log10_p"] = -np.log10(df["adj_p_value"].clip(lower=1e-300))
        sig_mask = (df["adj_p_value"] < 0.05) & (df["log2_fold_change"].abs() >= 1.0)
        up_mask  = sig_mask & (df["log2_fold_change"] > 0)
        dn_mask  = sig_mask & (df["log2_fold_change"] < 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df.loc[~sig_mask, "log2_fold_change"], df.loc[~sig_mask, "neg_log10_p"],
                   c=_NS_COLOR, alpha=0.5, s=18, linewidths=0, label="NS")
        ax.scatter(df.loc[up_mask, "log2_fold_change"], df.loc[up_mask, "neg_log10_p"],
                   c=_UP_COLOR, alpha=0.8, s=25, linewidths=0, label="Up")
        ax.scatter(df.loc[dn_mask, "log2_fold_change"], df.loc[dn_mask, "neg_log10_p"],
                   c=_DOWN_COLOR, alpha=0.8, s=25, linewidths=0, label="Down")
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.axvline( 1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(-1.0, color="grey", linestyle="--", linewidth=0.8)
        # Annotate top 8 significant by p-value
        for _, row in df[sig_mask].nlargest(8, "neg_log10_p").iterrows():
            ax.annotate(_short_name(row.get("protein",""), 18),
                        (row["log2_fold_change"], row["neg_log10_p"]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
        g1, g2 = (contrast_groups + ["G1", "G2"])[:2]
        ax.set_xlabel("log₂ Fold Change", fontsize=11)
        ax.set_ylabel("−log₁₀(adj. p-value)", fontsize=11)
        ax.set_title(f"Volcano — {g1} vs {g2}", fontsize=12)
        ax.legend(fontsize=9, framealpha=0.7)
        n_up = up_mask.sum(); n_dn = dn_mask.sum()
        ax.text(0.02, 0.98, f"↑{n_up}  ↓{n_dn}", transform=ax.transAxes,
                fontsize=9, va="top", color="dimgrey")
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_volcano.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Volcano plot failed: %s", exc)
        return ""


def plot_ma(
    top_proteins: List[Dict],
    contrast_groups: List[str],
    stem: str,
    output_dir: str,
) -> str:
    """MA plot: average expression (x) vs log2FC (y)."""
    try:
        plt = _mpl()
        df = pd.DataFrame(top_proteins)
        mean_cols = [c for c in df.columns if c.startswith("mean_")]
        if "log2_fold_change" not in df.columns or len(mean_cols) < 1:
            return ""
        # A = average intensity, M = log2FC
        df["A"] = df[mean_cols].mean(axis=1)
        df["M"] = df["log2_fold_change"]
        sig_mask = (df.get("adj_p_value", pd.Series(1, index=df.index)) < 0.05) & (df["M"].abs() >= 1.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df.loc[~sig_mask, "A"], df.loc[~sig_mask, "M"],
                   c=_NS_COLOR, alpha=0.5, s=18, linewidths=0, label="NS")
        ax.scatter(df.loc[sig_mask, "A"], df.loc[sig_mask, "M"],
                   c=_SIG_COLOR, alpha=0.8, s=25, linewidths=0, label="Significant")
        ax.axhline(0,    color="black",  linewidth=0.8)
        ax.axhline( 1.0, color="grey",   linestyle="--", linewidth=0.7)
        ax.axhline(-1.0, color="grey",   linestyle="--", linewidth=0.7)
        g1, g2 = (contrast_groups + ["G1", "G2"])[:2]
        ax.set_xlabel("Average log₂ Expression (A)", fontsize=11)
        ax.set_ylabel("log₂ Fold Change (M)", fontsize=11)
        ax.set_title(f"MA Plot — {g1} vs {g2}", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_ma_plot.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("MA plot failed: %s", exc)
        return ""


def plot_heatmap(
    data_path: str,
    top_proteins: List[Dict],
    sample_columns: List[str],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    top_n: int = 50,
) -> str:
    """Clustered heatmap of top N proteins, samples coloured by group."""
    try:
        import seaborn as sns
        plt = _mpl()
        if not data_path or not Path(data_path).exists():
            return ""
        df_raw = pd.read_csv(data_path, index_col=0)
        protein_names = [p.get("protein","") for p in top_proteins[:top_n]]
        protein_names = [p for p in protein_names if p in df_raw.index]
        if len(protein_names) < 3:
            return ""
        cols = [c for c in sample_columns if c in df_raw.columns]
        if not cols:
            return ""
        mat = df_raw.loc[protein_names, cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if mat.empty:
            return ""
        # Z-score rows for better colour contrast
        mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)
        # Column colour bar by group
        col_colors = pd.Series(
            {c: _G1_COLOR if c in group1_samples else _G2_COLOR for c in mat_z.columns},
            name="Group"
        )
        row_labels = [_short_name(p, 30) for p in mat_z.index]
        mat_z.index = row_labels
        g = sns.clustermap(
            mat_z,
            cmap=_HEAT_CMAP,
            center=0,
            col_colors=col_colors,
            figsize=(min(14, max(8, len(cols)*0.6)), min(18, max(6, len(mat_z)*0.25))),
            yticklabels=True,
            xticklabels=True,
            linewidths=0.0,
            dendrogram_ratio=(0.08, 0.08),
        )
        g.fig.suptitle(f"Top {len(mat_z)} Proteins — {group1_label} vs {group2_label}",
                       y=1.01, fontsize=12)
        out = str(Path(output_dir) / f"{stem}_heatmap.png")
        g.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Heatmap failed: %s", exc)
        return ""


def plot_pca(
    data_path: str,
    sample_columns: List[str],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
) -> str:
    """PCA of all samples, coloured by group."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        plt = _mpl()
        if not data_path or not Path(data_path).exists():
            return ""
        df_raw = pd.read_csv(data_path, index_col=0)
        cols = [c for c in sample_columns if c in df_raw.columns]
        if len(cols) < 3:
            return ""
        mat = df_raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0).T
        mat = mat.loc[:, mat.std() > 0]
        if mat.shape[1] < 2:
            return ""
        scaled = StandardScaler().fit_transform(mat)
        n_comp = min(2, scaled.shape[1], scaled.shape[0])
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(scaled)
        var_exp = pca.explained_variance_ratio_ * 100

        colors = [_G1_COLOR if c in group1_samples else
                  _G2_COLOR if c in group2_samples else "#888888"
                  for c in cols]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(coords[:, 0], coords[:, 1] if coords.shape[1] > 1 else [0]*len(coords),
                   c=colors, s=80, alpha=0.85, edgecolors="white", linewidths=0.8)
        for i, label in enumerate(cols):
            ax.annotate(label, (coords[i, 0],
                                coords[i, 1] if coords.shape[1] > 1 else 0),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")
        # Legend patches
        from matplotlib.patches import Patch
        legend = []
        if group1_samples:
            legend.append(Patch(color=_G1_COLOR, label=group1_label))
        if group2_samples:
            legend.append(Patch(color=_G2_COLOR, label=group2_label))
        if legend:
            ax.legend(handles=legend, fontsize=9)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)" if len(var_exp) > 1 else "PC2", fontsize=11)
        ax.set_title("PCA — Sample Separation", fontsize=12)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_pca.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("PCA plot failed: %s", exc)
        return ""


def plot_boxplot(
    data_path: str,
    top_proteins: List[Dict],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    top_n: int = 10,
) -> str:
    """Box plot of top N proteins, one box per group."""
    try:
        plt = _mpl()
        import matplotlib.pyplot as mpl_plt
        if not data_path or not Path(data_path).exists():
            return ""
        df_raw = pd.read_csv(data_path, index_col=0)
        protein_names = [p.get("protein","") for p in top_proteins[:top_n]
                         if p.get("protein","") in df_raw.index]
        if not protein_names:
            return ""
        g1 = [c for c in group1_samples if c in df_raw.columns]
        g2 = [c for c in group2_samples if c in df_raw.columns]
        if not g1 or not g2:
            return ""
        n = len(protein_names)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = mpl_plt.subplots(nrows, ncols,
                                     figsize=(ncols * 3.5, nrows * 3.2),
                                     squeeze=False)
        for idx, prot in enumerate(protein_names):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            v1 = df_raw.loc[prot, g1].apply(pd.to_numeric, errors="coerce").dropna().tolist()
            v2 = df_raw.loc[prot, g2].apply(pd.to_numeric, errors="coerce").dropna().tolist()
            bp = ax.boxplot([v1, v2],
                            patch_artist=True,
                            widths=0.5,
                            medianprops=dict(color="black", linewidth=1.5))
            bp["boxes"][0].set_facecolor(_G1_COLOR + "99")
            bp["boxes"][1].set_facecolor(_G2_COLOR + "99")
            ax.scatter([1]*len(v1), v1, color=_G1_COLOR, alpha=0.7, s=20, zorder=3)
            ax.scatter([2]*len(v2), v2, color=_G2_COLOR, alpha=0.7, s=20, zorder=3)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([group1_label, group2_label], fontsize=8)
            ax.set_title(_short_name(prot, 25), fontsize=7.5)
            ax.tick_params(axis="y", labelsize=7)
        # Hide empty axes
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)
        fig.suptitle(f"Top {n} Biomarkers — Expression by Group", fontsize=11, y=1.01)
        mpl_plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_boxplot.png")
        mpl_plt.savefig(out, dpi=150, bbox_inches="tight")
        mpl_plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Boxplot failed: %s", exc)
        return ""


def plot_sample_correlation(
    data_path: str,
    sample_columns: List[str],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
) -> str:
    """Pairwise Pearson correlation heatmap across all samples."""
    try:
        import seaborn as sns
        plt = _mpl()
        if not data_path or not Path(data_path).exists():
            return ""
        df_raw = pd.read_csv(data_path, index_col=0)
        cols = [c for c in sample_columns if c in df_raw.columns]
        if len(cols) < 3:
            return ""
        mat = df_raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        corr = mat.corr(method="pearson")
        # Column/row annotation by group
        annot_colors = pd.Series(
            {c: _G1_COLOR if c in group1_samples else
                _G2_COLOR if c in group2_samples else "#888888"
             for c in cols},
            name="Group"
        )
        g = sns.clustermap(
            corr,
            cmap="coolwarm",
            vmin=0.5, vmax=1.0,
            annot=len(cols) <= 20,
            fmt=".2f",
            annot_kws={"size": 6},
            row_colors=annot_colors,
            col_colors=annot_colors,
            figsize=(max(7, len(cols) * 0.5), max(6, len(cols) * 0.5)),
            linewidths=0.3,
        )
        g.fig.suptitle(
            f"Sample Correlation — {group1_label} vs {group2_label}", y=1.01, fontsize=11
        )
        out = str(Path(output_dir) / f"{stem}_sample_correlation.png")
        g.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Sample correlation heatmap failed: %s", exc)
        return ""


def plot_cv_distribution(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
) -> str:
    """Histogram of CV% — unsupervised analysis only."""
    try:
        plt = _mpl()
        df = pd.DataFrame(top_proteins)
        if "cv_percent" not in df.columns:
            return ""
        cv = df["cv_percent"].dropna()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(cv, bins=40, color="#3498DB", edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(cv.median(), color="#E74C3C", linestyle="--", linewidth=1.2,
                   label=f"Median CV: {cv.median():.1f}%")
        ax.set_xlabel("CV (%)", fontsize=11)
        ax.set_ylabel("Number of Proteins", fontsize=11)
        ax.set_title("Coefficient of Variation Distribution", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_cv_distribution.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("CV distribution plot failed: %s", exc)
        return ""


def plot_fc_heatmap(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 40,
) -> str:
    """Proteins × contrasts fold-change grid — pooled analysis."""
    try:
        import seaborn as sns
        plt = _mpl()
        df = pd.DataFrame(top_proteins).head(top_n)
        fc_cols = [c for c in df.columns
                   if "_vs_" in str(c) and c not in ("rank", "protein", "rescue_score")]
        if not fc_cols or df.empty:
            return ""
        labels = [_short_name(p, 30) for p in df["protein"]]
        mat = df[fc_cols].set_index(pd.Index(labels))
        fig_h = max(6, len(mat) * 0.28)
        fig_w = max(5, len(fc_cols) * 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            mat,
            cmap=_HEAT_CMAP,
            center=0,
            annot=len(mat) <= 25,
            fmt=".2f",
            annot_kws={"size": 6},
            linewidths=0.3,
            ax=ax,
            cbar_kws={"label": "log₂ FC", "shrink": 0.6},
        )
        ax.set_title("Fold-Change Heatmap — All Contrasts", fontsize=12)
        ax.set_xlabel("Contrast", fontsize=10)
        ax.set_ylabel("Protein", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_fc_heatmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("FC heatmap failed: %s", exc)
        return ""


def plot_topn_bar(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 20,
) -> str:
    """Horizontal bar chart of top N proteins by fold-change, CV, or rescue score."""
    try:
        plt = _mpl()
        df = pd.DataFrame(top_proteins).head(top_n)
        if df.empty:
            return ""
        value_col = (
            "log2_fold_change" if "log2_fold_change" in df.columns else
            "rescue_score"     if "rescue_score"     in df.columns else
            "cv_percent"       if "cv_percent"        in df.columns else None
        )
        if value_col is None:
            return ""
        labels = [_short_name(p, 35) for p in df["protein"]]
        values = df[value_col].values
        if value_col == "log2_fold_change":
            colors = [_UP_COLOR if v > 0 else _DOWN_COLOR for v in values]
        else:
            colors = "#3498DB"
        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
        ax.barh(labels[::-1], values[::-1],
                color=colors[::-1] if isinstance(colors, list) else colors,
                edgecolor="white", linewidth=0.3)
        if value_col == "log2_fold_change":
            ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(value_col.replace("_", " ").title(), fontsize=10)
        ax.set_title(f"Top {top_n} Biomarkers", fontsize=12)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_topn_bar.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Top-N bar chart failed: %s", exc)
        return ""


def plot_rescue_bar(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 20,
) -> str:
    try:
        plt = _mpl()
        df = pd.DataFrame(top_proteins).head(top_n)
        if "rescue_score" not in df.columns:
            return ""
        labels = [_short_name(p, 35) for p in df["protein"]]
        scores = df["rescue_score"].values
        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
        ax.barh(labels[::-1], scores[::-1], color="#2E86AB", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Rescue Score", fontsize=10)
        ax.set_title(f"Top {top_n} Rescued Proteins", fontsize=12)
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        out = str(Path(output_dir) / f"{stem}_rescue_bar.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out
    except Exception as exc:
        logger.warning("Rescue bar failed: %s", exc)
        return ""


def plot_pathway_dot(
    pathways: List[Dict],
    stem: str,
    output_dir: str,
    top_n: int = 15,
) -> str:
    try:
        plt = _mpl()
        df = pd.DataFrame(pathways).head(top_n)
        if df.empty or "pathway" not in df.columns:
            return ""
        p_col   = "p_adjust" if "p_adjust" in df.columns else "p_value"
        g_col   = "gene_count" if "gene_count" in df.columns else None
        neg_log = -np.log10(df[p_col].clip(lower=1e-300).astype(float))
        sizes   = (df[g_col].astype(float) * 10).clip(lower=20) if g_col else [80]*len(df)
        labels  = [str(t)[:55] for t in df["pathway"]]
        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.42)))
        sc = ax.scatter(neg_log[::-1], labels[::-1],
                        s=list(sizes)[::-1], c=neg_log[::-1],
                        cmap="RdYlBu_r", alpha=0.85, edgecolors="grey", linewidths=0.4)
        plt.colorbar(sc, ax=ax, label="−log₁₀(adj. p)", shrink=0.6)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — maps user-facing plot name → function
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_REGISTRY: Dict[str, Any] = {
    "volcano":           plot_volcano,
    "ma_plot":           plot_ma,
    "heatmap":           plot_heatmap,
    "pca":               plot_pca,
    "boxplot":           plot_boxplot,
    "sample_correlation": plot_sample_correlation,
    "cv_distribution":   plot_cv_distribution,
    "fc_heatmap":        plot_fc_heatmap,
    "topn_bar":          plot_topn_bar,
    "rescue_bar":        plot_rescue_bar,
    "pathway_dotplot":   plot_pathway_dot,
}

# Aliases for natural-language references
PLOT_ALIASES: Dict[str, str] = {
    "volcano plot":        "volcano",
    "ma plot":             "ma_plot",
    "ma-plot":             "ma_plot",
    "heatmap":             "heatmap",
    "heat map":            "heatmap",
    "pca":                 "pca",
    "pca plot":            "pca",
    "principal component": "pca",
    "box plot":            "boxplot",
    "boxplot":             "boxplot",
    "violin":              "boxplot",
    "correlation":         "sample_correlation",
    "sample correlation":  "sample_correlation",
    "corr":                "sample_correlation",
    "cv distribution":     "cv_distribution",
    "cv plot":             "cv_distribution",
    "fc heatmap":          "fc_heatmap",
    "fold change heatmap": "fc_heatmap",
    "bar chart":           "topn_bar",
    "ranking":             "topn_bar",
    "top n":               "topn_bar",
    "rescue":              "rescue_bar",
    "pathway":             "pathway_dotplot",
    "pathway dot":         "pathway_dotplot",
    "dot plot":            "pathway_dotplot",
}

# Standard suite per analysis mode
_STANDARD_SUPERVISED = ["volcano", "ma_plot", "heatmap", "pca", "boxplot", "sample_correlation", "topn_bar"]
_STANDARD_UNSUPERVISED = ["cv_distribution", "heatmap", "pca", "topn_bar"]
_STANDARD_POOLED = ["fc_heatmap", "rescue_bar", "topn_bar"]


def resolve_plot_names(requested: List[str]) -> List[str]:
    """Map user-provided plot names to canonical registry keys."""
    resolved = []
    for r in requested:
        r_lower = r.lower().strip()
        if r_lower in PLOT_REGISTRY:
            resolved.append(r_lower)
        elif r_lower in PLOT_ALIASES:
            resolved.append(PLOT_ALIASES[r_lower])
        else:
            # fuzzy match
            for alias, canonical in PLOT_ALIASES.items():
                if alias in r_lower or r_lower in alias:
                    resolved.append(canonical)
                    break
    return list(dict.fromkeys(resolved))  # deduplicate preserving order


# ═══════════════════════════════════════════════════════════════════════════════
# Main skill
# ═══════════════════════════════════════════════════════════════════════════════

class ProteomicsPlotSuite:
    """
    Generates proteomics visualizations.

    Parameters
    ----------
    plot_types : list of str, optional
        Canonical plot names (from PLOT_REGISTRY).
        If None / empty, the full standard suite is generated based on analysis_mode.
    """

    def execute(
        self,
        # Analysis results
        top_proteins: List[Dict[str, Any]],
        analysis_mode: str = "supervised",          # supervised | unsupervised | pooled
        # Raw / processed data (needed for PCA, heatmap, boxplot, correlation)
        data_path: str = "",
        sample_columns: Optional[List[str]] = None,
        group1_samples: Optional[List[str]] = None,
        group2_samples: Optional[List[str]] = None,
        group1_label: str = "Group1",
        group2_label: str = "Group2",
        # Enrichment (optional)
        top_pathways: Optional[List[Dict]] = None,
        enrichment_result_path: str = "",
        # Targeting
        contrast_groups: Optional[List[str]] = None,
        plot_types: Optional[List[str]] = None,   # None = standard suite
        output_dir: str = "outputs",
        stem: str = "biomarker",
    ) -> Dict[str, Any]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sample_columns  = sample_columns  or []
        group1_samples  = group1_samples  or []
        group2_samples  = group2_samples  or []
        contrast_groups = contrast_groups or [group1_label, group2_label]
        top_proteins    = top_proteins    or []

        # ── Detect design from biomarker fields ──────────────────────────────
        is_pooled = bool(top_proteins and "rescue_score" in top_proteins[0])
        if is_pooled:
            analysis_mode = "pooled"

        # ── Determine which plots to run ─────────────────────────────────────
        if plot_types:
            to_run = resolve_plot_names(plot_types)
        elif analysis_mode == "pooled":
            to_run = _STANDARD_POOLED[:]
        elif analysis_mode == "unsupervised":
            to_run = _STANDARD_UNSUPERVISED[:]
        else:
            to_run = _STANDARD_SUPERVISED[:]

        # Add pathway plot if enrichment data present
        pathways = top_pathways or []
        if not pathways and enrichment_result_path and Path(enrichment_result_path).exists():
            try:
                pathways = pd.read_csv(enrichment_result_path).head(15).to_dict("records")
            except Exception:
                pass
        if pathways and "pathway_dotplot" not in to_run:
            to_run.append("pathway_dotplot")

        logger.info("Generating plots: %s", to_run)

        # ── Shared kwargs for data-dependent plots ─────────────────────────────
        data_kwargs = dict(
            data_path=data_path,
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            group1_label=group1_label,
            group2_label=group2_label,
            stem=stem,
            output_dir=output_dir,
        )

        # ── Generate each plot ────────────────────────────────────────────────
        plot_paths: List[str] = []

        for plot_name in to_run:
            path = ""
            try:
                if plot_name == "volcano":
                    path = plot_volcano(top_proteins, contrast_groups, stem, output_dir)
                elif plot_name == "ma_plot":
                    path = plot_ma(top_proteins, contrast_groups, stem, output_dir)
                elif plot_name == "heatmap":
                    path = plot_heatmap(top_proteins=top_proteins, **data_kwargs)
                elif plot_name == "pca":
                    path = plot_pca(**data_kwargs)
                elif plot_name == "boxplot":
                    path = plot_boxplot(top_proteins=top_proteins, **data_kwargs)
                elif plot_name == "sample_correlation":
                    path = plot_sample_correlation(**data_kwargs)
                elif plot_name == "cv_distribution":
                    path = plot_cv_distribution(top_proteins, stem, output_dir)
                elif plot_name == "fc_heatmap":
                    path = plot_fc_heatmap(top_proteins, stem, output_dir)
                elif plot_name == "topn_bar":
                    path = plot_topn_bar(top_proteins, stem, output_dir)
                elif plot_name == "rescue_bar":
                    path = plot_rescue_bar(top_proteins, stem, output_dir)
                elif plot_name == "pathway_dotplot":
                    path = plot_pathway_dot(pathways, stem, output_dir)
            except Exception as exc:
                logger.warning("Plot '%s' raised: %s", plot_name, exc)

            if path:
                plot_paths.append(path)
                logger.info("Generated: %s", path)
            else:
                logger.info("Skipped (no data): %s", plot_name)

        report_path = str(Path(output_dir) / f"{stem}_biomarker_ranking.csv")
        if top_proteins:
            pd.DataFrame(top_proteins).to_csv(report_path, index=False)

        logger.info("ProteomicsPlotSuite done: %d plots", len(plot_paths))
        return {
            "plot_paths":   plot_paths,
            "report_path":  report_path,
            "plots_run":    to_run,
        }


# ── Backwards-compatible alias ────────────────────────────────────────────────
ReportingSkill = ProteomicsPlotSuite
