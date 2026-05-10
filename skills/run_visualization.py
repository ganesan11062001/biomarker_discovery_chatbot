"""
skills/run_visualization.py
Output Layer — Plotly-based interactive + exportable proteomics visualizations.

Each plot is saved as:
  • interactive HTML  (.html) — hover, zoom, pan; shareable link
  • static PNG        (.png)  — 2× high-resolution via kaleido; for reports

16 plot types covering every standard proteomics analysis mode.

Standard suites (auto-selected by analysis type)
─────────────────────────────────────────────────
Supervised (Welch / limma):
  volcano, ma_plot, waterfall, heatmap, pca, sample_correlation,
  boxplot, violin, topn_bar

Supervised + paired design:
  volcano, waterfall, paired_lines, heatmap, pca, violin,
  sample_correlation, topn_bar

ANOVA / multi-group (>2 groups):
  anova_multigroup, waterfall, heatmap, pca, violin,
  sample_correlation, topn_bar

SILAC:
  silac_ratio_dist, heatmap, pca, sample_correlation, topn_bar

Unsupervised (CV ranking):
  cv_distribution, heatmap, pca, topn_bar

Pooled fold-change:
  fc_heatmap, rescue_bar, topn_bar

Pathway dot plot appended automatically to any suite when enrichment
results are present.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
_UP        = "#E74C3C"
_DOWN      = "#2980B9"
_NS        = "#C8D0D8"
_TREND     = "#F39C12"
_G_COLORS  = [
    "#2980B9", "#E74C3C", "#27AE60", "#8E44AD",
    "#F39C12", "#16A085", "#C0392B", "#7F8C8D",
]
_COLOR_MAP = {"Up": _UP, "Down": _DOWN, "Trend": _TREND, "NS": _NS}

_FONT      = dict(family="Arial, Helvetica, sans-serif", size=13, color="#2C3E50")
_TEMPLATE  = "plotly_white"
_MARGIN    = dict(l=90, r=50, t=80, b=80)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fc_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("log2_fold_change", "log2_ratio", "log2fc", "max_pairwise_log2fc"):
        if c in df.columns:
            return c
    return None


def _pval_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("adj_p_value", "p_value", "adj_pval"):
        if c in df.columns:
            return c
    return None


def _short(protein: str, n: int = 30) -> str:
    return str(protein).split(" OS=")[0].split("|")[-1][:n]


def _sig_label(fc_val: float, pval: float, adj_pval_cutoff: float, log2fc_cutoff: float) -> str:
    if pval < adj_pval_cutoff and abs(fc_val) >= log2fc_cutoff:
        return "Up" if fc_val > 0 else "Down"
    if pval < 0.1 and abs(fc_val) >= 0.5:
        return "Trend"
    return "NS"


def _save(fig: go.Figure, png_path: str, width: int = 1400, height: int = 900) -> str:
    """Save figure as JSON (for Streamlit), HTML (standalone), and PNG (static)."""
    # JSON — used by Streamlit's st.plotly_chart for native interactive rendering
    json_path = png_path.replace(".png", ".json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(fig.to_json())
    except Exception as exc:
        logger.debug("JSON export skipped: %s", exc)

    # HTML — self-contained file for download / sharing
    html_path = png_path.replace(".png", ".html")
    try:
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    except Exception as exc:
        logger.debug("HTML export skipped: %s", exc)

    # PNG — static thumbnail for report / quick display
    try:
        fig.write_image(png_path, width=width, height=height, scale=2)
        return png_path
    except Exception as exc:
        logger.warning("PNG export failed (kaleido): %s — returning HTML path", exc)
        return html_path if Path(html_path).exists() else ""


def _load_wide(data_path: str, cols: List[str]) -> Optional[pd.DataFrame]:
    """Load processed data CSV and return columns that exist."""
    if not data_path or not Path(data_path).exists():
        return None
    df = pd.read_csv(data_path, index_col=0)
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return None
    return df[valid].apply(pd.to_numeric, errors="coerce")


def _cluster_order(mat: np.ndarray) -> List[int]:
    """Hierarchical cluster row order (ward linkage); returns original order on failure."""
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        if mat.shape[0] < 3:
            return list(range(mat.shape[0]))
        Z = linkage(np.nan_to_num(mat), method="ward")
        return dendrogram(Z, no_plot=True)["leaves"]
    except Exception:
        return list(range(mat.shape[0]))


def _rgba(hex_color: str, alpha: float = 0.33) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) — Plotly 6 does not accept 8-digit hex."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _group_color_strip(sample_list: List[str], group_map: Dict[str, List[str]]) -> List[str]:
    """Return a hex colour for each sample based on its group."""
    palette = {g: _G_COLORS[i % len(_G_COLORS)] for i, g in enumerate(group_map)}
    return [
        next((palette[g] for g, members in group_map.items() if s in members), "#888888")
        for s in sample_list
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Volcano Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_volcano(
    top_proteins: List[Dict],
    contrast_groups: List[str],
    stem: str,
    output_dir: str,
    adj_pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    **_,
) -> str:
    try:
        df = pd.DataFrame(top_proteins)
        fc = _fc_col(df)
        pv = _pval_col(df)
        if not fc or not pv or df.empty:
            return ""
        df = df.dropna(subset=[fc, pv])
        df["neg_log10_p"] = -np.log10(df[pv].clip(lower=1e-300))
        df["sig"]   = df.apply(lambda r: _sig_label(r[fc], r[pv], adj_pval_cutoff, log2fc_cutoff), axis=1)
        df["label"] = df.get("protein", pd.Series([""] * len(df))).apply(_short)

        g1, g2 = (contrast_groups + ["Group1", "Group2"])[:2]
        fig = go.Figure()

        for sig_type, order_z in [("NS", 1), ("Trend", 2), ("Down", 3), ("Up", 3)]:
            sub = df[df["sig"] == sig_type]
            if sub.empty:
                continue
            hover = (
                "<b>%{customdata[0]}</b><br>"
                "log₂FC: %{x:.3f}<br>"
                "adj.p: %{customdata[1]}<br>"
                "−log₁₀p: %{y:.2f}<br>"
                f"Status: {sig_type}<extra></extra>"
            )
            fig.add_trace(go.Scatter(
                x=sub[fc], y=sub["neg_log10_p"],
                mode="markers",
                name=sig_type,
                marker=dict(
                    color=_COLOR_MAP[sig_type],
                    size=8 if sig_type in ("Up", "Down") else 5,
                    opacity=0.85 if sig_type in ("Up", "Down") else 0.4,
                    line=dict(width=0.5, color="white") if sig_type in ("Up", "Down") else dict(width=0),
                ),
                customdata=list(zip(
                    sub["label"],
                    sub[pv].map(lambda v: f"{v:.2e}"),
                )),
                hovertemplate=hover,
            ))

        # Threshold lines
        p_line = -np.log10(adj_pval_cutoff)
        fig.add_hline(y=p_line, line=dict(color="#95A5A6", dash="dash", width=1.2),
                      annotation_text=f"adj.p = {adj_pval_cutoff}", annotation_font_size=10)
        fig.add_vline(x= log2fc_cutoff, line=dict(color="#95A5A6", dash="dash", width=1.2))
        fig.add_vline(x=-log2fc_cutoff, line=dict(color="#95A5A6", dash="dash", width=1.2))

        # Label top 10 significant by −log10p
        top_ann = df[df["sig"].isin(["Up", "Down"])].nlargest(10, "neg_log10_p")
        for _, row in top_ann.iterrows():
            fig.add_annotation(
                x=row[fc], y=row["neg_log10_p"],
                text=row["label"], showarrow=True,
                arrowhead=2, arrowwidth=1, arrowcolor="#95A5A6",
                ax=0, ay=-22, font=dict(size=9),
            )

        n_up = (df["sig"] == "Up").sum()
        n_dn = (df["sig"] == "Down").sum()
        fig.update_layout(
            title=dict(
                text=(f"Volcano Plot  —  <b>{g1}</b> vs <b>{g2}</b><br>"
                      f"<sup>↑ {n_up} up-regulated  ·  ↓ {n_dn} down-regulated  "
                      f"  (adj.p < {adj_pval_cutoff}, |log₂FC| ≥ {log2fc_cutoff})</sup>"),
                font=dict(size=17),
            ),
            xaxis=dict(title="log₂ Fold Change", zeroline=True, zerolinewidth=1, zerolinecolor="#ECF0F1"),
            yaxis=dict(title="−log₁₀(adj. p-value)"),
            legend=dict(title="Significance"),
            template=_TEMPLATE, font=_FONT, margin=_MARGIN,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_volcano.png"))
    except Exception as exc:
        logger.warning("Volcano failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MA Plot  (mean intensity vs fold change)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ma(
    top_proteins: List[Dict],
    contrast_groups: List[str],
    stem: str,
    output_dir: str,
    adj_pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    **_,
) -> str:
    try:
        df  = pd.DataFrame(top_proteins)
        fc  = _fc_col(df)
        pv  = _pval_col(df)
        if not fc or not pv or df.empty:
            return ""
        mean_cols = [c for c in df.columns if c.startswith("mean_")]
        if not mean_cols:
            return ""
        df = df.dropna(subset=[fc, pv])
        df["A"]   = df[mean_cols].mean(axis=1)
        df["sig"] = df.apply(lambda r: _sig_label(r[fc], r[pv], adj_pval_cutoff, log2fc_cutoff), axis=1)
        df["label"] = df.get("protein", pd.Series([""] * len(df))).apply(_short)

        g1, g2 = (contrast_groups + ["Group1", "Group2"])[:2]
        fig = go.Figure()
        for sig_type in ("NS", "Trend", "Down", "Up"):
            sub = df[df["sig"] == sig_type]
            if sub.empty:
                continue
            hover = (
                "<b>%{customdata}</b><br>"
                f"A (avg log₂): %{{x:.2f}}<br>M (log₂FC): %{{y:.3f}}<br>Status: {sig_type}<extra></extra>"
            )
            fig.add_trace(go.Scatter(
                x=sub["A"], y=sub[fc],
                mode="markers",
                name=sig_type,
                marker=dict(color=_COLOR_MAP[sig_type],
                            size=7 if sig_type in ("Up", "Down") else 5,
                            opacity=0.8 if sig_type in ("Up", "Down") else 0.4),
                customdata=sub["label"],
                hovertemplate=hover,
            ))

        fig.add_hline(y=0,              line=dict(color="#2C3E50", width=1.0))
        fig.add_hline(y= log2fc_cutoff, line=dict(color="#95A5A6", dash="dash", width=1))
        fig.add_hline(y=-log2fc_cutoff, line=dict(color="#95A5A6", dash="dash", width=1))
        fig.update_layout(
            title=dict(text=f"MA Plot  —  <b>{g1}</b> vs <b>{g2}</b>", font=dict(size=17)),
            xaxis=dict(title="A  (average log₂ intensity)"),
            yaxis=dict(title="M  (log₂ Fold Change)"),
            legend=dict(title="Significance"),
            template=_TEMPLATE, font=_FONT, margin=_MARGIN,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_ma_plot.png"))
    except Exception as exc:
        logger.warning("MA plot failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Waterfall Plot  (all proteins ranked by log2FC, coloured by significance)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_waterfall(
    top_proteins: List[Dict],
    contrast_groups: List[str],
    stem: str,
    output_dir: str,
    adj_pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    **_,
) -> str:
    """
    Ranked waterfall bar chart. Every detected protein appears ranked
    by log₂FC, coloured by significance tier. Instantly shows proportion
    up/down and where the boundaries fall.
    """
    try:
        df = pd.DataFrame(top_proteins)
        fc = _fc_col(df)
        pv = _pval_col(df)
        if not fc or df.empty:
            return ""
        if pv:
            df = df.dropna(subset=[fc, pv])
            df["sig"] = df.apply(lambda r: _sig_label(r[fc], r[pv], adj_pval_cutoff, log2fc_cutoff), axis=1)
        else:
            df = df.dropna(subset=[fc])
            df["sig"] = "NS"
        df = df.sort_values(fc, ascending=False).reset_index(drop=True)
        df["rank"]  = df.index + 1
        df["label"] = df.get("protein", pd.Series([""] * len(df))).apply(lambda x: _short(x, 25))
        df["color"] = df["sig"].map(_COLOR_MAP)

        g1, g2 = (contrast_groups + ["Group1", "Group2"])[:2]

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Rank: %{x}<br>"
            "log₂FC: %{y:.3f}<br>"
            "Status: %{customdata[1]}<extra></extra>"
        )
        fig = go.Figure(go.Bar(
            x=df["rank"],
            y=df[fc],
            marker_color=df["color"],
            marker_line_width=0,
            opacity=0.85,
            customdata=list(zip(df["label"], df["sig"])),
            hovertemplate=hover,
        ))
        fig.add_hline(y=0, line=dict(color="#2C3E50", width=1.2))

        # Legend patches via invisible scatter
        for sig_type in ("Up", "Down", "Trend", "NS"):
            if sig_type in df["sig"].values:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=_COLOR_MAP[sig_type]),
                    name=sig_type, showlegend=True,
                ))

        n_up = (df["sig"] == "Up").sum()
        n_dn = (df["sig"] == "Down").sum()
        fig.update_layout(
            title=dict(
                text=(f"Waterfall Plot  —  <b>{g1}</b> vs <b>{g2}</b><br>"
                      f"<sup>Proteins ranked by log₂FC  ·  ↑ {n_up} up  ↓ {n_dn} down</sup>"),
                font=dict(size=17),
            ),
            xaxis=dict(title="Protein rank (by log₂FC)"),
            yaxis=dict(title="log₂ Fold Change"),
            bargap=0,
            showlegend=True,
            legend=dict(title="Significance"),
            template=_TEMPLATE, font=_FONT, margin=_MARGIN,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_waterfall.png"))
    except Exception as exc:
        logger.warning("Waterfall plot failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Clustered Heatmap  (top significant proteins × samples, Z-score)
# ═══════════════════════════════════════════════════════════════════════════════

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
    all_groups: Optional[Dict[str, List[str]]] = None,
    top_n: int = 50,
    **_,
) -> str:
    try:
        df_raw = _load_wide(data_path, sample_columns)
        if df_raw is None:
            return ""
        proteins = [p.get("protein", "") for p in top_proteins[:top_n]
                    if p.get("protein", "") in df_raw.index]
        if len(proteins) < 3:
            return ""
        mat = df_raw.loc[proteins]
        mat = mat.dropna(how="all")
        if mat.empty:
            return ""
        # Z-score normalise each row
        row_mean = mat.mean(axis=1)
        row_std  = mat.std(axis=1).replace(0, 1)
        mat_z    = mat.subtract(row_mean, axis=0).div(row_std, axis=0)
        mat_z    = mat_z.fillna(0)

        # Cluster rows (proteins)
        row_order = _cluster_order(mat_z.values)
        mat_z = mat_z.iloc[row_order]
        mat_z.index = [_short(p, 35) for p in mat_z.index]

        # Cluster columns (samples)
        col_order = _cluster_order(mat_z.values.T)
        mat_z = mat_z.iloc[:, col_order]
        samples = list(mat_z.columns)

        # Group map for colour strip
        group_map = all_groups or {group1_label: group1_samples, group2_label: group2_samples}
        strip_colors = _group_color_strip(samples, group_map)

        # ── Build figure: thin annotation strip (row 1) + heatmap (row 2) ──
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.04, 0.96],
            vertical_spacing=0.004,
            shared_xaxes=True,
        )

        # Group annotation strip
        group_ids = [[list(group_map.keys()).index(
            next((g for g, m in group_map.items() if s in m), list(group_map.keys())[0])
        ) for s in samples]]
        n_groups   = len(group_map)
        colorscale = [[i / max(1, n_groups - 1), _G_COLORS[i % len(_G_COLORS)]]
                      for i in range(n_groups)]
        group_hover = [[f"<b>{s}</b><br>Group: "
                        + next((g for g, m in group_map.items() if s in m), "?")
                        + "<extra></extra>"
                        for s in samples]]
        fig.add_trace(go.Heatmap(
            z=group_ids, x=samples,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=group_hover,
            zmin=0, zmax=n_groups - 1,
        ), row=1, col=1)

        # Main heatmap
        hover_z = [
            [f"<b>{mat_z.index[r]}</b><br>Sample: {samples[c]}<br>Z-score: {mat_z.values[r,c]:.2f}<extra></extra>"
             for c in range(mat_z.shape[1])]
            for r in range(mat_z.shape[0])
        ]
        fig.add_trace(go.Heatmap(
            z=mat_z.values,
            x=samples,
            y=list(mat_z.index),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Z-score", len=0.85, y=0.45),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_z,
        ), row=2, col=1)

        h = max(500, min(1200, len(mat_z) * 18 + 120))
        fig.update_layout(
            title=dict(
                text=f"Protein Expression Heatmap  —  Top {len(mat_z)} proteins  (Z-score)",
                font=dict(size=16),
            ),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=200, r=60, t=80, b=100),
            height=h,
        )
        fig.update_yaxes(tickfont=dict(size=9), row=2, col=1)
        return _save(fig, str(Path(output_dir) / f"{stem}_heatmap.png"),
                     width=1200, height=h)
    except Exception as exc:
        logger.warning("Heatmap failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PCA  (sample-level clustering, no sklearn dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pca(
    data_path: str,
    sample_columns: List[str],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    all_groups: Optional[Dict[str, List[str]]] = None,
    **_,
) -> str:
    try:
        df_raw = _load_wide(data_path, sample_columns)
        if df_raw is None or df_raw.shape[1] < 3:
            return ""
        mat = df_raw.fillna(0).T.values.astype(float)   # samples × proteins
        # Z-score each feature
        std = mat.std(axis=0)
        mat = mat[:, std > 0]
        std = std[std > 0]
        mat = (mat - mat.mean(axis=0)) / std
        if mat.shape[1] < 2:
            return ""

        # SVD-based PCA
        U, S, _ = np.linalg.svd(mat, full_matrices=False)
        coords   = U[:, :2] * S[:2]
        var_exp  = (S[:2] ** 2) / (S ** 2).sum() * 100

        samples = [c for c in sample_columns if c in df_raw.columns]
        group_map = all_groups or {group1_label: group1_samples, group2_label: group2_samples}
        palette   = {g: _G_COLORS[i % len(_G_COLORS)] for i, g in enumerate(group_map)}

        fig = go.Figure()
        for g_name, members in group_map.items():
            idx = [i for i, s in enumerate(samples) if s in members]
            if not idx:
                continue
            hover = [f"<b>{samples[i]}</b><br>PC1: {coords[i,0]:.2f}<br>PC2: {coords[i,1]:.2f}"
                     f"<br>Group: {g_name}<extra></extra>" for i in idx]
            fig.add_trace(go.Scatter(
                x=coords[idx, 0], y=coords[idx, 1],
                mode="markers+text",
                name=g_name,
                marker=dict(color=palette[g_name], size=13,
                            line=dict(width=1.5, color="white")),
                text=[samples[i] for i in idx],
                textposition="top center",
                textfont=dict(size=9),
                hovertemplate=hover,
            ))

        fig.add_hline(y=0, line=dict(color="#ECF0F1", width=1))
        fig.add_vline(x=0, line=dict(color="#ECF0F1", width=1))
        fig.update_layout(
            title=dict(text="PCA  —  Sample Separation", font=dict(size=17)),
            xaxis=dict(title=f"PC1  ({var_exp[0]:.1f}% variance)"),
            yaxis=dict(title=f"PC2  ({var_exp[1]:.1f}% variance)"),
            legend=dict(title="Group"),
            template=_TEMPLATE, font=_FONT, margin=_MARGIN,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_pca.png"))
    except Exception as exc:
        logger.warning("PCA failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Sample Correlation Heatmap  (QC — checks for batch effects and outliers)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sample_correlation(
    data_path: str,
    sample_columns: List[str],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    all_groups: Optional[Dict[str, List[str]]] = None,
    **_,
) -> str:
    try:
        df_raw = _load_wide(data_path, sample_columns)
        if df_raw is None or df_raw.shape[1] < 3:
            return ""
        mat  = df_raw.fillna(0)
        corr = mat.corr(method="pearson")
        samples = list(corr.columns)

        # Cluster
        order = _cluster_order(corr.values)
        corr  = corr.iloc[order, order]
        samples = [samples[i] for i in order]

        vmin = float(np.percentile(
            corr.values[np.triu_indices_from(corr.values, k=1)], 5
        ))
        vmin = max(0.0, round(vmin * 10) / 10)

        group_map = all_groups or {group1_label: group1_samples, group2_label: group2_samples}
        strip_colors = _group_color_strip(samples, group_map)

        # ── Two-subplot: group strip (row 1) + correlation heatmap (row 2) ──
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.04, 0.96],
            vertical_spacing=0.003,
            shared_xaxes=True,
        )
        n_groups   = len(group_map)
        colorscale_strip = [[i / max(1, n_groups - 1), _G_COLORS[i % len(_G_COLORS)]]
                            for i in range(n_groups)]
        group_ids = [[list(group_map.keys()).index(
            next((g for g, m in group_map.items() if s in m), list(group_map.keys())[0])
        ) for s in samples]]
        fig.add_trace(go.Heatmap(
            z=group_ids, x=samples,
            colorscale=colorscale_strip, showscale=False,
            hoverinfo="skip", zmin=0, zmax=max(0, n_groups - 1),
        ), row=1, col=1)

        annot_text = corr.values if len(samples) <= 20 else None
        hover_corr = [
            [f"<b>{samples[r]}</b> vs <b>{samples[c]}</b><br>Pearson r = {corr.values[r,c]:.3f}<extra></extra>"
             for c in range(len(samples))]
            for r in range(len(samples))
        ]
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=samples, y=samples,
            colorscale="RdYlGn",
            zmin=vmin, zmax=1.0,
            text=annot_text,
            texttemplate="%{text:.2f}" if annot_text is not None else None,
            textfont=dict(size=8),
            colorbar=dict(title="Pearson r", len=0.85, y=0.45),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_corr,
        ), row=2, col=1)

        sz = max(500, min(1000, len(samples) * 28 + 150))
        fig.update_layout(
            title=dict(text="Sample Correlation Heatmap  —  QC check for outliers & batch effects",
                       font=dict(size=16)),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=120, r=60, t=80, b=120),
            height=sz,
        )
        fig.update_xaxes(tickangle=-45)
        return _save(fig, str(Path(output_dir) / f"{stem}_sample_correlation.png"),
                     width=max(700, sz), height=sz)
    except Exception as exc:
        logger.warning("Sample correlation failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-Sample Boxplot  (QC — intensity distribution; checks normalization)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_boxplot(
    data_path: str,
    sample_columns: List[str],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    all_groups: Optional[Dict[str, List[str]]] = None,
    **_,
) -> str:
    """Per-sample intensity distribution — QC check for normalization uniformity."""
    try:
        df_raw = _load_wide(data_path, sample_columns)
        if df_raw is None:
            return ""
        group_map = all_groups or {group1_label: group1_samples, group2_label: group2_samples}
        palette   = {g: _G_COLORS[i % len(_G_COLORS)] for i, g in enumerate(group_map)}

        fig = go.Figure()
        seen_groups: set = set()
        for s in [c for c in sample_columns if c in df_raw.columns]:
            vals   = df_raw[s].dropna().values
            g_name = next((g for g, m in group_map.items() if s in m), "Other")
            color  = palette.get(g_name, "#888888")
            fig.add_trace(go.Box(
                y=vals, name=s,
                marker_color=color,
                line_color=color,
                fillcolor=_rgba(color, 0.33),
                boxpoints="outliers",
                hovertemplate=f"<b>{s}</b><br>Group: {g_name}<br>Value: %{{y:.2f}}<extra></extra>",
                legendgroup=g_name,
                legendgrouptitle_text=g_name if g_name not in seen_groups else None,
                showlegend=True,
            ))
            seen_groups.add(g_name)

        fig.update_layout(
            title=dict(text="Sample Intensity Distributions  —  QC: are all samples normalised uniformly?",
                       font=dict(size=15)),
            xaxis=dict(title="Sample", tickangle=-45),
            yaxis=dict(title="log₂ Intensity"),
            showlegend=True,
            legend=dict(title="Group"),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=80, r=40, t=80, b=140),
        )
        w = max(900, len(sample_columns) * 55)
        return _save(fig, str(Path(output_dir) / f"{stem}_boxplot.png"), width=w, height=700)
    except Exception as exc:
        logger.warning("Boxplot failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Violin Plot  (top protein expression distributions by group)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_violin(
    data_path: str,
    top_proteins: List[Dict],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    all_groups: Optional[Dict[str, List[str]]] = None,
    top_n: int = 9,
    **_,
) -> str:
    try:
        group_map = all_groups or {group1_label: group1_samples, group2_label: group2_samples}
        all_cols  = [c for members in group_map.values() for c in members]
        df_raw    = _load_wide(data_path, all_cols)
        if df_raw is None:
            return ""
        group_map = {k: [c for c in v if c in df_raw.columns] for k, v in group_map.items() if v}
        if not group_map:
            return ""
        proteins = [p.get("protein", "") for p in top_proteins[:top_n]
                    if p.get("protein", "") in df_raw.index]
        if not proteins:
            return ""

        n = len(proteins)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        palette = {g: _G_COLORS[i % len(_G_COLORS)] for i, g in enumerate(group_map)}

        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=[_short(p, 28) for p in proteins],
                            horizontal_spacing=0.08, vertical_spacing=0.12)

        for idx, prot in enumerate(proteins):
            r, c = divmod(idx, ncols)
            for g_name, cols in group_map.items():
                vals = df_raw.loc[prot, cols].dropna().values if prot in df_raw.index else np.array([])
                if len(vals) == 0:
                    continue
                color = palette[g_name]
                fig.add_trace(go.Violin(
                    y=vals, name=g_name,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.3,
                    marker=dict(size=5, color=color, opacity=0.6),
                    line_color=color,
                    fillcolor=_rgba(color, 0.33),
                    legendgroup=g_name,
                    showlegend=(idx == 0),
                    hovertemplate=f"<b>{_short(prot,25)}</b><br>Group: {g_name}<br>Value: %{{y:.2f}}<extra></extra>",
                ), row=r + 1, col=c + 1)

        h = max(400, nrows * 320)
        fig.update_layout(
            title=dict(text=f"Top {len(proteins)} Biomarkers  —  Expression Distribution by Group",
                       font=dict(size=16)),
            showlegend=True,
            legend=dict(title="Group"),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=60, r=40, t=100, b=60),
            height=h,
            violingap=0.2, violingroupgap=0.1,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_violin.png"),
                     width=1200, height=h)
    except Exception as exc:
        logger.warning("Violin failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Top-N Ranked Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_topn_bar(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    adj_pval_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    n: int = 30,
    **_,
) -> str:
    try:
        df = pd.DataFrame(top_proteins[:n])
        fc = _fc_col(df)
        pv = _pval_col(df)
        if df.empty:
            return ""
        df["label"] = df.get("protein", pd.Series([""] * len(df))).apply(lambda x: _short(x, 40))

        if fc and pv:
            df = df.dropna(subset=[fc, pv])
            df["sig"]   = df.apply(lambda r: _sig_label(r[fc], r[pv], adj_pval_cutoff, log2fc_cutoff), axis=1)
            df["color"] = df["sig"].map(_COLOR_MAP)
            df = df.sort_values(fc, key=abs, ascending=True)
            x_col, x_title = fc, "log₂ Fold Change"
        elif fc:
            df = df.dropna(subset=[fc])
            df["sig"]   = "NS"
            df["color"] = _G_COLORS[3]
            df = df.sort_values(fc, key=abs, ascending=True)
            x_title = "log₂ Ratio" if "log2_ratio" in df.columns else "log₂ Fold Change"
            x_col = fc
        elif "rescue_score" in df.columns:
            df["sig"]   = "Up"
            df["color"] = _UP
            df = df.sort_values("rescue_score", ascending=True)
            x_col, x_title = "rescue_score", "Rescue Score"
        elif "cv_percent" in df.columns:
            df["sig"]   = "NS"
            df["color"] = _G_COLORS[0]
            df = df.sort_values("cv_percent", ascending=True)
            x_col, x_title = "cv_percent", "CV (%)"
        else:
            return ""

        hover = (
            "<b>%{customdata[0]}</b><br>"
            f"{x_title}: %{{x:.3f}}<br>"
            "Status: %{customdata[1]}<extra></extra>"
        )
        fig = go.Figure(go.Bar(
            x=df[x_col], y=df["label"],
            orientation="h",
            marker_color=df["color"],
            marker_line_width=0,
            opacity=0.88,
            customdata=list(zip(df["label"], df.get("sig", [""] * len(df)))),
            hovertemplate=hover,
        ))
        if fc:
            fig.add_vline(x=0, line=dict(color="#2C3E50", width=1))

        h = max(400, len(df) * 22 + 120)
        fig.update_layout(
            title=dict(text=f"Top {len(df)} Biomarkers  —  Ranked by {x_title}", font=dict(size=16)),
            xaxis=dict(title=x_title),
            yaxis=dict(title="Protein", automargin=True, tickfont=dict(size=9)),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=230, r=50, t=80, b=80),
            height=h,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_topn_bar.png"), width=1100, height=h)
    except Exception as exc:
        logger.warning("Top-N bar failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Paired Lines / Spaghetti Plot  (matched before-after designs)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paired_lines(
    data_path: str,
    top_proteins: List[Dict],
    group1_samples: List[str],
    group2_samples: List[str],
    group1_label: str,
    group2_label: str,
    stem: str,
    output_dir: str,
    top_n: int = 9,
    **_,
) -> str:
    try:
        if not data_path or not Path(data_path).exists():
            return ""
        if len(group1_samples) != len(group2_samples):
            logger.info("Paired lines skipped — unequal group sizes")
            return ""
        df_raw = pd.read_csv(data_path, index_col=0)
        g1 = [c for c in group1_samples if c in df_raw.columns]
        g2 = [c for c in group2_samples if c in df_raw.columns]
        if len(g1) < 1 or len(g2) < 1:
            return ""
        proteins = [p.get("protein", "") for p in top_proteins[:top_n]
                    if p.get("protein", "") in df_raw.index]
        if not proteins:
            return ""

        n     = len(proteins)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig   = make_subplots(rows=nrows, cols=ncols,
                              subplot_titles=[_short(p, 28) for p in proteins],
                              horizontal_spacing=0.1, vertical_spacing=0.14)

        for idx, prot in enumerate(proteins):
            r, c = divmod(idx, ncols)
            if prot not in df_raw.index:
                continue
            pre  = df_raw.loc[prot, g1].apply(pd.to_numeric, errors="coerce").values
            post = df_raw.loc[prot, g2].apply(pd.to_numeric, errors="coerce").values

            for i in range(len(pre)):
                if np.isnan(pre[i]) or np.isnan(post[i]):
                    continue
                color = _UP if post[i] > pre[i] else _DOWN
                fig.add_trace(go.Scatter(
                    x=[group1_label, group2_label],
                    y=[pre[i], post[i]],
                    mode="lines+markers",
                    line=dict(color=color, width=1.2),
                    marker=dict(size=7, color=color),
                    showlegend=False,
                    hovertemplate=(f"<b>{_short(prot,25)}</b><br>"
                                   f"Subject {i+1}<br>"
                                   f"{group1_label}: {pre[i]:.2f}<br>"
                                   f"{group2_label}: {post[i]:.2f}<extra></extra>"),
                ), row=r + 1, col=c + 1)

        h = max(400, nrows * 300)
        fig.update_layout(
            title=dict(text=f"Paired Response  —  {group1_label} → {group2_label}",
                       font=dict(size=16)),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=60, r=40, t=100, b=60),
            height=h, showlegend=False,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_paired_lines.png"), width=1200, height=h)
    except Exception as exc:
        logger.warning("Paired lines failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ANOVA / Multi-group Bar Chart  (mean ± SEM per group per top protein)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_anova_multigroup(
    data_path: str,
    top_proteins: List[Dict],
    all_groups: Dict[str, List[str]],
    stem: str,
    output_dir: str,
    top_n: int = 9,
    **_,
) -> str:
    try:
        if not data_path or not Path(data_path).exists() or not all_groups:
            return ""
        df_raw  = pd.read_csv(data_path, index_col=0)
        proteins = [p.get("protein", "") for p in top_proteins[:top_n]
                    if p.get("protein", "") in df_raw.index]
        if not proteins:
            return ""
        palette = {g: _G_COLORS[i % len(_G_COLORS)] for i, g in enumerate(all_groups)}

        n     = len(proteins)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig   = make_subplots(rows=nrows, cols=ncols,
                              subplot_titles=[_short(p, 28) for p in proteins],
                              horizontal_spacing=0.10, vertical_spacing=0.14)

        for idx, prot in enumerate(proteins):
            r, c = divmod(idx, ncols)
            for g_name, cols in all_groups.items():
                valid = [col for col in cols if col in df_raw.columns]
                if not valid:
                    continue
                vals = df_raw.loc[prot, valid].apply(pd.to_numeric, errors="coerce").dropna().values
                if len(vals) == 0:
                    continue
                mean = vals.mean()
                sem  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                color = palette[g_name]
                # Bar + error bar
                fig.add_trace(go.Bar(
                    x=[g_name], y=[mean],
                    error_y=dict(type="data", array=[sem], visible=True),
                    name=g_name, marker_color=color,
                    legendgroup=g_name, showlegend=(idx == 0),
                    hovertemplate=(f"<b>{_short(prot,25)}</b><br>Group: {g_name}<br>"
                                   f"Mean: {mean:.3f}<br>SEM: ±{sem:.3f}<extra></extra>"),
                ), row=r + 1, col=c + 1)
                # Overlay individual points
                jitter = np.random.uniform(-0.2, 0.2, len(vals))
                fig.add_trace(go.Scatter(
                    x=[g_name] * len(vals),
                    y=vals,
                    mode="markers",
                    marker=dict(color=color, size=6, opacity=0.65,
                                line=dict(color="white", width=0.5)),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=r + 1, col=c + 1)

        h = max(400, nrows * 320)
        fig.update_layout(
            title=dict(text=f"Multi-group Comparison  —  Mean ± SEM  (top {len(proteins)} proteins)",
                       font=dict(size=16)),
            barmode="group",
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=70, r=40, t=100, b=60),
            height=h, legend=dict(title="Group"),
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_anova_multigroup.png"), width=1200, height=h)
    except Exception as exc:
        logger.warning("ANOVA multigroup failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CV Distribution  (unsupervised — protein variability landscape)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cv_distribution(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    **_,
) -> str:
    try:
        df = pd.DataFrame(top_proteins)
        if "cv_percent" not in df.columns or df.empty:
            return ""
        cvs    = df["cv_percent"].dropna().values
        median = float(np.median(cvs))
        top10  = df.nlargest(10, "cv_percent")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cvs, nbinsx=40,
            marker_color=_G_COLORS[0], opacity=0.75,
            hovertemplate="CV range: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            name="All proteins",
        ))
        fig.add_vline(x=median, line=dict(color="#E74C3C", width=2, dash="dash"),
                      annotation_text=f"Median CV = {median:.1f}%",
                      annotation_font_size=11)

        # Annotate top-10 high-CV proteins
        for _, row in top10.iterrows():
            fig.add_annotation(
                x=row["cv_percent"], y=0,
                text=_short(row.get("protein", ""), 20),
                showarrow=True, arrowhead=2, arrowcolor="#95A5A6",
                ax=0, ay=-35, yref="paper", font=dict(size=8),
            )

        fig.update_layout(
            title=dict(text="CV Distribution  —  Protein Variability Landscape  (unsupervised)",
                       font=dict(size=16)),
            xaxis=dict(title="Coefficient of Variation (%)"),
            yaxis=dict(title="Number of proteins"),
            template=_TEMPLATE, font=_FONT, margin=_MARGIN,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_cv_distribution.png"))
    except Exception as exc:
        logger.warning("CV distribution failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Multi-contrast FC Heatmap  (pooled — all conditions × all proteins)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fc_heatmap(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    **_,
) -> str:
    try:
        df = pd.DataFrame(top_proteins)
        if df.empty:
            return ""
        fc_cols = [c for c in df.columns if "log2fc" in c.lower() or "fold" in c.lower()
                   or "log2_ratio" in c.lower()]
        if not fc_cols:
            fc_cols = [c for c in df.columns if "vs" in c.lower() or "_fc" in c.lower()]
        if not fc_cols:
            return ""
        df["label"] = df.get("protein", pd.Series([""] * len(df))).apply(lambda x: _short(x, 40))
        df = df.dropna(subset=fc_cols, how="all")
        mat = df.set_index("label")[fc_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        row_order = _cluster_order(mat.values)
        mat = mat.iloc[row_order]

        hover = [
            [f"<b>{mat.index[r]}</b><br>Contrast: {fc_cols[c]}<br>log₂FC: {mat.values[r,c]:.3f}<extra></extra>"
             for c in range(mat.shape[1])]
            for r in range(mat.shape[0])
        ]
        fig = go.Figure(go.Heatmap(
            z=mat.values,
            x=[c.replace("_", " ") for c in fc_cols],
            y=list(mat.index),
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="log₂FC"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

        h = max(400, len(mat) * 20 + 150)
        fig.update_layout(
            title=dict(text="Multi-contrast Fold-Change Heatmap  —  Pooled Design", font=dict(size=16)),
            xaxis=dict(title="Contrast", tickangle=-35),
            yaxis=dict(title="Protein", tickfont=dict(size=9)),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=220, r=60, t=80, b=120),
            height=h,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_fc_heatmap.png"), width=1100, height=h)
    except Exception as exc:
        logger.warning("FC heatmap failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Rescue Score Bar Chart  (pooled — therapeutic biomarker candidates)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rescue_bar(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    n: int = 30,
    **_,
) -> str:
    try:
        df = pd.DataFrame(top_proteins[:n])
        if "rescue_score" not in df.columns or df.empty:
            return ""
        df["label"] = df.get("protein", pd.Series([""] * len(df))).apply(lambda x: _short(x, 40))
        df = df.sort_values("rescue_score", ascending=True)

        fig = go.Figure(go.Bar(
            x=df["rescue_score"],
            y=df["label"],
            orientation="h",
            marker=dict(
                color=df["rescue_score"],
                colorscale="YlOrRd",
                colorbar=dict(title="Rescue Score"),
                line_width=0,
            ),
            opacity=0.9,
            hovertemplate="<b>%{y}</b><br>Rescue Score: %{x:.3f}<extra></extra>",
        ))

        h = max(400, len(df) * 22 + 120)
        fig.update_layout(
            title=dict(text="Rescue Score  —  Top Therapeutic Biomarker Candidates<br>"
                            "<sup>Higher score = more dysregulated in disease AND restored by treatment</sup>",
                       font=dict(size=15)),
            xaxis=dict(title="Rescue Score  (|FC disease vs WT| + |FC treatment vs disease|)"),
            yaxis=dict(title="Protein", automargin=True, tickfont=dict(size=9)),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=230, r=80, t=100, b=80),
            height=h,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_rescue_bar.png"), width=1100, height=h)
    except Exception as exc:
        logger.warning("Rescue bar failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 15. SILAC Ratio Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_silac_ratio_dist(
    top_proteins: List[Dict],
    stem: str,
    output_dir: str,
    **_,
) -> str:
    try:
        df = pd.DataFrame(top_proteins)
        if "log2_ratio" not in df.columns or df.empty:
            return ""
        ratios = df["log2_ratio"].dropna().values
        median = float(np.median(ratios))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ratios, nbinsx=50,
            marker_color=_G_COLORS[3], opacity=0.75,
            hovertemplate="SILAC log₂ ratio: %{x:.2f}<br>Count: %{y}<extra></extra>",
            name="SILAC ratios",
        ))
        fig.add_vline(x=0,      line=dict(color="#2C3E50", width=1.5),
                      annotation_text="ratio = 0 (no change)", annotation_font_size=10)
        fig.add_vline(x=median, line=dict(color=_UP, width=2, dash="dash"),
                      annotation_text=f"Median = {median:.2f}", annotation_font_size=11)

        n_up = int((ratios > 0.5).sum())
        n_dn = int((ratios < -0.5).sum())
        fig.update_layout(
            title=dict(
                text=f"SILAC Log₂ Ratio Distribution<br>"
                     f"<sup>↑ {n_up} proteins enriched (ratio > 0.5)  ·  ↓ {n_dn} proteins depleted (ratio < −0.5)</sup>",
                font=dict(size=16),
            ),
            xaxis=dict(title="SILAC log₂ Ratio (Heavy/Light)"),
            yaxis=dict(title="Number of proteins"),
            template=_TEMPLATE, font=_FONT, margin=_MARGIN,
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_silac_ratio_dist.png"))
    except Exception as exc:
        logger.warning("SILAC ratio dist failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Pathway Enrichment Dot Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pathway_dot(
    pathways: List[Dict],
    stem: str,
    output_dir: str,
    **_,
) -> str:
    try:
        if not pathways:
            return ""
        df = pd.DataFrame(pathways)
        if "pathway" not in df.columns:
            return ""
        for col in ("p_adjust", "p_value", "Adjusted P-value", "P-value"):
            if col in df.columns:
                df["p_adj"] = pd.to_numeric(df[col], errors="coerce")
                break
        if "p_adj" not in df.columns:
            return ""
        for col in ("gene_count", "gene_ratio", "Count", "Overlap"):
            if col in df.columns:
                df["gcount"] = pd.to_numeric(df[col], errors="coerce").fillna(5)
                break
        if "gcount" not in df.columns:
            df["gcount"] = 8
        df = df.dropna(subset=["p_adj"]).sort_values("p_adj").head(25)
        df["neg_log10_p"] = -np.log10(df["p_adj"].clip(lower=1e-300))
        df["label"]       = df["pathway"].apply(lambda x: str(x)[:55])
        df["direction"]   = df.get("direction", pd.Series(["all"] * len(df)))

        # Map direction to colour
        dir_color = {"up": _UP, "down": _DOWN, "all": _G_COLORS[0]}
        df["color"] = df["direction"].map(lambda d: dir_color.get(str(d).lower(), _G_COLORS[0]))
        df = df.sort_values("neg_log10_p", ascending=True)

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "adj.p: %{customdata[1]}<br>"
            "Gene count: %{customdata[2]}<br>"
            "Direction: %{customdata[3]}<extra></extra>"
        )
        fig = go.Figure(go.Scatter(
            x=df["neg_log10_p"],
            y=df["label"],
            mode="markers",
            marker=dict(
                size=df["gcount"].clip(upper=30) * 1.5 + 8,
                color=df["color"],
                opacity=0.82,
                line=dict(width=0.8, color="white"),
            ),
            customdata=list(zip(
                df["label"],
                df["p_adj"].map(lambda v: f"{v:.2e}"),
                df["gcount"].astype(int),
                df["direction"],
            )),
            hovertemplate=hover,
        ))
        fig.add_vline(x=-np.log10(0.05), line=dict(color="#95A5A6", dash="dash", width=1),
                      annotation_text="adj.p = 0.05", annotation_font_size=10)

        has_direction = df["direction"].ne("all").any()
        if has_direction:
            for d, color in dir_color.items():
                if d != "all" and d in df["direction"].values:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode="markers",
                        marker=dict(size=10, color=color),
                        name=d.capitalize(), showlegend=True,
                    ))

        h = max(400, len(df) * 24 + 130)
        fig.update_layout(
            title=dict(text="Pathway Enrichment  —  Dot Plot<br>"
                            "<sup>Dot size = gene count  ·  x-axis = −log₁₀(adj. p-value)</sup>",
                       font=dict(size=16)),
            xaxis=dict(title="−log₁₀(adjusted p-value)"),
            yaxis=dict(title="", automargin=True, tickfont=dict(size=10)),
            template=_TEMPLATE, font=_FONT,
            margin=dict(l=360, r=50, t=100, b=80),
            height=h,
            showlegend=bool(has_direction),
        )
        return _save(fig, str(Path(output_dir) / f"{stem}_pathway_dotplot.png"), width=1200, height=h)
    except Exception as exc:
        logger.warning("Pathway dot plot failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Registry  &  Aliases
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_REGISTRY: Dict[str, Any] = {
    "volcano":            plot_volcano,
    "ma_plot":            plot_ma,
    "waterfall":          plot_waterfall,
    "heatmap":            plot_heatmap,
    "pca":                plot_pca,
    "sample_correlation": plot_sample_correlation,
    "boxplot":            plot_boxplot,
    "violin":             plot_violin,
    "topn_bar":           plot_topn_bar,
    "paired_lines":       plot_paired_lines,
    "anova_multigroup":   plot_anova_multigroup,
    "cv_distribution":    plot_cv_distribution,
    "fc_heatmap":         plot_fc_heatmap,
    "rescue_bar":         plot_rescue_bar,
    "silac_ratio_dist":   plot_silac_ratio_dist,
    "pathway_dotplot":    plot_pathway_dot,
}

PLOT_ALIASES: Dict[str, str] = {
    # Volcano
    "volcano plot":             "volcano",
    # MA plot
    "ma plot":                  "ma_plot",
    "ma-plot":                  "ma_plot",
    "bland-altman":             "ma_plot",
    # Waterfall
    "waterfall":                "waterfall",
    "waterfall plot":           "waterfall",
    "ranked proteins":          "waterfall",
    # Heatmap
    "heatmap":                  "heatmap",
    "heat map":                 "heatmap",
    "protein heatmap":          "heatmap",
    "expression heatmap":       "heatmap",
    # PCA
    "pca":                      "pca",
    "pca plot":                 "pca",
    "principal component":      "pca",
    "principal components":     "pca",
    "scatter plot":             "pca",
    # Sample correlation
    "correlation":              "sample_correlation",
    "sample correlation":       "sample_correlation",
    "correlation heatmap":      "sample_correlation",
    "corr":                     "sample_correlation",
    # Boxplot
    "box plot":                 "boxplot",
    "boxplot":                  "boxplot",
    "sample distribution":      "boxplot",
    "intensity distribution":   "boxplot",
    # Violin
    "violin":                   "violin",
    "violin plot":              "violin",
    "violin plots":             "violin",
    # Top-N bar
    "bar chart":                "topn_bar",
    "ranking":                  "topn_bar",
    "top n":                    "topn_bar",
    "top proteins":             "topn_bar",
    "ranked bar":               "topn_bar",
    # Paired lines
    "paired":                   "paired_lines",
    "spaghetti":                "paired_lines",
    "spaghetti plot":           "paired_lines",
    "before after":             "paired_lines",
    "paired lines":             "paired_lines",
    "matched":                  "paired_lines",
    # ANOVA multigroup
    "anova":                    "anova_multigroup",
    "multi group":              "anova_multigroup",
    "multigroup":               "anova_multigroup",
    "multi-group":              "anova_multigroup",
    "group comparison":         "anova_multigroup",
    # CV distribution
    "cv distribution":          "cv_distribution",
    "cv plot":                  "cv_distribution",
    "variability":              "cv_distribution",
    "cv histogram":             "cv_distribution",
    # FC heatmap
    "fc heatmap":               "fc_heatmap",
    "fold change heatmap":      "fc_heatmap",
    "contrast heatmap":         "fc_heatmap",
    # Rescue bar
    "rescue":                   "rescue_bar",
    "rescue score":             "rescue_bar",
    "rescue bar":               "rescue_bar",
    "therapeutic biomarkers":   "rescue_bar",
    # SILAC
    "silac":                    "silac_ratio_dist",
    "ratio distribution":       "silac_ratio_dist",
    "silac distribution":       "silac_ratio_dist",
    # Pathway
    "pathway":                  "pathway_dotplot",
    "pathway dot":              "pathway_dotplot",
    "dot plot":                 "pathway_dotplot",
    "enrichment":               "pathway_dotplot",
    "pathway enrichment":       "pathway_dotplot",
}


# ── Standard plot suites per analysis mode ────────────────────────────────────

_STANDARD_SUPERVISED = [
    "volcano", "ma_plot", "waterfall",
    "heatmap", "pca", "sample_correlation",
    "boxplot", "violin", "topn_bar",
]

_STANDARD_SUPERVISED_PAIRED = [
    "volcano", "waterfall", "paired_lines",
    "heatmap", "pca", "violin",
    "sample_correlation", "topn_bar",
]

_STANDARD_SUPERVISED_ANOVA = [
    "anova_multigroup", "waterfall",
    "heatmap", "pca", "violin",
    "sample_correlation", "topn_bar",
]

_STANDARD_SILAC = [
    "silac_ratio_dist",
    "heatmap", "pca", "sample_correlation", "topn_bar",
]

_STANDARD_UNSUPERVISED = [
    "cv_distribution",
    "heatmap", "pca", "topn_bar",
]

_STANDARD_POOLED = [
    "fc_heatmap", "rescue_bar", "topn_bar",
]


# ── Resolve user-requested plot names (string → canonical) ───────────────────

def resolve_plot_names(requested: List[str]) -> List[str]:
    resolved = []
    for r in requested:
        r_lower = r.lower().strip()
        if r_lower in PLOT_REGISTRY:
            resolved.append(r_lower)
        elif r_lower in PLOT_ALIASES:
            resolved.append(PLOT_ALIASES[r_lower])
        else:
            for alias, canonical in PLOT_ALIASES.items():
                if r_lower in alias or alias in r_lower:
                    resolved.append(canonical)
                    break
    seen: set = set()
    return [p for p in resolved if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]


# ═══════════════════════════════════════════════════════════════════════════════
# ProteomicsPlotSuite — main entry point
# ═══════════════════════════════════════════════════════════════════════════════

class ProteomicsPlotSuite:
    """
    Selects and runs the appropriate plot suite based on analysis type.

    Returns:
        dict with plot_paths (list), report_path (str), plots_run (list)
    """

    def execute(
        self,
        top_proteins: List[Dict[str, Any]],
        analysis_mode: str = "supervised",
        omic_type: str = "proteomics",
        test_method: str = "welch",
        is_paired: bool = False,
        all_groups: Optional[Dict[str, List[str]]] = None,
        # Raw data
        data_path: str = "",
        sample_columns: Optional[List[str]] = None,
        group1_samples: Optional[List[str]] = None,
        group2_samples: Optional[List[str]] = None,
        group1_label: str = "Group1",
        group2_label: str = "Group2",
        # Thresholds from actual analysis
        adj_pval_cutoff: float = 0.05,
        log2fc_cutoff: float = 1.0,
        # Enrichment
        top_pathways: Optional[List[Dict]] = None,
        enrichment_result_path: str = "",
        # Targeting
        contrast_groups: Optional[List[str]] = None,
        plot_types: Optional[List[str]] = None,
        output_dir: str = "outputs",
        stem: str = "biomarker",
    ) -> Dict[str, Any]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sample_columns  = sample_columns  or []
        group1_samples  = group1_samples  or []
        group2_samples  = group2_samples  or []
        contrast_groups = contrast_groups or [group1_label, group2_label]
        top_proteins    = top_proteins    or []

        # Auto-detect mode from biomarker fields
        if top_proteins:
            if "rescue_score" in top_proteins[0]:
                analysis_mode = "pooled"
            elif "log2_ratio" in top_proteins[0]:
                omic_type = "proteomics_silac"

        # Select suite
        if plot_types:
            to_run = resolve_plot_names(plot_types)
        elif analysis_mode == "pooled":
            to_run = _STANDARD_POOLED[:]
        elif analysis_mode == "unsupervised":
            to_run = _STANDARD_UNSUPERVISED[:]
        elif omic_type == "proteomics_silac":
            to_run = _STANDARD_SILAC[:]
        elif test_method == "anova" or (all_groups and len(all_groups) > 2):
            to_run = _STANDARD_SUPERVISED_ANOVA[:]
        elif is_paired:
            to_run = _STANDARD_SUPERVISED_PAIRED[:]
        else:
            to_run = _STANDARD_SUPERVISED[:]

        # Append pathway dot plot if enrichment data available
        pathways = top_pathways or []
        if not pathways and enrichment_result_path and Path(enrichment_result_path).exists():
            try:
                pathways = pd.read_csv(enrichment_result_path).head(25).to_dict("records")
            except Exception:
                pass
        if pathways and "pathway_dotplot" not in to_run:
            to_run.append("pathway_dotplot")

        logger.info("Generating %d plots for %s / %s: %s",
                    len(to_run), analysis_mode, omic_type, to_run)

        # Shared data kwargs
        data_kwargs = dict(
            data_path=data_path,
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            group1_label=group1_label,
            group2_label=group2_label,
            stem=stem,
            output_dir=output_dir,
            all_groups=all_groups or None,
        )

        plot_paths: List[str] = []

        for plot_name in to_run:
            path = ""
            try:
                if plot_name == "volcano":
                    path = plot_volcano(top_proteins, contrast_groups, stem, output_dir,
                                        adj_pval_cutoff, log2fc_cutoff)
                elif plot_name == "ma_plot":
                    path = plot_ma(top_proteins, contrast_groups, stem, output_dir,
                                   adj_pval_cutoff, log2fc_cutoff)
                elif plot_name == "waterfall":
                    path = plot_waterfall(top_proteins, contrast_groups, stem, output_dir,
                                          adj_pval_cutoff, log2fc_cutoff)
                elif plot_name == "heatmap":
                    path = plot_heatmap(top_proteins=top_proteins, **data_kwargs)
                elif plot_name == "pca":
                    path = plot_pca(**data_kwargs)
                elif plot_name == "sample_correlation":
                    path = plot_sample_correlation(**data_kwargs)
                elif plot_name == "boxplot":
                    path = plot_boxplot(**data_kwargs)
                elif plot_name == "violin":
                    path = plot_violin(top_proteins=top_proteins, **data_kwargs)
                elif plot_name == "topn_bar":
                    path = plot_topn_bar(top_proteins, stem, output_dir,
                                         adj_pval_cutoff=adj_pval_cutoff,
                                         log2fc_cutoff=log2fc_cutoff)
                elif plot_name == "paired_lines":
                    path = plot_paired_lines(top_proteins=top_proteins, **data_kwargs)
                elif plot_name == "anova_multigroup":
                    if all_groups:
                        path = plot_anova_multigroup(
                            data_path, top_proteins, all_groups, stem, output_dir
                        )
                elif plot_name == "cv_distribution":
                    path = plot_cv_distribution(top_proteins, stem, output_dir)
                elif plot_name == "fc_heatmap":
                    path = plot_fc_heatmap(top_proteins, stem, output_dir)
                elif plot_name == "rescue_bar":
                    path = plot_rescue_bar(top_proteins, stem, output_dir)
                elif plot_name == "silac_ratio_dist":
                    path = plot_silac_ratio_dist(top_proteins, stem, output_dir)
                elif plot_name == "pathway_dotplot":
                    path = plot_pathway_dot(pathways, stem, output_dir)
            except Exception as exc:
                logger.warning("Plot '%s' raised: %s", plot_name, exc)

            if path:
                plot_paths.append(path)
                logger.info("Generated: %s", Path(path).name)
            else:
                logger.info("Skipped (no data): %s", plot_name)

        report_path = str(Path(output_dir) / f"{stem}_biomarker_ranking.csv")
        if top_proteins:
            pd.DataFrame(top_proteins).to_csv(report_path, index=False)

        logger.info("ProteomicsPlotSuite done: %d / %d plots generated",
                    len(plot_paths), len(to_run))
        return {
            "plot_paths":  plot_paths,
            "report_path": report_path,
            "plots_run":   to_run,
        }
