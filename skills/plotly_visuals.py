"""
skills/plotly_visuals.py
Interactive Plotly visualisations for the canonical proteomics pipeline.

Every plot:
  • uses **Accession Number** as the primary point label (protein names
    are too long for plot annotations)
  • exposes the protein name + gene symbol + log2FC + adj.p in the
    hover tooltip
  • is exported to both **interactive HTML** (Plotly) and **static PNG**
    (via kaleido) so the figures can be embedded in reports

Public API
----------
``build_visualisation_suite(...)`` runs every supported plot type and
returns a dict of ``{plot_name: {"html": path, "png": path}}``. Individual
``build_*`` functions exist for callers that only want one chart.

Plot types
~~~~~~~~~~
- volcano        : log2FC vs −log10(adj.p), top hits labelled by accession
- ma             : log2FC vs mean expression
- pca            : sample-level PCA scatter coloured by group
- heatmap        : top-N proteins × samples (z-scored)
- box            : per-protein boxplot across groups (top hits)
- engine_scatter : Python log2FC vs R log2FC for the dual-engine intersection
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# Default colour scheme — colourblind-safe
_COLOURS = {
    "up":      "#D62728",   # red
    "down":    "#1F77B4",   # blue
    "ns":      "#B0B0B0",   # grey
    "highly":  "#2CA02C",   # green
    "trend":   "#FF7F0E",   # orange
}


def _save_fig(fig: go.Figure, out_dir: Path, name: str) -> Dict[str, str]:
    """Write fig as both interactive HTML and static PNG. Return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{name}.html"
    png_path  = out_dir / f"{name}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=False)
    try:
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
    except Exception as exc:
        logger.warning("PNG export failed for %s (kaleido issue?): %s", name, exc)
        png_path = None  # type: ignore[assignment]
    return {
        "html": str(html_path),
        "png":  str(png_path) if png_path else "",
    }


# ── Plot 1: Volcano ──────────────────────────────────────────────────────────

def build_volcano(
    results_df:      pd.DataFrame,
    out_dir:         Path,
    title:           str = "Volcano plot",
    adj_pval_cutoff: float = 0.05,
    log2fc_cutoff:   float = 1.0,
    top_n_label:     int   = 20,
    accession_col:   str   = "accession",
) -> Dict[str, str]:
    """Volcano with Accession labels on the top hits."""
    df = results_df.copy()
    df["-log10_adj_p"] = -np.log10(df["adj_p_value"].clip(lower=1e-300))
    df["status"] = "NS"
    up = (df["adj_p_value"] < adj_pval_cutoff) & (df["log2_fold_change"] >=  log2fc_cutoff)
    dn = (df["adj_p_value"] < adj_pval_cutoff) & (df["log2_fold_change"] <= -log2fc_cutoff)
    df.loc[up, "status"] = "Up"
    df.loc[dn, "status"] = "Down"

    if accession_col not in df.columns and "protein" in df.columns:
        df = df.rename(columns={"protein": accession_col})

    fig = px.scatter(
        df, x="log2_fold_change", y="-log10_adj_p",
        color="status",
        color_discrete_map={"Up": _COLOURS["up"],
                            "Down": _COLOURS["down"],
                            "NS": _COLOURS["ns"]},
        hover_data={
            accession_col:        True,
            "log2_fold_change":   ":.3f",
            "adj_p_value":        ":.2e",
            "-log10_adj_p":       False,
            "status":             False,
        },
        title=title,
        labels={"log2_fold_change": "log₂ fold change",
                "-log10_adj_p":     "−log₁₀ adj. p"},
    )

    # Significance threshold lines
    fig.add_hline(y=-np.log10(adj_pval_cutoff), line_dash="dash",
                  line_color="gray", opacity=0.6)
    for x in (-log2fc_cutoff, log2fc_cutoff):
        fig.add_vline(x=x, line_dash="dash", line_color="gray", opacity=0.6)

    # Label the top hits by accession
    hits = df.loc[(up | dn)].sort_values("adj_p_value").head(top_n_label)
    for _, row in hits.iterrows():
        fig.add_annotation(
            x=row["log2_fold_change"], y=row["-log10_adj_p"],
            text=str(row[accession_col]),
            showarrow=True, arrowhead=1, arrowsize=0.5,
            font=dict(size=9), bgcolor="white", opacity=0.85,
        )

    fig.update_layout(
        template="plotly_white", height=700, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _save_fig(fig, out_dir, "volcano")


# ── Plot 2: MA plot ──────────────────────────────────────────────────────────

def build_ma_plot(
    results_df: pd.DataFrame,
    out_dir:    Path,
    title:      str = "MA plot",
    adj_pval_cutoff: float = 0.05,
    top_n_label:    int  = 15,
    accession_col:  str  = "accession",
) -> Dict[str, str]:
    """log2FC vs mean expression (A vs M)."""
    df = results_df.copy()
    if accession_col not in df.columns and "protein" in df.columns:
        df = df.rename(columns={"protein": accession_col})

    # Pick a mean-expression column — limma exports avg_expr; Python uses mean_*
    if "avg_expr" in df.columns:
        x_col = "avg_expr"
    else:
        mean_cols = [c for c in df.columns if c.startswith("mean_")]
        df["mean_expr"] = df[mean_cols].mean(axis=1) if mean_cols else df.get("avg_expr", 0.0)
        x_col = "mean_expr"

    df["significant"] = df["adj_p_value"] < adj_pval_cutoff

    fig = px.scatter(
        df, x=x_col, y="log2_fold_change",
        color="significant",
        color_discrete_map={True: _COLOURS["up"], False: _COLOURS["ns"]},
        hover_data={accession_col: True,
                    "log2_fold_change": ":.3f",
                    "adj_p_value": ":.2e"},
        title=title,
        labels={x_col: "Mean log₂ expression",
                "log2_fold_change": "log₂ fold change",
                "significant": f"adj.p < {adj_pval_cutoff}"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4)

    hits = df.loc[df["significant"]].sort_values("adj_p_value").head(top_n_label)
    for _, row in hits.iterrows():
        fig.add_annotation(x=row[x_col], y=row["log2_fold_change"],
                           text=str(row[accession_col]),
                           showarrow=True, arrowhead=1, arrowsize=0.5,
                           font=dict(size=9), bgcolor="white", opacity=0.85)

    fig.update_layout(template="plotly_white", height=600, width=1000)
    return _save_fig(fig, out_dir, "ma_plot")


# ── Plot 3: PCA ──────────────────────────────────────────────────────────────

def build_pca(
    expression_df:    pd.DataFrame,
    sample_to_group:  Dict[str, str],
    out_dir:          Path,
    title:            str = "PCA — samples",
) -> Dict[str, str]:
    """Sample-level PCA scatter. Rows = proteins, cols = samples."""
    from sklearn.decomposition import PCA

    samples = [s for s in expression_df.columns if s in sample_to_group]
    if len(samples) < 3:
        raise ValueError("PCA requires ≥3 samples mapped to a group.")

    X = expression_df[samples].T.fillna(expression_df[samples].T.mean())
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.values)

    df = pd.DataFrame({
        "Sample": samples,
        "PC1":    coords[:, 0],
        "PC2":    coords[:, 1],
        "Group":  [sample_to_group.get(s, "?") for s in samples],
    })
    ev = pca.explained_variance_ratio_

    fig = px.scatter(
        df, x="PC1", y="PC2", color="Group", text="Sample",
        title=f"{title}  (PC1 {ev[0]*100:.1f}%, PC2 {ev[1]*100:.1f}%)",
        labels={"PC1": f"PC1 ({ev[0]*100:.1f}%)",
                "PC2": f"PC2 ({ev[1]*100:.1f}%)"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1)))
    fig.update_layout(template="plotly_white", height=700, width=1000)
    return _save_fig(fig, out_dir, "pca")


# ── Plot 4: Heatmap (top-N proteins) ─────────────────────────────────────────

def build_heatmap(
    expression_df:   pd.DataFrame,
    top_accessions:  Sequence[str],
    sample_to_group: Dict[str, str],
    out_dir:         Path,
    title:           str = "Top biomarkers — z-scored heatmap",
) -> Dict[str, str]:
    """Z-scored heatmap of top-N proteins, columns ordered by group."""
    samples = [s for s in expression_df.columns if s in sample_to_group]
    samples = sorted(samples, key=lambda s: (sample_to_group.get(s, ""), s))
    sub = expression_df.loc[
        [a for a in top_accessions if a in expression_df.index], samples
    ].astype(float)
    if sub.empty:
        raise ValueError("None of the requested accessions are in the expression matrix.")

    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)

    fig = go.Figure(data=go.Heatmap(
        z=z.values,
        x=z.columns.tolist(),
        y=z.index.tolist(),                       # accession numbers!
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="z-score"),
        hovertemplate=(
            "Accession: %{y}<br>Sample: %{x}<br>z-score: %{z:.2f}<extra></extra>"
        ),
    ))

    # Annotate group bands on top
    groups = list(dict.fromkeys(sample_to_group.get(s, "?") for s in samples))
    band_height = 0.03
    for i, sample in enumerate(samples):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=len(z) - 0.5, y1=len(z) - 0.5 + band_height * len(z),
            yref="y", xref="x",
            fillcolor=px.colors.qualitative.Set2[
                groups.index(sample_to_group.get(sample, "?")) % 8
            ],
            line=dict(width=0), opacity=0.5,
        )

    fig.update_layout(
        title=title, template="plotly_white",
        height=max(500, 22 * len(z) + 200), width=max(900, 60 * len(samples) + 200),
        xaxis=dict(tickangle=-60),
        yaxis=dict(autorange="reversed"),
    )
    return _save_fig(fig, out_dir, "heatmap")


# ── Plot 5: Per-protein boxplots ─────────────────────────────────────────────

def build_top_boxplots(
    expression_df:    pd.DataFrame,
    top_accessions:   Sequence[str],
    sample_to_group:  Dict[str, str],
    out_dir:          Path,
    title:            str = "Top biomarkers — distribution per group",
    cols:             int = 4,
) -> Dict[str, str]:
    """Faceted boxplots of the top-N proteins by accession."""
    accs = [a for a in top_accessions if a in expression_df.index][:12]
    if not accs:
        raise ValueError("None of the requested accessions are in the expression matrix.")

    rows = (len(accs) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=accs)

    samples = [s for s in expression_df.columns if s in sample_to_group]
    groups  = [sample_to_group[s] for s in samples]
    palette = {g: px.colors.qualitative.Set2[i % 8]
               for i, g in enumerate(dict.fromkeys(groups))}

    for idx, acc in enumerate(accs):
        r = idx // cols + 1; c = idx % cols + 1
        values = expression_df.loc[acc, samples].astype(float).values
        for g in dict.fromkeys(groups):
            mask = np.array([sample_to_group[s] == g for s in samples])
            fig.add_trace(
                go.Box(y=values[mask], name=g, marker_color=palette[g],
                       boxpoints="all", jitter=0.5, pointpos=0,
                       showlegend=(idx == 0)),
                row=r, col=c,
            )
        fig.update_yaxes(title_text="log₂ expression", row=r, col=c)

    fig.update_layout(template="plotly_white", title=title,
                      height=300 * rows + 100, width=1100)
    return _save_fig(fig, out_dir, "top_boxplots")


# ── Plot 6: Engine-agreement scatter ─────────────────────────────────────────

def build_engine_scatter(
    intersected_df: pd.DataFrame,
    out_dir:        Path,
    title:          str = "Dual-engine agreement — Python vs R log₂FC",
) -> Dict[str, str]:
    """Scatter of Python log2FC vs R log2FC for the intersected biomarkers."""
    if intersected_df.empty:
        raise ValueError("Intersection is empty — no dual-engine hits to plot.")

    df = intersected_df.copy()
    fig = px.scatter(
        df, x="log2fc_python", y="log2fc_r",
        color="combined_adj_p",
        color_continuous_scale="Viridis_r",
        hover_data={"accession": True,
                    "log2fc_python": ":.3f", "log2fc_r": ":.3f",
                    "adj_p_python": ":.2e", "adj_p_r": ":.2e"},
        title=title,
        labels={"log2fc_python": "Python log₂FC",
                "log2fc_r":      "R / limma log₂FC",
                "combined_adj_p":"max(adj.p)"},
    )
    # y = x reference line
    lo = float(min(df["log2fc_python"].min(), df["log2fc_r"].min()))
    hi = float(max(df["log2fc_python"].max(), df["log2fc_r"].max()))
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi,
                  line=dict(dash="dot", color="gray"))

    # Label the top dual-engine hits
    for _, row in df.head(15).iterrows():
        fig.add_annotation(x=row["log2fc_python"], y=row["log2fc_r"],
                           text=str(row["accession"]),
                           showarrow=True, arrowhead=1, arrowsize=0.5,
                           font=dict(size=9), bgcolor="white", opacity=0.85)

    fig.update_layout(template="plotly_white", height=700, width=900)
    return _save_fig(fig, out_dir, "dual_engine_agreement")


# ── Orchestration ────────────────────────────────────────────────────────────

def build_visualisation_suite(
    *,
    python_results:   pd.DataFrame,
    expression_df:    pd.DataFrame,
    sample_to_group:  Dict[str, str],
    intersected_df:   Optional[pd.DataFrame] = None,
    out_dir:          str = "outputs/plots",
    top_n_for_heatmap: int = 30,
    adj_pval_cutoff:  float = 0.05,
    log2fc_cutoff:    float = 1.0,
) -> Dict[str, Dict[str, str]]:
    """Run every plot type that can be produced from the available inputs."""
    out = Path(out_dir)
    suite: Dict[str, Dict[str, str]] = {}

    # Volcano / MA require both log2_fold_change AND adj_p_value. Pooled
    # (fold-change-only) results have neither p_value nor adj_p_value, so we
    # silently skip those plots — PCA / heatmap / boxplots still build.
    has_adj_p = "adj_p_value" in python_results.columns

    # 1. Volcano
    if has_adj_p:
        try:
            suite["volcano"] = build_volcano(
                python_results, out, adj_pval_cutoff=adj_pval_cutoff,
                log2fc_cutoff=log2fc_cutoff,
            )
        except Exception as exc:
            logger.warning("Volcano failed: %s", exc)

    # 2. MA plot
    if has_adj_p:
        try:
            suite["ma_plot"] = build_ma_plot(
                python_results, out, adj_pval_cutoff=adj_pval_cutoff,
            )
        except Exception as exc:
            logger.warning("MA plot failed: %s", exc)

    # 3. PCA
    try:
        suite["pca"] = build_pca(expression_df, sample_to_group, out)
    except Exception as exc:
        logger.warning("PCA failed: %s", exc)

    # Top accessions (for heatmap + boxplots) — prefer the intersection,
    # fall back to top Python hits by adj.p (Welch/limma) or |log2FC|
    # (fold-change-only pooled designs).
    if intersected_df is not None and not intersected_df.empty:
        top_accs = intersected_df["accession"].head(top_n_for_heatmap).tolist()
    else:
        df = python_results.copy()
        if "accession" not in df.columns and "protein" in df.columns:
            df = df.rename(columns={"protein": "accession"})
        if "adj_p_value" in df.columns:
            df = df.sort_values("adj_p_value")
        elif "abs_log2_fold_change" in df.columns:
            df = df.sort_values("abs_log2_fold_change", ascending=False)
        elif "log2_fold_change" in df.columns:
            df = df.reindex(df["log2_fold_change"].abs().sort_values(ascending=False).index)
        top_accs = df.head(top_n_for_heatmap)["accession"].astype(str).tolist()

    # 4. Heatmap
    try:
        suite["heatmap"] = build_heatmap(
            expression_df, top_accs, sample_to_group, out,
        )
    except Exception as exc:
        logger.warning("Heatmap failed: %s", exc)

    # 5. Boxplots (top 12)
    try:
        suite["boxplots"] = build_top_boxplots(
            expression_df, top_accs[:12], sample_to_group, out,
        )
    except Exception as exc:
        logger.warning("Boxplots failed: %s", exc)

    # 6. Engine-agreement scatter
    if intersected_df is not None and not intersected_df.empty:
        try:
            suite["dual_engine"] = build_engine_scatter(intersected_df, out)
        except Exception as exc:
            logger.warning("Dual-engine scatter failed: %s", exc)

    logger.info(
        "Visualisation suite complete: %d plots written to %s",
        len(suite), out,
    )
    return suite
