"""
ui/components/results_panel.py
Tabbed results panel: QC · DEA · Pathways · Plots
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_results_panel(state: dict[str, Any], api_base: str) -> None:
    """
    Render the full results panel.
    `state` is the JSON response from GET /results/{session_id}.
    """
    if not state or state.get("status") in (None, "ready"):
        st.info("No analysis results yet. Upload your data and start a conversation.")
        return

    tab_qc, tab_dea, tab_pathways, tab_plots = st.tabs(
        ["🔬 QC", "📊 Differential Expression", "🧭 Pathways", "🖼 Plots & Reports"]
    )

    with tab_qc:
        _render_qc_tab(state)

    with tab_dea:
        _render_dea_tab(state)

    with tab_pathways:
        _render_pathways_tab(state)

    with tab_plots:
        _render_plots_tab(state, api_base)


# ── QC tab ────────────────────────────────────────────────────────────────────

def _render_qc_tab(state: dict[str, Any]) -> None:
    st.subheader("Quality Control Summary")

    qc_passed = state.get("qc_passed")
    if qc_passed is None:
        st.info("QC has not been run yet.")
        return

    if qc_passed:
        st.success("✅ Data passed quality control")
    else:
        st.warning("⚠️ Data quality issues detected — review before proceeding")

    c1, c2 = st.columns(2)
    c1.metric("Proteins retained", state.get("n_proteins", "–"))
    c2.metric("Samples retained", state.get("n_samples", "–"))

    # Load QC report JSON if available
    qc_path = state.get("qc_report_path")
    if qc_path and Path(qc_path).exists():
        import json
        with open(qc_path) as f:
            qc_report = json.load(f)
        with st.expander("QC Report Details"):
            st.json(qc_report)


# ── DEA tab ───────────────────────────────────────────────────────────────────

def _render_dea_tab(state: dict[str, Any]) -> None:
    st.subheader("Differential Expression Analysis")

    dea_path = state.get("dea_result_path")
    top_proteins = state.get("top_proteins")

    if not dea_path:
        st.info("DEA has not been run yet. Ask: *'Run differential expression analysis'*")
        return

    contrast = state.get("contrast_groups", [])
    contrast_label = " vs ".join(contrast) if contrast else "Group comparison"
    st.caption(f"Contrast: **{contrast_label}**  ·  Significant proteins (adj.P<0.05, |logFC|≥0.5): "
               f"**{state.get('n_significant', 0)}**")

    # Top proteins table
    if top_proteins:
        df = pd.DataFrame(top_proteins)
        df["adj_pval"] = df["adj_pval"].apply(lambda x: f"{x:.3e}")
        df["logFC"] = df["logFC"].round(3)
        df.index = range(1, len(df) + 1)
        df.columns = ["Protein", "logFC", "adj. p-value", "Direction"]

        st.dataframe(
            df.style
              .applymap(lambda v: "color: #c0392b" if v == "up" else "color: #2980b9",
                        subset=["Direction"])
              .format({"logFC": "{:.3f}"}),
            use_container_width=True,
            height=min(400, 35 * len(df) + 40),
        )

    # Full results table (from CSV)
    if Path(dea_path).exists():
        with st.expander("Full DEA Results Table"):
            full_df = pd.read_csv(dea_path)
            st.dataframe(full_df, use_container_width=True)
            csv = full_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download DEA Results (CSV)",
                data=csv,
                file_name="dea_results.csv",
                mime="text/csv",
            )

    # Interactive volcano plot (Plotly, from DEA CSV)
    if Path(dea_path).exists():
        _render_interactive_volcano(dea_path, contrast)


def _render_interactive_volcano(dea_path: str, contrast: list) -> None:
    """Build an interactive Plotly volcano plot from the DEA CSV."""
    try:
        df = pd.read_csv(dea_path)
        if "logFC" not in df.columns:
            return

        # Normalise column names
        if "adj_pval" not in df.columns and "adj.P.Val" in df.columns:
            df = df.rename(columns={"adj.P.Val": "adj_pval"})

        import numpy as np
        df["-log10p"] = -np.log10(df["adj_pval"].clip(lower=1e-300))
        df["color"] = "NS"
        df.loc[(df["adj_pval"] < 0.05) & (df["logFC"] >  0.5), "color"] = "Up"
        df.loc[(df["adj_pval"] < 0.05) & (df["logFC"] < -0.5), "color"] = "Down"

        fig = px.scatter(
            df,
            x="logFC",
            y="-log10p",
            color="color",
            color_discrete_map={"Up": "#c0392b", "Down": "#2980b9", "NS": "#aaaaaa"},
            hover_name=df.columns[0],  # protein column
            hover_data={"logFC": ":.3f", "adj_pval": ":.3e", "color": False, "-log10p": False},
            labels={"logFC": "log₂FC", "-log10p": "−log₁₀(adj. p-value)"},
            title=" vs ".join(contrast) if contrast else "Volcano Plot",
        )
        import numpy as _np
        fig.add_hline(y=-_np.log10(0.05), line_dash="dash", line_color="grey", line_width=1)
        fig.add_vline(x=0.5,  line_dash="dash", line_color="grey", line_width=1)
        fig.add_vline(x=-0.5, line_dash="dash", line_color="grey", line_width=1)
        fig.update_layout(legend_title_text="", height=450)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Interactive volcano unavailable: {e}")


# ── Pathways tab ──────────────────────────────────────────────────────────────

def _render_pathways_tab(state: dict[str, Any]) -> None:
    st.subheader("Pathway Enrichment Analysis")

    enrichment_path = state.get("enrichment_result_path")
    pathways = state.get("pathways")

    if not enrichment_path:
        st.info("Pathway enrichment has not been run yet. Ask: *'Run pathway enrichment analysis'*")
        return

    if pathways:
        df = pd.DataFrame(pathways)
        df["p_adjust"] = df["p_adjust"].apply(lambda x: f"{x:.3e}")
        df.index = range(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)

    if enrichment_path and Path(enrichment_path).exists():
        full_df = pd.read_csv(enrichment_path)

        # Interactive dot plot
        try:
            import numpy as np
            kegg = full_df[full_df.get("source", pd.Series()).eq("KEGG")].head(20) if "source" in full_df.columns else full_df.head(20)
            if len(kegg) > 0:
                kegg = kegg.copy()
                kegg["-log10p"] = -np.log10(kegg["p_adjust"].clip(lower=1e-300))
                fig = px.scatter(
                    kegg,
                    x="-log10p",
                    y="pathway",
                    size="gene_count" if "gene_count" in kegg.columns else None,
                    color="-log10p",
                    color_continuous_scale="Blues",
                    labels={"-log10p": "−log₁₀(adj. p-value)", "pathway": ""},
                    title="Top KEGG Pathways",
                )
                fig.update_layout(height=max(300, len(kegg) * 30), showlegend=False)
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Dot plot unavailable: {e}")

        with st.expander("Full Enrichment Table"):
            st.dataframe(full_df, use_container_width=True)
            csv = full_df.to_csv(index=False).encode()
            st.download_button("⬇ Download Enrichment Results (CSV)", csv,
                               "enrichment_results.csv", "text/csv")


# ── Plots tab ─────────────────────────────────────────────────────────────────

def _render_plots_tab(state: dict[str, Any], api_base: str) -> None:
    st.subheader("Generated Plots & Reports")

    plot_paths: list[str] = state.get("plot_paths") or []
    report_path: str | None = state.get("report_path")

    if not plot_paths and not report_path:
        st.info("No plots generated yet. Ask: *'Generate a volcano plot'* or *'Create a report'*")
        return

    # Show each plot
    for path in plot_paths:
        p = Path(path)
        if p.exists():
            if p.suffix.lower() == ".png":
                st.image(str(p), caption=p.name, use_container_width=True)
            elif p.suffix.lower() == ".html":
                with open(p) as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
        else:
            st.caption(f"Plot not found: {p.name}")

    # Download biomarker ranking table
    if report_path and Path(report_path).exists():
        df = pd.read_csv(report_path)
        st.subheader("Biomarker Ranking Table")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Biomarker Ranking (CSV)", csv,
                           "biomarker_ranking.csv", "text/csv")
