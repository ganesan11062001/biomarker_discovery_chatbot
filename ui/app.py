"""
ui/app.py
Standalone Streamlit application for the Proteomics Biomarker Discovery Platform.

Run (API mode):
    streamlit run ui/app.py

Requires the FastAPI backend:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Proteomics Biomarker Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { padding: 0.5rem 0.75rem; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    .metric-label { font-size: 0.8rem; color: #666; }
    .group-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _api_create_session(disease_program: str, organism: str) -> str | None:
    try:
        r = requests.post(
            f"{API_BASE}/chat/session",
            params={"disease_program": disease_program, "organism": organism},
            timeout=10,
        )
        if r.status_code == 201:
            return r.json()["session_id"]
    except requests.exceptions.ConnectionError:
        pass
    return None


def _api_fetch_state(session_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/results/{session_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _api_send_message(session_id: str, message: str) -> dict | None:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "message": message,
        "disease_program": st.session_state.get("disease_program", "FA"),
        "organism": st.session_state.get("organism", "human"),
    }

    g1 = st.session_state.get("group1_samples") or []
    g2 = st.session_state.get("group2_samples") or []
    g1_label = st.session_state.get("group1_label", "Group1").strip() or "Group1"
    g2_label = st.session_state.get("group2_label", "Group2").strip() or "Group2"

    if g1:
        payload["group1_samples"] = g1
        payload["group1_label"]   = g1_label
    if g2:
        payload["group2_samples"] = g2
        payload["group2_label"]   = g2_label

    try:
        r = requests.post(f"{API_BASE}/chat/", json=payload, timeout=300)
        if r.status_code == 200:
            return r.json()
        st.session_state["api_error"] = f"API {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        st.session_state["api_error"] = (
            "Cannot reach the API server. "
            "Run: `uvicorn api.main:app --reload --port 8000`"
        )
    except requests.exceptions.Timeout:
        st.session_state["api_error"] = "Request timed out — analysis still running."
    return None


def _api_upload_file(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    session_id: str,
    disease_program: str,
    organism: str,
) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/upload/",
            files={"file": (filename, file_bytes, file_type)},
            data={
                "session_id": session_id,
                "disease_program": disease_program,
                "organism": organism,
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API server. Is it running on `localhost:8000`?")
        return None

    if r.status_code in (200, 201):
        return r.json()

    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    st.error(f"Upload failed ({r.status_code}): {detail}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Session init
# ══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    defaults: dict[str, Any] = {
        "session_id":      None,
        "messages":        [],
        "analysis_state":  {},
        "upload_result":   None,
        "disease_program": "FA",
        "organism":        "human",
        # group assignment
        "group1_samples":  [],
        "group2_samples":  [],
        "group1_label":    "Disease",
        "group2_label":    "Control",
        "api_error":       None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧬 Biomarker Discovery")
        st.caption("Proteomics · Multi-Agent AI Platform")
        st.divider()

        # ── Session config ────────────────────────────────────────────────────
        st.subheader("Session")
        disease = st.selectbox(
            "Disease program",
            ["FA", "DMD", "SMA", "Other"],
            index=["FA", "DMD", "SMA", "Other"].index(
                st.session_state["disease_program"]
                if st.session_state["disease_program"] in ["FA", "DMD", "SMA", "Other"]
                else "Other"
            ),
        )
        organism = st.selectbox(
            "Organism",
            ["human", "mouse", "rat"],
            index=["human", "mouse", "rat"].index(st.session_state["organism"]),
        )

        config_changed = (
            disease   != st.session_state["disease_program"]
            or organism != st.session_state["organism"]
        )
        if config_changed:
            st.session_state["disease_program"] = disease
            st.session_state["organism"]        = organism
            st.session_state["session_id"]      = None

        # Ensure a session exists
        if st.session_state["session_id"] is None:
            sid = _api_create_session(disease, organism)
            if sid:
                st.session_state["session_id"] = sid
            else:
                st.error("API server not reachable.")
                st.code("uvicorn api.main:app --reload --port 8000")
                return

        st.caption(f"Session: `{st.session_state['session_id'][:8]}…`")
        st.divider()

        # ── File upload ───────────────────────────────────────────────────────
        st.subheader("Upload Data")
        uploaded = st.file_uploader(
            "Proteomics matrix (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
            help=(
                "Rows = proteins, Columns = samples.\n"
                "Supported: Olink NPX, LFQ, TMT, generic intensity matrix."
            ),
            key="sidebar_uploader",
        )

        if uploaded is not None:
            last = st.session_state.get("_last_upload_name")
            if last != uploaded.name:
                with st.spinner(f"Uploading {uploaded.name}…"):
                    result = _api_upload_file(
                        uploaded.getvalue(),
                        uploaded.name,
                        uploaded.type or "application/octet-stream",
                        st.session_state["session_id"],
                        disease,
                        organism,
                    )
                if result:
                    st.session_state["_last_upload_name"] = uploaded.name
                    st.session_state["upload_result"]     = result
                    st.session_state["analysis_state"]    = _api_fetch_state(
                        st.session_state["session_id"]
                    )
                    # Reset group assignments when new file is loaded
                    st.session_state["group1_samples"] = []
                    st.session_state["group2_samples"] = []
                    st.rerun()

        ur = st.session_state.get("upload_result")
        if ur:
            st.success(f"Loaded: **{ur.get('filename', '')}**")
            c1, c2 = st.columns(2)
            c1.metric("Proteins", ur.get("n_proteins", "–"))
            c2.metric("Samples",  ur.get("n_samples",  "–"))
            st.caption(
                f"Type: **{ur.get('data_type', '?')}** · "
                f"Format: **{(ur.get('data_format') or '').upper()}**"
            )

        st.divider()

        # ── Group assignment ──────────────────────────────────────────────────
        sample_cols: list[str] = (ur or {}).get("sample_columns") or []

        if sample_cols:
            st.subheader("Group Assignment")
            st.caption("Assign sample columns to each comparison group.")

            col_label1, col_label2 = st.columns(2)
            with col_label1:
                st.session_state["group1_label"] = st.text_input(
                    "Group 1 label",
                    value=st.session_state.get("group1_label", "Disease"),
                    key="g1_label_input",
                )
            with col_label2:
                st.session_state["group2_label"] = st.text_input(
                    "Group 2 label",
                    value=st.session_state.get("group2_label", "Control"),
                    key="g2_label_input",
                )

            # Determine already-assigned columns for exclusion
            g2_assigned = set(st.session_state.get("group2_samples") or [])
            g1_assigned = set(st.session_state.get("group1_samples") or [])

            g1_options = [c for c in sample_cols if c not in g2_assigned]
            g2_options = [c for c in sample_cols if c not in g1_assigned]

            g1_default = [c for c in (st.session_state.get("group1_samples") or [])
                          if c in g1_options]
            g2_default = [c for c in (st.session_state.get("group2_samples") or [])
                          if c in g2_options]

            st.session_state["group1_samples"] = st.multiselect(
                f"Group 1 — {st.session_state['group1_label']}",
                options=g1_options,
                default=g1_default,
                key="g1_multiselect",
            )
            st.session_state["group2_samples"] = st.multiselect(
                f"Group 2 — {st.session_state['group2_label']}",
                options=g2_options,
                default=g2_default,
                key="g2_multiselect",
            )

            g1_n = len(st.session_state["group1_samples"])
            g2_n = len(st.session_state["group2_samples"])
            if g1_n and g2_n:
                st.success(f"{g1_n} vs {g2_n} samples assigned")

                if st.button("▶ Run Analysis", type="primary", use_container_width=True):
                    g1_lbl = st.session_state["group1_label"]
                    g2_lbl = st.session_state["group2_label"]
                    msg = (
                        f"Run differential expression analysis comparing "
                        f"{g1_lbl} ({g1_n} samples) vs {g2_lbl} ({g2_n} samples)"
                    )
                    st.session_state["messages"].append({"role": "user", "content": msg})
                    with st.spinner("Running analysis…"):
                        resp = _api_send_message(st.session_state["session_id"], msg)
                    if resp:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": resp["response"]}
                        )
                        st.session_state["analysis_state"] = _api_fetch_state(
                            st.session_state["session_id"]
                        )
                    st.rerun()
            elif g1_n or g2_n:
                st.warning("Assign samples to both groups to run analysis.")

        elif ur:
            st.info("No numeric sample columns detected. Check your file format.")

        st.divider()

        # ── Pipeline status ───────────────────────────────────────────────────
        st.subheader("Pipeline Status")
        astate = st.session_state.get("analysis_state") or {}
        steps = [
            ("Data loaded",        bool(astate.get("data_type"))),
            ("QC passed",          bool(astate.get("qc_passed"))),
            ("Analysis complete",  bool(astate.get("n_significant") is not None)),
            ("Excel report ready", bool(astate.get("excel_path"))),
        ]
        for label, done in steps:
            icon = "✅" if done else "⭕"
            st.markdown(f"{icon} {label}")

        st.divider()

        # Excel download
        astate = st.session_state.get("analysis_state") or {}
        if astate.get("excel_path"):
            sid = st.session_state["session_id"]
            try:
                r = requests.get(
                    f"{API_BASE}/results/{sid}/excel",
                    timeout=30,
                )
                if r.status_code == 200:
                    st.download_button(
                        label="⬇ Download Excel Report",
                        data=r.content,
                        file_name=f"biomarkers_{sid[:8]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="primary",
                    )
            except Exception:
                pass

        # New session
        if st.button("🔄 New Session", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main content area
# ══════════════════════════════════════════════════════════════════════════════

def _render_main() -> None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.info("Initialising session — please wait…")
        return

    if st.session_state.get("api_error"):
        st.error(st.session_state.pop("api_error"))

    chat_col, results_col = st.columns([5, 6], gap="large")

    # ── Chat column ───────────────────────────────────────────────────────────
    with chat_col:
        st.subheader("Chat")

        # Inline upload prompt if no data yet
        astate = st.session_state.get("analysis_state") or {}
        if not astate.get("data_type") and not st.session_state.get("upload_result"):
            st.info("Upload your proteomics data in the sidebar to get started.", icon="📂")

        # Message history
        with st.container(height=420, border=False):
            messages = st.session_state.get("messages") or []
            if not messages:
                st.markdown("#### Welcome to Biomarker Discovery")
                st.markdown(
                    "Upload a proteomics matrix, assign sample groups in the sidebar, "
                    "then click **Run Analysis** — or ask me anything below."
                )
            else:
                for m in messages:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

        # Suggested quick-action buttons
        astate = st.session_state.get("analysis_state") or {}
        data_loaded   = bool(astate.get("data_type"))
        analysis_done = bool(astate.get("excel_path"))

        if data_loaded and not analysis_done:
            g1 = st.session_state.get("group1_samples") or []
            g2 = st.session_state.get("group2_samples") or []
            if not g1 or not g2:
                st.caption("Tip: assign sample groups in the sidebar, then click Run Analysis.")
            else:
                cols = st.columns(2)
                if cols[0].button("What proteins are in my data?", use_container_width=True):
                    st.session_state["_quick_prompt"] = "What proteins are in my data?"
                if cols[1].button("Describe the dataset", use_container_width=True):
                    st.session_state["_quick_prompt"] = "Describe the uploaded dataset"

        if analysis_done:
            cols = st.columns(3)
            if cols[0].button("Summarise results", use_container_width=True):
                st.session_state["_quick_prompt"] = "Summarise the biomarker discovery results"
            if cols[1].button("Top 10 biomarkers", use_container_width=True):
                st.session_state["_quick_prompt"] = "Show me the top 10 biomarkers"
            if cols[2].button("Run pathway enrichment", use_container_width=True):
                st.session_state["_quick_prompt"] = "Run pathway enrichment analysis"

        # Chat input
        user_input = st.chat_input("Ask anything about your proteomics data…")
        quick = st.session_state.pop("_quick_prompt", None)
        user_input = user_input or quick

        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                resp = _api_send_message(session_id, user_input)
            if resp:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": resp["response"]}
                )
                st.session_state["analysis_state"] = _api_fetch_state(session_id)
            st.rerun()

    # ── Results column ────────────────────────────────────────────────────────
    with results_col:
        st.subheader("Results")
        _render_results(st.session_state.get("analysis_state") or {}, session_id)


# ══════════════════════════════════════════════════════════════════════════════
# Results panel
# ══════════════════════════════════════════════════════════════════════════════

def _render_results(state: dict[str, Any], session_id: str) -> None:
    if not state or not state.get("status") or state.get("status") == "ready":
        st.info("No results yet. Upload data and run the analysis.")
        return

    tab_overview, tab_biomarkers, tab_qc = st.tabs(
        ["📊 Overview", "🏆 Top Biomarkers", "🔬 QC Summary"]
    )

    with tab_overview:
        _render_overview(state, session_id)

    with tab_biomarkers:
        _render_biomarkers(state)

    with tab_qc:
        _render_qc(state)


def _render_overview(state: dict[str, Any], session_id: str) -> None:
    st.markdown("### Analysis Overview")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proteins", state.get("n_proteins", "–"))
    c2.metric("Samples",  state.get("n_samples",  "–"))
    c3.metric("Significant", state.get("n_significant", "–"))
    c4.metric("Mode", (state.get("analysis_mode") or "–").title())

    # Contrast label
    g1 = state.get("group1_label")
    g2 = state.get("group2_label")
    if g1 and g2:
        st.caption(f"Comparison: **{g1}** vs **{g2}**")

    st.divider()

    # LLM summary
    summary = state.get("analysis_summary")
    if summary:
        st.markdown("#### AI Summary")
        st.markdown(summary)
        st.divider()

    # Excel download
    if state.get("excel_path"):
        try:
            r = requests.get(f"{API_BASE}/results/{session_id}/excel", timeout=30)
            if r.status_code == 200:
                st.download_button(
                    label="⬇ Download Full Excel Report",
                    data=r.content,
                    file_name=f"biomarkers_{session_id[:8]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )
        except Exception:
            st.caption("Excel file available — use the sidebar download button.")

    # Quick stats
    if state.get("qc_summary"):
        with st.expander("QC snapshot"):
            qc = state["qc_summary"]
            cols = st.columns(3)
            cols[0].metric("Proteins in", qc.get("proteins_input",    "–"))
            cols[1].metric("After QC",    qc.get("proteins_after_qc", "–"))
            cols[2].metric("Removed",     qc.get("proteins_removed",  "–"))


def _render_biomarkers(state: dict[str, Any]) -> None:
    top = state.get("top_biomarkers")
    analysis_mode = state.get("analysis_mode", "supervised")

    if not top:
        if state.get("status") in ("data_loaded", "routed", "ready"):
            st.info("Run the analysis to see biomarkers.")
        else:
            st.info("No biomarkers found yet.")
        return

    import pandas as pd

    df = pd.DataFrame(top)

    # Rename columns for display
    rename = {
        "protein":          "Protein",
        "rank":             "Rank",
        "log2_fold_change": "log2 FC",
        "p_value":          "p-value",
        "adj_p_value":      "adj. p-value",
        "significance":     "Significance",
        "cv_percent":       "CV %",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Display columns present in df
    desired_cols = ["Rank", "Protein", "log2 FC", "p-value", "adj. p-value",
                    "Significance", "CV %"]
    show_cols = [c for c in desired_cols if c in df.columns]
    df_show = df[show_cols].copy()

    # Format numerics
    for col in ("p-value", "adj. p-value"):
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f"{x:.3e}" if isinstance(x, float) else x
            )
    for col in ("log2 FC", "CV %"):
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, float) else x
            )

    if "Rank" in df_show.columns:
        df_show = df_show.set_index("Rank")

    # Color rows by significance
    def _row_color(row):
        sig = str(row.get("Significance", "")).lower()
        if "highly" in sig:
            return ["background-color: #C6EFCE"] * len(row)
        if "significant" in sig and "highly" not in sig:
            return ["background-color: #E2EFDA"] * len(row)
        if "trend" in sig:
            return ["background-color: #FFEB9C"] * len(row)
        return [""] * len(row)

    styled = df_show.style
    if "Significance" in df_show.columns:
        styled = styled.apply(_row_color, axis=1)

    st.dataframe(styled, use_container_width=True, height=min(600, 38 * len(df_show) + 40))

    # Simple inline volcano plot if we have the needed columns
    if "log2 FC" in df.columns and "adj. p-value" in df.columns:
        _render_inline_volcano(top, state)


def _render_inline_volcano(top_biomarkers: list, state: dict[str, Any]) -> None:
    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(top_biomarkers)
        if "log2_fold_change" not in df.columns or "adj_p_value" not in df.columns:
            return

        df = df.dropna(subset=["log2_fold_change", "adj_p_value"])
        df["adj_p_clipped"] = df["adj_p_value"].clip(lower=1e-300)
        df["-log10p"] = -np.log10(df["adj_p_clipped"])

        def _sig_class(row):
            if row.get("adj_p_value", 1) < 0.05 and abs(row.get("log2_fold_change", 0)) >= 1.0:
                return "Significant"
            return "NS"

        df["class"] = df.apply(_sig_class, axis=1)

        g1 = state.get("group1_label", "Group1")
        g2 = state.get("group2_label", "Group2")

        fig = px.scatter(
            df,
            x="log2_fold_change",
            y="-log10p",
            color="class",
            color_discrete_map={"Significant": "#C0392B", "NS": "#AAAAAA"},
            hover_name="protein",
            hover_data={"log2_fold_change": ":.3f", "adj_p_value": ":.3e",
                        "class": False, "-log10p": False, "adj_p_clipped": False},
            labels={"log2_fold_change": "log₂FC", "-log10p": "−log₁₀(adj. p-value)"},
            title=f"Volcano: {g1} vs {g2}",
        )
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="grey", line_width=1)
        fig.add_vline(x=1.0,  line_dash="dash", line_color="grey", line_width=1)
        fig.add_vline(x=-1.0, line_dash="dash", line_color="grey", line_width=1)
        fig.update_layout(legend_title_text="", height=380, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Volcano plot unavailable: {e}")


def _render_qc(state: dict[str, Any]) -> None:
    st.markdown("### QC Summary")

    qc_passed = state.get("qc_passed")
    if qc_passed is None:
        st.info("QC has not been run yet.")
        return

    if qc_passed:
        st.success("Data passed quality control")
    else:
        st.warning("Quality issues detected — results may be affected")

    qc = state.get("qc_summary") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Proteins in",    qc.get("proteins_input",    state.get("n_proteins", "–")))
    c2.metric("After QC",       qc.get("proteins_after_qc", "–"))
    c3.metric("Removed",        qc.get("proteins_removed",  "–"))

    c4, c5, c6 = st.columns(3)
    c4.metric("Samples in",    qc.get("samples_input",    state.get("n_samples", "–")))
    c5.metric("After QC",      qc.get("samples_after_qc", "–"))

    log2 = qc.get("log2_transformed")
    if log2 is not None:
        c6.metric("log2 transform", "Yes" if log2 else "No")

    missing_thr = qc.get("missing_threshold")
    if missing_thr is not None:
        st.caption(f"Missing value threshold: **{missing_thr * 100:.0f}%**")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session()
    _render_sidebar()
    _render_main()


main()
