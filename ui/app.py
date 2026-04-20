"""
ui/app.py
Streamlit interface — in-chat file attachment (ChatGPT / Claude style).

Files are attached directly in the chat area.  Sending a file immediately
triggers ingestion and surfaces the result as a chat message.  The sidebar
holds only session config, pipeline status, and the download button.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Biomarker Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── General ── */
    .stChatMessage { padding: 0.4rem 0.75rem; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

    /* ── Compact file-uploader (chat attachment area) ── */
    div[data-testid="stFileUploader"] > label {
        font-weight: 500;
        font-size: 0.85rem;
        color: #555;
    }
    div[data-testid="stFileUploaderDropzone"] {
        padding: 8px 14px !important;
        min-height: 0 !important;
        border-radius: 12px !important;
        border: 1.5px dashed #bbb !important;
        background: #fafafa !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 0.8rem !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
        display: none !important;
    }

    /* ── Attachment preview chip ── */
    .attach-preview {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #EBF5FB;
        border: 1px solid #AED6F1;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 0.875rem;
    }
    .attach-preview .fn { font-weight: 600; color: #1a1a2e; }
    .attach-preview .meta { color: #555; font-size: 0.8rem; }

    /* ── Input bar container ── */
    .input-bar {
        border: 1.5px solid #ddd;
        border-radius: 14px;
        padding: 6px 10px;
        background: #fff;
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
    g1       = st.session_state.get("group1_samples") or []
    g2       = st.session_state.get("group2_samples") or []
    g1_label = (st.session_state.get("group1_label") or "Group1").strip() or "Group1"
    g2_label = (st.session_state.get("group2_label") or "Group2").strip() or "Group2"
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
        st.session_state["api_error"] = "Request timed out — analysis may still be running."
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
            data={"session_id": session_id,
                  "disease_program": disease_program,
                  "organism": organism},
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
        "session_id":        None,
        "messages":          [],
        "analysis_state":    {},
        "upload_result":     None,
        "disease_program":   "FA",
        "organism":          "human",
        "group1_samples":    [],
        "group2_samples":    [],
        "group1_label":      "Disease",
        "group2_label":      "Control",
        "api_error":         None,
        # file-attach state
        "_attach_ver":       0,     # bumped to reset the file_uploader widget
        "_last_attach_name": None,  # guards against duplicate uploads on rerun
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ingestion_chat_message(result: dict) -> str:
    """Build the assistant chat bubble text from an upload response."""
    n_p   = result.get("n_proteins", "?")
    n_s   = result.get("n_samples",  "?")
    dtype = result.get("data_type",  "unknown")
    is_pooled = result.get("is_pooled_design", False)
    label_map = result.get("label_map") or {}

    lines = [
        "### Data loaded ✅\n",
        f"| | |",
        f"|---|---|",
        f"| Proteins | **{n_p}** |",
        f"| Samples / groups | **{n_s}** |",
        f"| Data type | **{dtype}** |",
    ]

    if is_pooled and label_map:
        groups = ", ".join(f"**{k}** → {v}" for k, v in label_map.items())
        lines += [
            "",
            "---",
            "**Pooled design detected.** "
            "Each group is a single pooled sample — "
            "fold-change analysis runs across all contrasts automatically.",
            f"Groups detected: {groups}",
            "",
            "Click **▶ Run Fold-Change Analysis** in the sidebar, "
            "or just type *'run analysis'* below.",
        ]
    elif is_pooled:
        lines += ["", "**Pooled design detected.** Click **▶ Run Fold-Change Analysis**."]
    else:
        lines += [
            "",
            "Assign samples to **Group 1** and **Group 2** in the sidebar, "
            "then click **▶ Run Analysis**.",
        ]

    return "\n".join(lines)


def _handle_file_attach(attached, session_id: str) -> None:
    """Upload the attached file, update state, and add chat messages."""
    last = st.session_state.get("_last_attach_name")
    if last == attached.name:
        return  # already processed this file

    st.session_state["_last_attach_name"] = attached.name

    # User bubble — show the attachment
    st.session_state["messages"].append({
        "role":    "user",
        "content": f"📎 **{attached.name}**  ({len(attached.getvalue()) // 1024} KB)",
    })

    with st.spinner(f"Uploading and processing **{attached.name}** …"):
        result = _api_upload_file(
            attached.getvalue(),
            attached.name,
            attached.type or "application/octet-stream",
            session_id,
            st.session_state.get("disease_program", "FA"),
            st.session_state.get("organism", "human"),
        )

    if result:
        # Sync session_id — upload may have created a new session (e.g. after server restart)
        returned_sid = result.get("session_id") or session_id
        st.session_state["session_id"]     = returned_sid
        st.session_state["upload_result"]  = result
        st.session_state["group1_samples"] = []
        st.session_state["group2_samples"] = []
        st.session_state["analysis_state"] = _api_fetch_state(returned_sid)

        # Assistant bubble — ingestion summary
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": _ingestion_chat_message(result),
        })

        # Reset the file_uploader widget so it shows empty again
        st.session_state["_attach_ver"] += 1

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar  (settings + status + download — NO file upload here)
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

        if (disease != st.session_state["disease_program"]
                or organism != st.session_state["organism"]):
            st.session_state["disease_program"] = disease
            st.session_state["organism"]        = organism
            st.session_state["session_id"]      = None

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

        # ── Dataset info (after upload) ───────────────────────────────────────
        ur = st.session_state.get("upload_result")
        if ur:
            st.subheader("Dataset")
            c1, c2 = st.columns(2)
            c1.metric("Proteins", ur.get("n_proteins", "–"))
            c2.metric("Samples",  ur.get("n_samples",  "–"))
            st.caption(
                f"Type: **{ur.get('data_type', '?')}** · "
                f"Format: **{(ur.get('data_format') or '').upper()}**"
            )
            st.divider()

            sample_cols = ur.get("sample_columns") or []
            is_pooled   = bool(ur.get("is_pooled_design", False))
            label_map   = ur.get("label_map") or {}

            if is_pooled:
                # ── Pooled: auto-route, just show a run button ────────────────
                st.subheader("Analysis")
                st.info("**Pooled design** — fold-change analysis runs automatically.")
                if label_map:
                    with st.expander("Groups", expanded=True):
                        for code, name in label_map.items():
                            st.markdown(f"- **{code}** → {name}")

                if st.button(
                    "▶ Run Fold-Change Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    _trigger_analysis(
                        st.session_state["session_id"],
                        "Run pooled fold-change analysis on the uploaded dataset",
                    )

            elif sample_cols:
                # ── Standard: manual group assignment ─────────────────────────
                st.subheader("Group Assignment")
                st.caption("Assign sample columns to comparison groups.")

                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    st.session_state["group1_label"] = st.text_input(
                        "Group 1 label",
                        value=st.session_state.get("group1_label", "Disease"),
                        key="g1_label_input",
                    )
                with col_l2:
                    st.session_state["group2_label"] = st.text_input(
                        "Group 2 label",
                        value=st.session_state.get("group2_label", "Control"),
                        key="g2_label_input",
                    )

                g2_assigned = set(st.session_state.get("group2_samples") or [])
                g1_assigned = set(st.session_state.get("group1_samples") or [])

                st.session_state["group1_samples"] = st.multiselect(
                    f"Group 1 — {st.session_state['group1_label']}",
                    options=[c for c in sample_cols if c not in g2_assigned],
                    default=[c for c in (st.session_state.get("group1_samples") or [])
                             if c not in g2_assigned],
                    key="g1_multiselect",
                )
                st.session_state["group2_samples"] = st.multiselect(
                    f"Group 2 — {st.session_state['group2_label']}",
                    options=[c for c in sample_cols if c not in g1_assigned],
                    default=[c for c in (st.session_state.get("group2_samples") or [])
                             if c not in g1_assigned],
                    key="g2_multiselect",
                )

                g1_n = len(st.session_state["group1_samples"])
                g2_n = len(st.session_state["group2_samples"])
                if g1_n and g2_n:
                    st.success(f"{g1_n} vs {g2_n} samples assigned")
                    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
                        g1_lbl = st.session_state["group1_label"]
                        g2_lbl = st.session_state["group2_label"]
                        _trigger_analysis(
                            st.session_state["session_id"],
                            f"Run differential expression analysis comparing "
                            f"{g1_lbl} ({g1_n} samples) vs {g2_lbl} ({g2_n} samples)",
                        )
                elif g1_n or g2_n:
                    st.warning("Assign samples to both groups to run analysis.")

        st.divider()

        # ── Pipeline status ───────────────────────────────────────────────────
        st.subheader("Pipeline Status")
        astate = st.session_state.get("analysis_state") or {}
        for label, done in [
            ("Data loaded",        bool(astate.get("data_type"))),
            ("QC passed",          bool(astate.get("qc_passed"))),
            ("Analysis complete",  astate.get("n_significant") is not None),
            ("Excel report ready", bool(astate.get("excel_path"))),
        ]:
            st.markdown(f"{'✅' if done else '⭕'} {label}")

        st.divider()

        # ── Excel download ────────────────────────────────────────────────────
        if astate.get("excel_path"):
            sid = st.session_state["session_id"]
            try:
                r = requests.get(f"{API_BASE}/results/{sid}/excel", timeout=30)
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

        if st.button("🔄 New Session", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Analysis trigger (shared by sidebar buttons and chat)
# ══════════════════════════════════════════════════════════════════════════════

def _trigger_analysis(session_id: str, message: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": message})
    with st.spinner("Running analysis…"):
        resp = _api_send_message(session_id, message)
    if resp:
        st.session_state["messages"].append(
            {"role": "assistant", "content": resp["response"]}
        )
        st.session_state["analysis_state"] = _api_fetch_state(session_id)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main content
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

        # Message history
        with st.container(height=380, border=False):
            messages = st.session_state.get("messages") or []
            if not messages:
                st.markdown("#### Welcome to Biomarker Discovery 🧬")
                st.markdown(
                    "Attach your proteomics file using the **📎 clip below**, "
                    "then ask a question or click **Run Analysis** in the sidebar."
                )
            else:
                for m in messages:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

        # ── Quick-action buttons ──────────────────────────────────────────────
        astate        = st.session_state.get("analysis_state") or {}
        data_loaded   = bool(astate.get("data_type"))
        analysis_done = bool(astate.get("excel_path"))

        if analysis_done:
            q1, q2, q3 = st.columns(3)
            if q1.button("Summarise results",    use_container_width=True):
                st.session_state["_quick"] = "Summarise the biomarker discovery results"
            if q2.button("Top 10 biomarkers",    use_container_width=True):
                st.session_state["_quick"] = "Show me the top 10 biomarkers"
            if q3.button("Pathway enrichment",   use_container_width=True):
                st.session_state["_quick"] = "Run pathway enrichment analysis"

        elif data_loaded:
            ur        = st.session_state.get("upload_result") or {}
            is_pooled = bool(ur.get("is_pooled_design"))
            g1        = st.session_state.get("group1_samples") or []
            g2        = st.session_state.get("group2_samples") or []
            if not is_pooled and (not g1 or not g2):
                st.caption("💡 Assign sample groups in the sidebar, then click **▶ Run Analysis**.")
            else:
                qa, qb = st.columns(2)
                if qa.button("What proteins are in my data?", use_container_width=True):
                    st.session_state["_quick"] = "What proteins are in my data?"
                if qb.button("Describe the dataset",          use_container_width=True):
                    st.session_state["_quick"] = "Describe the uploaded dataset"

        # ── File attachment area (ChatGPT / Claude style) ─────────────────────
        st.markdown("---")
        st.markdown(
            "📎 **Attach file** — drop your proteomics CSV or Excel here, "
            "or browse to select it.",
            help="Supported: CSV, XLSX, XLS"
        )

        attach_ver    = st.session_state.get("_attach_ver", 0)
        attached_file = st.file_uploader(
            "Attach proteomics file",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
            key=f"chat_attach_{attach_ver}",
            label_visibility="collapsed",
        )

        if attached_file is not None:
            _handle_file_attach(attached_file, session_id)

        # ── Chat input ────────────────────────────────────────────────────────
        user_input = st.chat_input("Ask anything about your proteomics data…")
        quick      = st.session_state.pop("_quick", None)
        user_input = user_input or quick

        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                resp = _api_send_message(session_id, user_input)
            if resp:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": resp["response"]}
                )
                # Sync session_id — server may have issued a replacement on expiry
                if resp.get("session_id") and resp["session_id"] != session_id:
                    st.session_state["session_id"] = resp["session_id"]
                    st.session_state["upload_result"]  = {}
                    st.session_state["analysis_state"] = {}
                else:
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
        st.info("No results yet. Attach your data file and run the analysis.")
        return

    has_plots    = bool(state.get("plot_paths"))
    has_pathways = bool(state.get("pathways"))

    tabs = ["📊 Overview", "🏆 Top Biomarkers", "🔬 QC Summary"]
    if has_plots:    tabs.append("🖼️ Plots")
    if has_pathways: tabs.append("🧬 Pathways")

    rendered = st.tabs(tabs)
    idx = 0
    with rendered[idx]: _render_overview(state, session_id);   idx += 1
    with rendered[idx]: _render_biomarkers(state);              idx += 1
    with rendered[idx]: _render_qc(state);                      idx += 1
    if has_plots:
        with rendered[idx]: _render_plots(state, session_id);  idx += 1
    if has_pathways:
        with rendered[idx]: _render_pathways(state)


def _render_overview(state: dict[str, Any], session_id: str) -> None:
    st.markdown("### Analysis Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proteins",    state.get("n_proteins",   "–"))
    c2.metric("Samples",     state.get("n_samples",    "–"))
    c3.metric("Significant", state.get("n_significant","–"))
    c4.metric("Mode",        (state.get("analysis_mode") or "–").title())

    g1 = state.get("group1_label")
    g2 = state.get("group2_label")
    if g1 and g2:
        st.caption(f"Comparison: **{g1}** vs **{g2}**")

    st.divider()

    summary = state.get("analysis_summary")
    if summary:
        st.markdown("#### AI Summary")
        st.markdown(summary)
        st.divider()

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

    if state.get("qc_summary"):
        with st.expander("QC snapshot"):
            qc = state["qc_summary"]
            cols = st.columns(3)
            cols[0].metric("Proteins in",  qc.get("proteins_input",    "–"))
            cols[1].metric("After QC",     qc.get("proteins_after_qc", "–"))
            cols[2].metric("Removed",      qc.get("proteins_removed",  "–"))


def _render_biomarkers(state: dict[str, Any]) -> None:
    top = state.get("top_biomarkers")
    if not top:
        if state.get("status") in ("data_loaded", "routed", "ready"):
            st.info("Run the analysis to see biomarkers.")
        else:
            st.info("No biomarkers found yet.")
        return

    import pandas as pd

    df = pd.DataFrame(top)
    rename = {
        "protein":          "Protein",
        "rank":             "Rank",
        "log2_fold_change": "log2 FC",
        "p_value":          "p-value",
        "adj_p_value":      "adj. p-value",
        "significance":     "Significance",
        "cv_percent":       "CV %",
        "rescue_score":     "Rescue Score",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # For pooled results, show contrast columns too
    omic_type = state.get("omic_type", "")
    if omic_type == "proteomics_pooled":
        desired_cols = ["Rank", "Protein", "mdx_vs_WT", "uDys5_vs_mdx",
                        "H2_vs_mdx", "uDys5_vs_H2", "Rescue Score"]
    else:
        desired_cols = ["Rank", "Protein", "log2 FC", "p-value",
                        "adj. p-value", "Significance", "CV %"]

    show_cols = [c for c in desired_cols if c in df.columns]
    df_show = df[show_cols].copy() if show_cols else df.copy()

    for col in ("p-value", "adj. p-value"):
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f"{x:.3e}" if isinstance(x, float) else x
            )
    for col in ("log2 FC", "CV %", "Rescue Score",
                "mdx_vs_WT", "uDys5_vs_mdx", "H2_vs_mdx", "uDys5_vs_H2"):
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, float) else x
            )

    if "Rank" in df_show.columns:
        df_show = df_show.set_index("Rank")

    def _row_color(row):
        sig = str(row.get("Significance", "")).lower()
        if "highly" in sig:
            return ["background-color: #C6EFCE"] * len(row)
        if "significant" in sig:
            return ["background-color: #E2EFDA"] * len(row)
        if "trend" in sig:
            return ["background-color: #FFEB9C"] * len(row)
        return [""] * len(row)

    styled = df_show.style
    if "Significance" in df_show.columns:
        styled = styled.apply(_row_color, axis=1)

    st.dataframe(styled, use_container_width=True, height=min(600, 38 * len(df_show) + 40))

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

        df["class"] = df.apply(
            lambda r: "Significant"
            if (r.get("adj_p_value", 1) < 0.05 and abs(r.get("log2_fold_change", 0)) >= 1.0)
            else "NS",
            axis=1,
        )
        fig = px.scatter(
            df, x="log2_fold_change", y="-log10p", color="class",
            color_discrete_map={"Significant": "#C0392B", "NS": "#AAAAAA"},
            hover_name="protein",
            hover_data={"log2_fold_change": ":.3f", "adj_p_value": ":.3e",
                        "class": False, "-log10p": False, "adj_p_clipped": False},
            labels={"log2_fold_change": "log₂FC", "-log10p": "−log₁₀(adj. p)"},
            title=f"Volcano: {state.get('group1_label','G1')} vs {state.get('group2_label','G2')}",
        )
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="grey", line_width=1)
        fig.add_vline(x=1.0,  line_dash="dash", line_color="grey", line_width=1)
        fig.add_vline(x=-1.0, line_dash="dash", line_color="grey", line_width=1)
        fig.update_layout(legend_title_text="", height=380)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Volcano plot unavailable: {e}")


def _render_qc(state: dict[str, Any]) -> None:
    st.markdown("### QC Summary")
    qc_passed = state.get("qc_passed")
    qc = state.get("qc_summary") or {}

    if qc_passed is None:
        st.info("QC has not been run yet.")
        return

    # Pooled / pre-normalised designs: standard sample-level QC does not apply
    if qc.get("qc_type") == "pooled_prefilter":
        st.success("Data is pre-normalised ✅")
        note = qc.get("qc_note", "")
        if note:
            st.info(note)
        c1, c2 = st.columns(2)
        c1.metric("Proteins before filter", qc.get("proteins_before_filter", "–"))
        c2.metric("After contaminant removal", qc.get("proteins_after_qc", "–"))
        removed = qc.get("proteins_removed", 0)
        if removed:
            st.caption(f"Removed: **{removed}** contaminant / all-zero rows")
        return

    if qc_passed:
        st.success("Data passed quality control")
    else:
        st.warning("Quality issues detected — results may be affected")

    c1, c2, c3 = st.columns(3)
    c1.metric("Proteins in",  qc.get("proteins_input",    state.get("n_proteins", "–")))
    c2.metric("After QC",     qc.get("proteins_after_qc", "–"))
    c3.metric("Removed",      qc.get("proteins_removed",  "–"))

    c4, c5, c6 = st.columns(3)
    c4.metric("Samples in",   qc.get("samples_input",    state.get("n_samples", "–")))
    c5.metric("After QC",     qc.get("samples_after_qc", "–"))
    log2 = qc.get("log2_transformed")
    if log2 is not None:
        c6.metric("log2 transform", "Yes" if log2 else "No")

    missing_thr = qc.get("missing_threshold")
    if missing_thr is not None:
        st.caption(f"Missing value threshold: **{missing_thr * 100:.0f}%**")


def _render_plots(state: dict[str, Any], session_id: str) -> None:
    """Display all generated plots fetched from the API file endpoint."""
    st.markdown("### Generated Plots")
    plot_paths = state.get("plot_paths") or []
    if not plot_paths:
        st.info("No plots available yet.")
        return

    # Show plots in a 2-column grid
    cols = st.columns(2)
    for i, path in enumerate(plot_paths):
        filename = Path(path).name
        label    = filename.replace("_", " ").replace(".png", "").title()
        try:
            r = requests.get(
                f"{API_BASE}/results/{session_id}/file",
                params={"path": path},
                timeout=15,
            )
            if r.status_code == 200:
                with cols[i % 2]:
                    st.markdown(f"**{label}**")
                    st.image(r.content, use_container_width=True)
            else:
                cols[i % 2].warning(f"Could not load: {filename}")
        except Exception as exc:
            cols[i % 2].warning(f"Plot unavailable: {exc}")


def _render_pathways(state: dict[str, Any]) -> None:
    """Display pathway enrichment results."""
    st.markdown("### Pathway Enrichment Results")
    pathways = state.get("pathways") or []
    if not pathways:
        st.info("No enrichment results yet. Ask me to 'run pathway enrichment'.")
        return

    import pandas as pd

    df = pd.DataFrame(pathways)
    rename = {
        "pathway":    "Pathway",
        "library":    "Database",
        "p_value":    "p-value",
        "p_adjust":   "adj. p-value",
        "gene_count": "Genes Hit",
        "overlap":    "Overlap",
        "genes":      "Gene List",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ("p-value", "adj. p-value"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3e}" if isinstance(x, float) else x)

    show_cols = [c for c in ["Pathway", "Database", "adj. p-value", "Genes Hit", "Overlap"]
                 if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=min(500, 38 * len(df) + 40))

    if "Gene List" in df.columns:
        with st.expander("Gene lists per pathway"):
            for _, row in pd.DataFrame(pathways).head(5).iterrows():
                st.markdown(f"**{row.get('pathway','')}**")
                st.caption(str(row.get("genes", ""))[:300])


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session()
    _render_sidebar()
    _render_main()


main()
