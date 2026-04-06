"""
ui/app.py
Main Streamlit application for the Proteomics Biomarker Discovery Platform.

Run:
    streamlit run ui/app.py

Requires the FastAPI backend to be running:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# ── Path setup (run from biomarker-platform/ or project root) ─────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.components.chat import (
    render_chat_input,
    render_messages,
    render_suggested_prompts,
    render_welcome,
)
from ui.components.results_panel import render_results_panel
from ui.components.uploader import render_upload_success, render_uploader

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Proteomics Biomarker Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tighten chat area */
    .stChatMessage { padding: 0.5rem 0.75rem; }
    /* Reduce sidebar top padding */
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    /* Highlight pipeline status items */
    .status-complete  { color: #27ae60; font-weight: 600; }
    .status-pending   { color: #95a5a6; }
    .status-error     { color: #e74c3c; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Session management helpers
# ══════════════════════════════════════════════════════════════════════════════

def _create_api_session(disease_program: str, organism: str) -> str | None:
    try:
        resp = requests.post(
            f"{API_BASE}/chat/session",
            params={"disease_program": disease_program, "organism": organism},
            timeout=10,
        )
        if resp.status_code == 201:
            return resp.json()["session_id"]
    except requests.exceptions.ConnectionError:
        pass
    return None


def _init_session_state() -> None:
    """Initialise all st.session_state keys on first load."""
    defaults = {
        "session_id": None,
        "messages": [],          # list of {role, content}
        "analysis_state": {},    # last /results/{session_id} response
        "upload_result": None,   # last upload metadata
        "disease_program": "FA",
        "organism": "human",
        "sample_group_col": "",
        "contrast_group_1": "",
        "contrast_group_2": "",
        "api_error": None,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def _fetch_analysis_state(session_id: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE}/results/{session_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _send_message(session_id: str, message: str) -> dict | None:
    payload: dict = {"session_id": session_id, "message": message}

    # Attach analysis config if the user has filled it in
    grp_col = st.session_state.get("sample_group_col", "").strip()
    g1 = st.session_state.get("contrast_group_1", "").strip()
    g2 = st.session_state.get("contrast_group_2", "").strip()
    if grp_col:
        payload["sample_group_col"] = grp_col
    if g1 and g2:
        payload["contrast_groups"] = [g1, g2]
    payload["disease_program"] = st.session_state.get("disease_program", "FA")
    payload["organism"] = st.session_state.get("organism", "human")

    try:
        resp = requests.post(f"{API_BASE}/chat/", json=payload, timeout=300)
        if resp.status_code == 200:
            return resp.json()
        st.session_state["api_error"] = f"API {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        st.session_state["api_error"] = (
            "Cannot reach the API server. "
            "Run: `uvicorn api.main:app --reload --port 8000`"
        )
    except requests.exceptions.Timeout:
        st.session_state["api_error"] = "Request timed out. Analysis is still running."
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧬 Biomarker Discovery")
        st.caption("Proteomics · FA & DMD · Multi-Agent AI")
        st.divider()

        # ── Session config ────────────────────────────────────────────────────
        st.subheader("Session")
        disease = st.selectbox(
            "Disease program",
            ["FA", "DMD", "Other"],
            index=["FA", "DMD", "Other"].index(st.session_state["disease_program"]),
            key="sidebar_disease",
        )
        organism = st.selectbox(
            "Organism",
            ["human", "mouse", "rat"],
            index=["human", "mouse", "rat"].index(st.session_state["organism"]),
            key="sidebar_organism",
        )

        # Update session state and re-create API session if config changed
        if (
            disease != st.session_state["disease_program"]
            or organism != st.session_state["organism"]
        ):
            st.session_state["disease_program"] = disease
            st.session_state["organism"] = organism
            st.session_state["session_id"] = None  # force new session

        # Create API session if not yet created
        if st.session_state["session_id"] is None:
            sid = _create_api_session(disease, organism)
            if sid:
                st.session_state["session_id"] = sid
            else:
                st.error("⚠️ API server not reachable. Start the FastAPI backend.")
                st.code("uvicorn api.main:app --reload --port 8000")
                return

        st.caption(f"Session: `{st.session_state['session_id'][:8]}…`")

        st.divider()

        # ── File upload ───────────────────────────────────────────────────────
        st.subheader("Upload Data")
        upload_result = render_uploader(
            api_base=API_BASE,
            session_id=st.session_state["session_id"],
            disease_program=st.session_state["disease_program"],
            organism=st.session_state["organism"],
        )
        if upload_result:
            st.session_state["upload_result"] = upload_result
            render_upload_success(upload_result)
            # Refresh analysis state after upload
            st.session_state["analysis_state"] = _fetch_analysis_state(
                st.session_state["session_id"]
            )

        st.divider()

        # ── Analysis config ───────────────────────────────────────────────────
        st.subheader("Analysis Config")
        st.caption(
            "Specify your sample group column and contrast groups. "
            "Required for differential expression."
        )
        st.text_input(
            "Sample group column",
            placeholder="e.g.  Group  or  Condition",
            key="sample_group_col",
            help="The column name in your data that contains group labels.",
        )
        col1, col2 = st.columns(2)
        col1.text_input("Group 1 (case)", placeholder="Disease", key="contrast_group_1")
        col2.text_input("Group 2 (control)", placeholder="Control", key="contrast_group_2")

        st.divider()

        # ── Pipeline status ───────────────────────────────────────────────────
        st.subheader("Pipeline Status")
        astate = st.session_state.get("analysis_state", {})
        steps = [
            ("Data loaded",    bool(astate.get("data_type"))),
            ("QC passed",      bool(astate.get("qc_passed"))),
            ("DEA complete",   bool(astate.get("dea_result_path"))),
            ("Pathways done",  bool(astate.get("enrichment_result_path"))),
            ("Report ready",   bool(astate.get("report_path"))),
        ]
        for label, done in steps:
            icon = "✅" if done else "⭕"
            st.markdown(f"{icon} {label}")

        st.divider()

        # ── New session button ────────────────────────────────────────────────
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

    # Display any API errors
    if st.session_state.get("api_error"):
        st.error(st.session_state.pop("api_error"))

    # Two-column layout: chat (left) | results (right)
    chat_col, results_col = st.columns([5, 6], gap="large")

    # ── Chat column ───────────────────────────────────────────────────────────
    with chat_col:
        st.subheader("Chat")

        # ── Inline file uploader (shown until data is loaded) ────────────────
        astate = st.session_state.get("analysis_state", {})
        if not astate.get("data_type"):
            st.info("📎 Upload your proteomics data to get started", icon="📂")
            inline_upload = render_uploader(
                api_base=API_BASE,
                session_id=st.session_state["session_id"],
                disease_program=st.session_state["disease_program"],
                organism=st.session_state["organism"],
                widget_key="proteomics_file_uploader_inline",
            )
            if inline_upload:
                st.session_state["upload_result"] = inline_upload
                render_upload_success(inline_upload)
                st.session_state["analysis_state"] = _fetch_analysis_state(
                    st.session_state["session_id"]
                )
                st.rerun()

        # Message history
        chat_container = st.container(height=400, border=False)
        with chat_container:
            if not st.session_state["messages"]:
                render_welcome()
            else:
                render_messages(st.session_state["messages"])

        # Suggested prompts
        astate = st.session_state.get("analysis_state", {})
        suggested = render_suggested_prompts(
            data_loaded=bool(astate.get("data_type")),
            qc_done=bool(astate.get("qc_passed")),
            dea_done=bool(astate.get("dea_result_path")),
        )

        # Chat input (returns value only when submitted)
        user_input = render_chat_input() or suggested

        if user_input:
            # Append user message immediately
            st.session_state["messages"].append({"role": "user", "content": user_input})

            with st.spinner("Analysing …"):
                response = _send_message(session_id, user_input)

            if response:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response["response"]}
                )
                # Refresh analysis state from API
                st.session_state["analysis_state"] = _fetch_analysis_state(session_id)

            st.rerun()

    # ── Results column ────────────────────────────────────────────────────────
    with results_col:
        st.subheader("Results")
        render_results_panel(
            state=st.session_state.get("analysis_state", {}),
            api_base=API_BASE,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session_state()
    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()
else:
    # Streamlit discovers this file and runs it as a module
    main()
