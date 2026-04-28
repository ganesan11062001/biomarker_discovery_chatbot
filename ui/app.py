"""
ui/app.py  —  BiomarkerAI  (ChatGPT-style interface)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BiomarkerAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — ChatGPT look
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Page background ── */
.stApp { background: #212121; }
[data-testid="stMain"] { background: #212121; }

/* ── Sidebar (dark) ── */
[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2f2f2f;
}
[data-testid="stSidebar"] * { color: #ececec !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stMultiSelect label { color: #8e8ea0 !important; font-size:0.78rem !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] > div > div > input {
    background: #2f2f2f !important;
    border: 1px solid #3f3f3f !important;
    color: #ececec !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div {
    background: #2f2f2f !important;
    border: 1px solid #3f3f3f !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] hr { border-color: #2f2f2f !important; }
[data-testid="stSidebar"] .stMetric { background: #2a2a2a; border-radius:8px; padding:8px 12px; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ececec !important; font-size:1.2rem !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #8e8ea0 !important; font-size:0.72rem !important; }

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: #2f2f2f !important;
    color: #ececec !important;
    border: 1px solid #3f3f3f !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 6px 12px !important;
    width: 100%;
    text-align: left;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #3a3a3a !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #10a37f !important;
    border-color: #10a37f !important;
    color: #fff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background: #0d8a6b !important; }
[data-testid="stSidebar"] .stDownloadButton > button {
    background: #10a37f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    width: 100%;
}

/* ── Main chat column background ── */
.main-chat-col {
    max-width: 780px;
    margin: 0 auto;
    padding-bottom: 100px;
}

/* ── Chat message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
}
/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}
/* User bubble pill */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: #2f2f2f !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    max-width: 80% !important;
    margin-left: auto !important;
    color: #ececec !important;
}
/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
    background: transparent !important;
    padding: 4px 0 !important;
    color: #ececec !important;
}

/* ── Markdown inside messages ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td { color: #ececec !important; font-size: 0.95rem; line-height:1.65; }
[data-testid="stChatMessage"] h1,[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,[data-testid="stChatMessage"] h4 {
    color: #ececec !important; margin-top: 0.8rem; margin-bottom: 0.3rem;
}
[data-testid="stChatMessage"] code {
    background: #2f2f2f !important;
    color: #e2e2e2 !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
}
[data-testid="stChatMessage"] pre {
    background: #1e1e1e !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
[data-testid="stChatMessage"] table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88rem;
}
[data-testid="stChatMessage"] th {
    background: #2f2f2f !important;
    color: #ececec !important;
    padding: 7px 12px;
    text-align: left;
    border: 1px solid #3a3a3a;
}
[data-testid="stChatMessage"] td {
    padding: 6px 12px;
    border: 1px solid #2f2f2f;
    color: #d0d0d0 !important;
}
[data-testid="stChatMessage"] tr:nth-child(even) td { background: #262626; }
[data-testid="stChatMessage"] hr { border-color: #3a3a3a !important; }
[data-testid="stChatMessage"] strong { color: #fff !important; }

/* ── Chat input bar ── */
[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8e8ea0 !important; }
[data-testid="stChatInput"] button { color: #8e8ea0 !important; }
[data-testid="stChatInput"] button:hover { color: #ececec !important; }

/* ── File uploader (attachment) ── */
[data-testid="stFileUploader"] {
    background: #2f2f2f !important;
    border: 1.5px dashed #4a4a4a !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #2a2a2a !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #8e8ea0 !important; font-size:0.82rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { display:none !important; }

/* ── Quick-action suggestion pills ── */
.suggestion-grid { display:flex; flex-wrap:wrap; gap:10px; margin:16px 0; justify-content:center; }
.suggestion-pill {
    background: #2f2f2f;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 10px 16px;
    cursor: pointer;
    font-size: 0.88rem;
    color: #d0d0d0;
    transition: background 0.15s, border-color 0.15s;
    text-align: left;
    line-height: 1.4;
}
.suggestion-pill:hover { background: #3a3a3a; border-color: #555; color: #fff; }
.suggestion-pill .pill-icon { font-size: 1.1rem; margin-bottom: 4px; display:block; }
.suggestion-pill .pill-text { display:block; font-size:0.82rem; color:#8e8ea0; }

/* ── Quick action buttons (above input) ── */
.stButton > button {
    background: #2f2f2f !important;
    color: #d0d0d0 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    padding: 5px 14px !important;
    transition: background 0.15s;
}
.stButton > button:hover { background: #3a3a3a !important; color: #fff !important; }

/* ── Tabs in sidebar ── */
[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8e8ea0 !important;
    font-size: 0.78rem !important;
    padding: 4px 8px !important;
}
[data-testid="stSidebar"] .stTabs [aria-selected="true"] { color: #ececec !important; }

/* ── Expanders in dark theme ── */
[data-testid="stExpander"] {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #ececec !important; }

/* ── Welcome page ── */
.welcome-title { text-align:center; color:#ececec; font-size:1.9rem; font-weight:600; margin-top:60px; }
.welcome-sub { text-align:center; color:#8e8ea0; font-size:0.95rem; margin-bottom:32px; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

/* ── Attach chip ── */
.attach-chip {
    display:inline-flex; align-items:center; gap:8px;
    background:#2a2a2a; border:1px solid #3a3a3a; border-radius:20px;
    padding:5px 12px; font-size:0.82rem; color:#ececec; margin-bottom:8px;
}
.attach-chip .fname { font-weight:600; }
.attach-chip .fmeta { color:#8e8ea0; font-size:0.75rem; }

/* ── Section label (above attachment) ── */
.section-label { color:#8e8ea0; font-size:0.78rem; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)


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
        "session_id":      session_id,
        "message":         message,
        "disease_program": st.session_state.get("disease_program", "General"),
        "organism":        st.session_state.get("organism", "human"),
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
    file_bytes: bytes, filename: str, file_type: str,
    session_id: str, disease_program: str, organism: str,
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
        st.error("Cannot reach the API server.")
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
        "disease_program":   "General",
        "organism":          "human",
        "group1_samples":    [],
        "group2_samples":    [],
        "group1_label":      "Group1",
        "group2_label":      "Group2",
        "api_error":         None,
        "_attach_ver":       0,
        "_last_attach_name": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────────────
        st.markdown(
            "<div style='padding:12px 0 4px; font-size:1.15rem; font-weight:700;"
            "color:#ececec; letter-spacing:-0.3px;'>🧬 BiomarkerAI</div>"
            "<div style='font-size:0.72rem; color:#8e8ea0; margin-bottom:14px;'>"
            "Proteomics · Multi-Agent Platform</div>",
            unsafe_allow_html=True,
        )

        # ── New chat button ────────────────────────────────────────────────────
        if st.button("＋  New conversation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

        # ── Settings ──────────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#8e8ea0; "
            "text-transform:uppercase; letter-spacing:0.8px; "
            "margin-bottom:6px;'>Settings</div>",
            unsafe_allow_html=True,
        )
        disease = st.selectbox(
            "Disease / program",
            ["General", "DMD", "FA", "SMA", "Cancer", "Other"],
            index=["General", "DMD", "FA", "SMA", "Cancer", "Other"].index(
                st.session_state["disease_program"]
                if st.session_state["disease_program"] in ["General", "DMD", "FA", "SMA", "Cancer", "Other"]
                else "General"
            ),
            label_visibility="visible",
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

        # ── Ensure session ─────────────────────────────────────────────────────
        if st.session_state["session_id"] is None:
            sid = _api_create_session(disease, organism)
            if sid:
                st.session_state["session_id"] = sid
            else:
                st.error("API server not reachable.")
                st.code("uvicorn api.main:app --reload --port 8000", language="bash")
                return

        st.markdown(
            f"<div style='font-size:0.7rem; color:#555; margin-top:4px;'>"
            f"Session: {st.session_state['session_id'][:8]}…</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

        # ── Dataset (after upload) ─────────────────────────────────────────────
        ur = st.session_state.get("upload_result")
        if ur:
            st.markdown(
                "<div style='font-size:0.72rem; color:#8e8ea0; "
                "text-transform:uppercase; letter-spacing:0.8px; "
                "margin-bottom:8px;'>Dataset</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Proteins", ur.get("n_proteins", "–"))
            c2.metric("Samples",  ur.get("n_samples",  "–"))
            st.markdown(
                f"<div style='font-size:0.75rem; color:#8e8ea0; margin:4px 0 10px;'>"
                f"{ur.get('data_type','?').upper()} · "
                f"{(ur.get('data_format') or '').upper()}</div>",
                unsafe_allow_html=True,
            )

            sample_cols = ur.get("sample_columns") or []
            is_pooled   = bool(ur.get("is_pooled_design", False))
            label_map   = ur.get("label_map") or {}

            if is_pooled:
                st.info("Pooled design — analysis runs automatically.")
                if label_map:
                    with st.expander("Groups detected", expanded=False):
                        for code, name in label_map.items():
                            st.markdown(f"`{code}` → **{name}**")
                if st.button("▶  Run Fold-Change Analysis", type="primary", use_container_width=True):
                    _trigger_analysis(
                        st.session_state["session_id"],
                        "Run pooled fold-change analysis",
                    )

            elif sample_cols:
                st.markdown(
                    "<div style='font-size:0.72rem; color:#8e8ea0; "
                    "text-transform:uppercase; letter-spacing:0.8px; "
                    "margin-bottom:6px;'>Group Assignment</div>",
                    unsafe_allow_html=True,
                )
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    st.session_state["group1_label"] = st.text_input(
                        "Group 1", value=st.session_state.get("group1_label", "Group1"),
                        key="g1_label_input",
                    )
                with col_l2:
                    st.session_state["group2_label"] = st.text_input(
                        "Group 2", value=st.session_state.get("group2_label", "Group2"),
                        key="g2_label_input",
                    )
                g2_assigned = set(st.session_state.get("group2_samples") or [])
                g1_assigned = set(st.session_state.get("group1_samples") or [])
                st.session_state["group1_samples"] = st.multiselect(
                    f"Samples → {st.session_state['group1_label']}",
                    options=[c for c in sample_cols if c not in g2_assigned],
                    default=[c for c in (st.session_state.get("group1_samples") or [])
                             if c not in g2_assigned],
                    key="g1_multiselect",
                )
                st.session_state["group2_samples"] = st.multiselect(
                    f"Samples → {st.session_state['group2_label']}",
                    options=[c for c in sample_cols if c not in g1_assigned],
                    default=[c for c in (st.session_state.get("group2_samples") or [])
                             if c not in g1_assigned],
                    key="g2_multiselect",
                )
                g1_n = len(st.session_state["group1_samples"])
                g2_n = len(st.session_state["group2_samples"])
                if g1_n >= 2 and g2_n >= 2:
                    st.success(f"{g1_n} vs {g2_n} samples ready")
                    if st.button("▶  Run Analysis", type="primary", use_container_width=True):
                        g1l = st.session_state["group1_label"]
                        g2l = st.session_state["group2_label"]
                        _trigger_analysis(
                            st.session_state["session_id"],
                            f"Run differential expression analysis: {g1l} vs {g2l}",
                        )
                elif g1_n or g2_n:
                    st.warning("Need ≥2 samples per group.")

            st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

        # ── Pipeline status ────────────────────────────────────────────────────
        astate = st.session_state.get("analysis_state") or {}
        st.markdown(
            "<div style='font-size:0.72rem; color:#8e8ea0; "
            "text-transform:uppercase; letter-spacing:0.8px; "
            "margin-bottom:8px;'>Pipeline</div>",
            unsafe_allow_html=True,
        )
        steps = [
            ("Data loaded",       bool(astate.get("data_type"))),
            ("QC passed",         bool(astate.get("qc_passed"))),
            ("Analysis complete", astate.get("n_significant") is not None),
            ("Excel ready",       bool(astate.get("excel_path"))),
        ]
        for label, done in steps:
            icon  = "🟢" if done else "⚪"
            color = "#ececec" if done else "#555"
            st.markdown(
                f"<div style='font-size:0.82rem; color:{color}; "
                f"padding:3px 0;'>{icon} {label}</div>",
                unsafe_allow_html=True,
            )

        # ── Excel download ──────────────────────────────────────────────────────
        if astate.get("excel_path"):
            st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
            sid = st.session_state["session_id"]
            try:
                r = requests.get(f"{API_BASE}/results/{sid}/excel", timeout=30)
                if r.status_code == 200:
                    st.download_button(
                        "⬇  Download Excel Report",
                        data=r.content,
                        file_name=f"biomarkers_{sid[:8]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# File upload handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_file_attach(attached, session_id: str) -> None:
    if st.session_state.get("_last_attach_name") == attached.name:
        return
    st.session_state["_last_attach_name"] = attached.name

    size_kb = len(attached.getvalue()) // 1024
    st.session_state["messages"].append({
        "role":    "user",
        "content": f"📎 **{attached.name}**  ({size_kb} KB)",
    })

    with st.spinner(f"Processing **{attached.name}**…"):
        result = _api_upload_file(
            attached.getvalue(), attached.name,
            attached.type or "application/octet-stream",
            session_id,
            st.session_state.get("disease_program", "General"),
            st.session_state.get("organism", "human"),
        )

    if result:
        returned_sid = result.get("session_id") or session_id
        st.session_state["session_id"]     = returned_sid
        st.session_state["upload_result"]  = result
        st.session_state["group1_samples"] = []
        st.session_state["group2_samples"] = []
        st.session_state["analysis_state"] = _api_fetch_state(returned_sid)

        # Use the server response message directly
        server_msg = result.get("message") or _build_upload_message(result)
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": server_msg,
        })
        st.session_state["_attach_ver"] += 1

    st.rerun()


def _build_upload_message(result: dict) -> str:
    n_p  = result.get("n_proteins", "?")
    n_s  = result.get("n_samples", "?")
    dtype = result.get("data_type", "unknown")
    is_pooled = result.get("is_pooled_design", False)
    label_map = result.get("label_map") or {}
    lines = [
        f"**Data loaded** — {n_p} proteins · {n_s} samples · {dtype}",
    ]
    if is_pooled and label_map:
        groups = ", ".join(f"{k}→{v}" for k, v in label_map.items())
        lines += [f"\nPooled design detected: **{groups}**",
                  "Click **▶ Run Fold-Change Analysis** in the sidebar or type *run analysis*."]
    else:
        lines.append("\nAssign samples to groups in the sidebar, then run analysis.")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Trigger analysis
# ══════════════════════════════════════════════════════════════════════════════

def _trigger_analysis(session_id: str, message: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": message})
    with st.spinner("Running analysis…"):
        resp = _api_send_message(session_id, message)
    if resp:
        st.session_state["messages"].append({
            "role": "assistant", "content": resp["response"]
        })
        st.session_state["analysis_state"] = _api_fetch_state(session_id)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Welcome screen  (shown when no messages)
# ══════════════════════════════════════════════════════════════════════════════

def _render_welcome() -> str | None:
    st.markdown(
        "<div class='welcome-title'>What can I help with?</div>"
        "<div class='welcome-sub'>Upload proteomics data and ask anything — "
        "analysis, visualisation, enrichment, or general questions.</div>",
        unsafe_allow_html=True,
    )

    suggestions = [
        ("📊", "Run differential expression analysis",
         "Compare two groups and find significant biomarkers"),
        ("🔬", "What is a volcano plot?",
         "Explain the statistics used in proteomics"),
        ("🧬", "Run pathway enrichment on my results",
         "KEGG and GO enrichment analysis"),
        ("📈", "Generate PCA and heatmap",
         "Visualise sample clustering and top proteins"),
        ("💬", "Explain the Benjamini-Hochberg correction",
         "Statistical methods for multiple testing"),
        ("🔍", "Show me the top 20 biomarkers",
         "Ranked list with fold changes and p-values"),
    ]

    # 3-column grid of suggestion cards
    cols = st.columns(3)
    for i, (icon, title, sub) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(
                f"{icon} {title}\n\n_{sub}_",
                key=f"sug_{i}",
                use_container_width=True,
            ):
                return title
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Inline plot renderer  (shown inside assistant messages)
# ══════════════════════════════════════════════════════════════════════════════

def _render_inline_plots(session_id: str, plot_paths: list[str]) -> None:
    if not plot_paths:
        return
    n = len(plot_paths)
    cols_per_row = 2 if n > 1 else 1
    cols = st.columns(cols_per_row)
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
                with cols[i % cols_per_row]:
                    st.image(r.content, caption=label, use_container_width=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Quick actions (above input bar)
# ══════════════════════════════════════════════════════════════════════════════

def _render_quick_actions(astate: dict) -> str | None:
    data_loaded   = bool(astate.get("data_type"))
    analysis_done = bool(astate.get("excel_path"))

    if analysis_done:
        actions = [
            "Summarise results",
            "Show top 10 biomarkers",
            "Generate standard plots",
            "Run pathway enrichment",
            "Show analysis code",
        ]
    elif data_loaded:
        ur        = st.session_state.get("upload_result") or {}
        is_pooled = bool(ur.get("is_pooled_design"))
        g1        = st.session_state.get("group1_samples") or []
        g2        = st.session_state.get("group2_samples") or []
        if is_pooled or (g1 and g2):
            actions = ["Run analysis", "Describe the dataset", "What groups are detected?"]
        else:
            actions = ["Describe the dataset", "What proteins are in my data?"]
    else:
        return None

    cols = st.columns(len(actions))
    for col, action in zip(cols, actions):
        if col.button(action, key=f"qa_{action[:15]}", use_container_width=True):
            return action
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main  —  chat area
# ══════════════════════════════════════════════════════════════════════════════

def _render_main() -> None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.markdown(
            "<div style='text-align:center; color:#8e8ea0; padding-top:100px;'>"
            "Connecting to API…</div>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.get("api_error"):
        err = st.session_state.pop("api_error")
        st.error(err)

    # ── Centre the chat column ─────────────────────────────────────────────────
    _, chat_col, _ = st.columns([1, 6, 1])

    with chat_col:
        messages = st.session_state.get("messages") or []
        astate   = st.session_state.get("analysis_state") or {}

        # ── Welcome / empty state ──────────────────────────────────────────────
        if not messages:
            triggered = _render_welcome()
            if triggered:
                st.session_state["_quick"] = triggered
                st.rerun()

            # File attachment on welcome screen
            st.markdown("<br>", unsafe_allow_html=True)
            _render_attachment_area(session_id)
            return

        # ── Message history ────────────────────────────────────────────────────
        plot_paths = astate.get("plot_paths") or []
        shown_plots = False

        for idx, m in enumerate(messages):
            role    = m.get("role", "assistant")
            content = m.get("content", "")

            with st.chat_message(role, avatar="🧬" if role == "assistant" else None):
                st.markdown(content)

                # After the last assistant message, show plots inline
                if (role == "assistant"
                        and idx == len(messages) - 1
                        and plot_paths
                        and not shown_plots):
                    _render_inline_plots(session_id, plot_paths)
                    shown_plots = True

        # ── Quick action pills ─────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        quick_action = _render_quick_actions(astate)

        # ── Attachment area ────────────────────────────────────────────────────
        _render_attachment_area(session_id)

        # ── Chat input ─────────────────────────────────────────────────────────
        user_input = st.chat_input("Message BiomarkerAI…")
        pending    = st.session_state.pop("_quick", None)
        user_input = user_input or quick_action or pending

        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.spinner(""):
                resp = _api_send_message(session_id, user_input)
            if resp:
                st.session_state["messages"].append({
                    "role": "assistant", "content": resp["response"]
                })
                new_sid = resp.get("session_id")
                if new_sid and new_sid != session_id:
                    st.session_state["session_id"]     = new_sid
                    st.session_state["upload_result"]  = {}
                    st.session_state["analysis_state"] = {}
                else:
                    st.session_state["analysis_state"] = _api_fetch_state(session_id)
            st.rerun()


def _render_attachment_area(session_id: str) -> None:
    """Compact file attachment row — shown above the chat input."""
    attach_ver    = st.session_state.get("_attach_ver", 0)
    attached_file = st.file_uploader(
        "📎 Attach proteomics file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key=f"attach_{attach_ver}",
        label_visibility="collapsed",
        help="Drag & drop or browse — CSV, XLSX, XLS supported",
    )
    if attached_file is not None:
        _handle_file_attach(attached_file, session_id)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session()
    _render_sidebar()
    _render_main()


main()
