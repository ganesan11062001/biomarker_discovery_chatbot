"""
ui/app.py  —  BiomarkerAI  (BioSpace modern UI)
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BiomarkerAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  —  BioSpace Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none; }

/* ── Page background ── */
.stApp {
    background: linear-gradient(160deg, #060b18 0%, #0b1428 55%, #0d1a2f 100%) !important;
    min-height: 100vh;
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 1rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #08101e !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div { padding: 0 0.8rem !important; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Brand ── */
.brand-wrap {
    padding: 18px 4px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 14px;
}
.brand-name {
    font-size: 1.15rem;
    font-weight: 800;
    background: linear-gradient(120deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    display: block;
}
.brand-tag {
    font-size: 0.67rem;
    color: #475569 !important;
    letter-spacing: 0.6px;
    display: block;
    margin-top: 2px;
}

/* ── Section header ── */
.sh {
    font-size: 0.65rem;
    font-weight: 700;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 14px 0 7px;
    display: block;
}

/* ── Glass card ── */
.gcard {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

/* ── Metric chips ── */
.mchips { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 6px 0 10px; }
.mchip {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 10px;
    padding: 9px 10px;
    text-align: center;
}
.mchip-val {
    display: block;
    font-size: 1.25rem;
    font-weight: 700;
    background: linear-gradient(120deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.mchip-lbl {
    display: block;
    font-size: 0.63rem;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ── Badge chip (data type) ── */
.dtype-badge {
    display: inline-block;
    background: rgba(129,140,248,0.1);
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.7rem;
    color: #818cf8 !important;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

/* ── Pipeline steps ── */
.pipeline { display: flex; flex-direction: column; gap: 5px; margin: 4px 0; }
.pstep {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 11px;
    border-radius: 9px;
    font-size: 0.8rem;
    font-weight: 500;
    transition: all 0.2s;
}
.pstep.done {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.18);
    color: #34d399 !important;
}
.pstep.running {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.22);
    color: #38bdf8 !important;
}
.pstep.idle {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    color: #334155 !important;
}
.pdot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pdot.done { background: #34d399; box-shadow: 0 0 6px #34d399; }
.pdot.running {
    background: #38bdf8;
    box-shadow: 0 0 8px #38bdf8;
    animation: blink 1.6s ease-in-out infinite;
}
.pdot.idle { background: #1e293b; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Session chip ── */
.session-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.67rem;
    color: #475569 !important;
    margin-bottom: 10px;
}
.session-dot { width: 5px; height: 5px; border-radius: 50%; background: #34d399; box-shadow: 0 0 5px #34d399; }

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.03) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 9px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 7px 13px !important;
    width: 100%;
    text-align: left;
    transition: all 0.18s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(56,189,248,0.3) !important;
    background: rgba(56,189,248,0.05) !important;
    color: #38bdf8 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(129,140,248,0.12)) !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.2), rgba(129,140,248,0.2)) !important;
    box-shadow: 0 2px 16px rgba(56,189,248,0.18) !important;
}
[data-testid="stSidebar"] .stDownloadButton > button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    width: 100%;
    box-shadow: 0 2px 14px rgba(56,189,248,0.28);
    font-size: 0.82rem !important;
}

/* ── Sidebar form controls ── */
[data-testid="stSidebar"] label {
    color: #475569 !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.05) !important; margin: 10px 0 !important; }
[data-testid="stSidebar"] .stMetric { background: transparent !important; }
[data-testid="stSidebar"] .stAlert {
    border-radius: 9px !important;
    font-size: 0.78rem !important;
}

/* ── Expanders in sidebar ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 9px !important;
}

/* ── Main chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 3px 0 !important;
}
/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: rgba(56,189,248,0.05) !important;
    border: 1px solid rgba(56,189,248,0.14) !important;
    border-radius: 18px 18px 5px 18px !important;
    padding: 12px 18px !important;
    max-width: 78% !important;
    margin-left: auto !important;
    backdrop-filter: blur(8px);
}
/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
    background: transparent !important;
    padding: 4px 0 !important;
}
/* Avatar icon */
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15)) !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    border-radius: 50% !important;
}

/* ── Text in messages ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td { color: #cbd5e1 !important; font-size: 0.94rem; line-height: 1.75; }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 { color: #e2e8f0 !important; margin: 0.8rem 0 0.3rem; }
[data-testid="stChatMessage"] strong { color: #f1f5f9 !important; }
[data-testid="stChatMessage"] em { color: #94a3b8 !important; }
[data-testid="stChatMessage"] code {
    background: rgba(56,189,248,0.08) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.18) !important;
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 0.84em;
}
[data-testid="stChatMessage"] pre {
    background: rgba(6,11,24,0.7) !important;
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    backdrop-filter: blur(8px);
}
[data-testid="stChatMessage"] table { border-collapse: collapse; width: 100%; font-size: 0.87rem; }
[data-testid="stChatMessage"] th {
    background: rgba(56,189,248,0.07) !important;
    color: #38bdf8 !important;
    padding: 8px 14px;
    border: 1px solid rgba(56,189,248,0.12);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
}
[data-testid="stChatMessage"] td {
    padding: 7px 14px;
    border: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1 !important;
}
[data-testid="stChatMessage"] tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
[data-testid="stChatMessage"] hr { border-color: rgba(255,255,255,0.07) !important; }
[data-testid="stChatMessage"] blockquote {
    border-left: 3px solid #38bdf8 !important;
    margin: 0;
    padding: 4px 14px;
    background: rgba(56,189,248,0.04);
    border-radius: 0 8px 8px 0;
    color: #94a3b8 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    transition: all 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(56,189,248,0.4) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.07), 0 0 20px rgba(56,189,248,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-size: 0.94rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }
[data-testid="stChatInput"] button { color: #475569 !important; }
[data-testid="stChatInput"] button:hover { color: #38bdf8 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(56,189,248,0.02) !important;
    border: 1.5px dashed rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
    transition: all 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.45) !important;
    background: rgba(56,189,248,0.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #475569 !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }

/* ── Quick action pill buttons ── */
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    transition: all 0.18s;
}
.stButton > button:hover {
    border-color: rgba(56,189,248,0.35) !important;
    color: #38bdf8 !important;
    background: rgba(56,189,248,0.06) !important;
    box-shadow: 0 0 12px rgba(56,189,248,0.1) !important;
}

/* ── Welcome hero ── */
.hero {
    text-align: center;
    padding: 56px 20px 36px;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: #38bdf8;
    margin-bottom: 22px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 30%, #38bdf8 65%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.12;
    margin-bottom: 18px;
    letter-spacing: -1.2px;
}
.hero-sub {
    font-size: 1rem;
    color: #475569;
    max-width: 460px;
    margin: 0 auto 40px;
    line-height: 1.75;
}

/* ── Feature cards grid ── */
.fc-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 11px;
    max-width: 720px;
    margin: 0 auto 32px;
}
.fc-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 14px;
    text-align: left;
    transition: all 0.22s;
    cursor: pointer;
}
.fc-card:hover {
    border-color: rgba(56,189,248,0.28);
    background: rgba(56,189,248,0.04);
    box-shadow: 0 4px 24px rgba(56,189,248,0.08);
    transform: translateY(-2px);
}
.fc-icon { font-size: 1.3rem; margin-bottom: 9px; display: block; }
.fc-title { font-size: 0.86rem; font-weight: 600; color: #cbd5e1; margin-bottom: 5px; }
.fc-desc { font-size: 0.73rem; color: #475569; line-height: 1.55; }

/* ── Upload zone ── */
.upload-cta {
    background: rgba(56,189,248,0.03);
    border: 1.5px dashed rgba(56,189,248,0.18);
    border-radius: 16px;
    padding: 20px 24px 14px;
    margin-bottom: 16px;
    text-align: center;
    transition: all 0.22s;
}
.upload-cta:hover { border-color: rgba(56,189,248,0.4); background: rgba(56,189,248,0.05); }
.upload-cta-title { font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 4px; }
.upload-cta-sub { font-size: 0.73rem; color: #475569; margin-bottom: 12px; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.83rem !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(56,189,248,0.07) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
}

/* ── Expander (main area) ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #94a3b8 !important; font-size: 0.83rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.3); }
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
            "Cannot reach API. Run: `uvicorn api.main:app --reload --port 8000`"
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
            "<div class='brand-wrap'>"
            "<span class='brand-name'>🧬 BiomarkerAI</span>"
            "<span class='brand-tag'>Proteomics · Multi-Agent Platform</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── New chat button ───────────────────────────────────────────────────
        if st.button("＋  New Conversation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # ── Settings ──────────────────────────────────────────────────────────
        st.markdown("<span class='sh'>Settings</span>", unsafe_allow_html=True)
        disease = st.selectbox(
            "Disease / Program",
            ["General", "DMD", "FA", "SMA", "Cancer", "Other"],
            index=["General", "DMD", "FA", "SMA", "Cancer", "Other"].index(
                st.session_state["disease_program"]
                if st.session_state["disease_program"] in ["General", "DMD", "FA", "SMA", "Cancer", "Other"]
                else "General"
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
            f"<div class='session-chip'>"
            f"<span class='session-dot'></span>"
            f"Session {st.session_state['session_id'][:8]}…"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Dataset card (after upload) ────────────────────────────────────────
        ur = st.session_state.get("upload_result")
        if ur:
            st.markdown("<span class='sh'>Dataset</span>", unsafe_allow_html=True)

            dtype = (ur.get("data_type") or "unknown").upper()
            dformat = (ur.get("data_format") or "").upper()
            badge_label = f"{dtype}" + (f" · {dformat}" if dformat else "")
            st.markdown(
                f"<div class='gcard'>"
                f"<span class='dtype-badge'>{badge_label}</span>"
                f"<div class='mchips'>"
                f"<div class='mchip'><span class='mchip-val'>{ur.get('n_proteins','–')}</span><span class='mchip-lbl'>Proteins</span></div>"
                f"<div class='mchip'><span class='mchip-val'>{ur.get('n_samples','–')}</span><span class='mchip-lbl'>Samples</span></div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            sample_cols = ur.get("sample_columns") or []
            is_pooled   = bool(ur.get("is_pooled_design", False))
            label_map   = ur.get("label_map") or {}

            if is_pooled:
                st.info("Pooled design — groups auto-detected.")
                if label_map:
                    with st.expander("Detected groups", expanded=False):
                        for code, name in label_map.items():
                            st.markdown(f"`{code}` → **{name}**")
                if st.button("▶  Run Fold-Change Analysis", type="primary", use_container_width=True):
                    _trigger_analysis(
                        st.session_state["session_id"],
                        "Run pooled fold-change analysis",
                    )

            elif sample_cols:
                st.markdown("<span class='sh'>Group Assignment</span>", unsafe_allow_html=True)
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
                    f"→ {st.session_state['group1_label']}",
                    options=[c for c in sample_cols if c not in g2_assigned],
                    default=[c for c in (st.session_state.get("group1_samples") or [])
                             if c not in g2_assigned],
                    key="g1_multiselect",
                )
                st.session_state["group2_samples"] = st.multiselect(
                    f"→ {st.session_state['group2_label']}",
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
                    st.warning("Need ≥ 2 samples per group.")

        # ── Pipeline steps ─────────────────────────────────────────────────────
        astate = st.session_state.get("analysis_state") or {}
        st.markdown("<span class='sh'>Pipeline</span>", unsafe_allow_html=True)

        def _pstep(label: str, icon: str, state_class: str) -> str:
            return (
                f"<div class='pstep {state_class}'>"
                f"<span class='pdot {state_class}'></span>"
                f"{icon} {label}"
                f"</div>"
            )

        data_done    = bool(astate.get("data_type"))
        analysis_done = astate.get("n_significant") is not None
        excel_done   = bool(astate.get("excel_path"))
        enrich_done  = bool(astate.get("pathways"))
        plots_done   = bool(astate.get("plot_paths"))

        st.markdown(
            "<div class='pipeline'>"
            + _pstep("Data Loaded",   "📂", "done" if data_done else "idle")
            + _pstep("Analysis",      "🔬", "done" if analysis_done else ("running" if data_done else "idle"))
            + _pstep("Enrichment",    "🧬", "done" if enrich_done else "idle")
            + _pstep("Visualisation", "📊", "done" if plots_done else "idle")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Excel download ──────────────────────────────────────────────────────
        if excel_done:
            st.markdown("<hr>", unsafe_allow_html=True)
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

        server_msg = result.get("message") or _build_upload_message(result)
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": server_msg,
        })
        st.session_state["_attach_ver"] += 1

    st.rerun()


def _build_upload_message(result: dict) -> str:
    n_p       = result.get("n_proteins", "?")
    n_s       = result.get("n_samples", "?")
    dtype     = result.get("data_type", "unknown")
    is_pooled = result.get("is_pooled_design", False)
    label_map = result.get("label_map") or {}
    lines = [f"**Data loaded** — {n_p} proteins · {n_s} samples · {dtype}"]
    if is_pooled and label_map:
        groups = ", ".join(f"{k}→{v}" for k, v in label_map.items())
        lines += [
            f"\nPooled design detected: **{groups}**",
            "Click **▶ Run Fold-Change Analysis** in the sidebar or type *run analysis*.",
        ]
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
# Welcome screen
# ══════════════════════════════════════════════════════════════════════════════

def _render_welcome() -> str | None:
    st.markdown(
        "<div class='hero'>"
        "<div class='hero-eyebrow'>⚡ Multi-Agent · LangGraph · Azure OpenAI</div>"
        "<div class='hero-title'>Biomarker Discovery,<br>Reimagined</div>"
        "<div class='hero-sub'>Upload your proteomics data and ask anything — "
        "from differential analysis to pathway enrichment and interactive visualisations.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    suggestions = [
        ("📊", "Run differential expression analysis",
         "Compare two groups and find significant biomarkers"),
        ("🧬", "Run pathway enrichment on my results",
         "KEGG, GO, Reactome & WikiPathways"),
        ("📈", "Generate PCA and heatmap",
         "Visualise sample clustering and top proteins"),
        ("🔬", "What is a volcano plot?",
         "Explain the statistics used in proteomics"),
        ("💬", "Explain Benjamini-Hochberg FDR",
         "Multiple testing correction for omics"),
        ("🔍", "Show top 20 biomarkers",
         "Ranked list with fold changes and p-values"),
    ]

    # Inject the feature-card grid via HTML (non-interactive visual)
    cards_html = "<div class='fc-grid'>"
    for icon, title, desc in suggestions:
        cards_html += (
            f"<div class='fc-card'>"
            f"<span class='fc-icon'>{icon}</span>"
            f"<div class='fc-title'>{title}</div>"
            f"<div class='fc-desc'>{desc}</div>"
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Invisible buttons that map to each card (using st.columns for click logic)
    cols = st.columns(3)
    for i, (_, title, _desc) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(title, key=f"sug_{i}", use_container_width=True,
                         help=_desc):
                return title

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Inline plot renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_inline_plots(session_id: str, plot_paths: list[str]) -> None:
    if not plot_paths:
        return
    n             = len(plot_paths)
    cols_per_row  = 2 if n > 1 else 1
    cols          = st.columns(cols_per_row)
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
# Quick action chips (above input)
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
# Attachment area
# ══════════════════════════════════════════════════════════════════════════════

def _render_attachment_area(session_id: str, welcome: bool = False) -> None:
    if welcome:
        st.markdown(
            "<div class='upload-cta'>"
            "<div class='upload-cta-title'>📎 Attach your proteomics file to get started</div>"
            "<div class='upload-cta-sub'>Supports CSV, XLSX, XLS · Max 100 MB</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    attached_file = st.file_uploader(
        "Attach proteomics file",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key=f"attach_{st.session_state.get('_attach_ver', 0)}",
        label_visibility="collapsed",
        help="Drag & drop or browse — CSV, XLSX, XLS supported",
    )
    if attached_file is not None:
        _handle_file_attach(attached_file, session_id)


# ══════════════════════════════════════════════════════════════════════════════
# Main chat area
# ══════════════════════════════════════════════════════════════════════════════

def _render_main() -> None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.markdown(
            "<div style='text-align:center; color:#475569; padding-top:120px; font-size:0.9rem;'>"
            "Connecting to API server…</div>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.get("api_error"):
        err = st.session_state.pop("api_error")
        st.error(err)

    # Centre column
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
            st.markdown("<br>", unsafe_allow_html=True)
            _render_attachment_area(session_id, welcome=True)
            return

        # ── Message history ────────────────────────────────────────────────────
        plot_paths = astate.get("plot_paths") or []

        for m in messages:
            role    = m.get("role", "assistant")
            content = m.get("content", "")

            with st.chat_message(role, avatar="🧬" if role == "assistant" else None):
                st.markdown(content)

                if role == "assistant" and m.get("has_plots") and plot_paths:
                    _render_inline_plots(session_id, plot_paths)

        # ── Quick action chips ─────────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        quick_action = _render_quick_actions(astate)

        # ── Attachment area ────────────────────────────────────────────────────
        _render_attachment_area(session_id)

        # ── Chat input ─────────────────────────────────────────────────────────
        user_input = st.chat_input("Ask BiomarkerAI anything…")
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


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session()
    _render_sidebar()
    _render_main()


main()
