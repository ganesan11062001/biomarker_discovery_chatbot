"""
ui/app.py  —  BiomarkerAI  (chat-only, no sidebar)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import json as _json

import plotly.graph_objects as _go
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
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  —  BioSpace Theme (chat-only)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stSidebar"] { display: none !important; }

/* ── Page background ── */
.stApp {
    background: linear-gradient(160deg, #060b18 0%, #0b1428 55%, #0d1a2f 100%) !important;
    min-height: 100vh;
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 0.5rem !important; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 6px;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-name {
    font-size: 1.05rem;
    font-weight: 800;
    background: linear-gradient(120deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.4px;
}
.topbar-tag {
    font-size: 0.65rem;
    color: #334155;
    letter-spacing: 0.5px;
}
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
}
.session-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 5px #34d399;
    display: inline-block;
}

/* ── Pipeline status strip ── */
.pstrip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0 10px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 4px;
    flex-wrap: wrap;
}
.pstep {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 500;
}
.pstep.done {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.2);
    color: #34d399;
}
.pstep.idle {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    color: #1e293b;
}
.pdot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.pdot.done { background: #34d399; box-shadow: 0 0 5px #34d399; }
.pdot.idle { background: #1e293b; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 3px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: rgba(56,189,248,0.05) !important;
    border: 1px solid rgba(56,189,248,0.14) !important;
    border-radius: 18px 18px 5px 18px !important;
    padding: 12px 18px !important;
    max-width: 78% !important;
    margin-left: auto !important;
    backdrop-filter: blur(8px);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
    background: transparent !important;
    padding: 4px 0 !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15)) !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    border-radius: 50% !important;
}

/* ── Message text ── */
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

/* ── Quick action / top-bar pill buttons ── */
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

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(129,140,248,0.12)) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    color: #38bdf8 !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 5px 14px !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(129,140,248,0.22)) !important;
    box-shadow: 0 2px 16px rgba(56,189,248,0.18) !important;
}

/* ── Welcome hero ── */
.hero {
    text-align: center;
    padding: 48px 20px 32px;
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
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 30%, #38bdf8 65%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.12;
    margin-bottom: 16px;
    letter-spacing: -1.1px;
}
.hero-sub {
    font-size: 1rem;
    color: #475569;
    max-width: 520px;
    margin: 0 auto 36px;
    line-height: 1.75;
}

/* ── Feature cards ── */
.fc-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 11px;
    max-width: 740px;
    margin: 0 auto 28px;
}
.fc-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 14px;
    text-align: left;
    transition: all 0.22s;
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
    padding: 18px 24px 12px;
    margin-bottom: 14px;
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

/* ── Expander ── */
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

def _api_create_session(disease_program: str = "General", organism: str = "human") -> str | None:
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
    """Send a chat message. Groups come entirely from the LLM — not pre-set by the UI."""
    payload: dict[str, Any] = {
        "session_id":      session_id,
        "message":         message,
        "disease_program": st.session_state.get("disease_program", "General"),
        "organism":        st.session_state.get("organism", "human"),
    }
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
    file_bytes: bytes, filename: str, file_type: str, session_id: str,
) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/upload/",
            files={"file": (filename, file_bytes, file_type)},
            data={
                "session_id":      session_id,
                "disease_program": st.session_state.get("disease_program", "General"),
                "organism":        st.session_state.get("organism", "human"),
            },
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
        "session_id":     None,
        "messages":       [],
        "analysis_state": {},
        "upload_result":  None,
        "disease_program": "General",
        "organism":        "human",
        "api_error":       None,
        "_attach_ver":     0,
        "_last_attach_name": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _ensure_session() -> str | None:
    """Create a backend session on first load if one doesn't exist yet."""
    if st.session_state.get("session_id"):
        return st.session_state["session_id"]
    sid = _api_create_session(
        st.session_state.get("disease_program", "General"),
        st.session_state.get("organism", "human"),
    )
    if sid:
        st.session_state["session_id"] = sid
    return sid


# ══════════════════════════════════════════════════════════════════════════════
# Top bar
# ══════════════════════════════════════════════════════════════════════════════

def _render_topbar(session_id: str | None, astate: dict) -> None:
    _, bar_col, _ = st.columns([1, 10, 1])
    with bar_col:
        left, right = st.columns([4, 2])

        with left:
            sid_text = f"Session {session_id[:8]}…" if session_id else "No session"
            st.markdown(
                "<div class='topbar-brand'>"
                "<span class='topbar-name'>🧬 BiomarkerAI</span>"
                "<span class='topbar-tag'>Proteomics · Multi-Agent Platform</span>"
                f"<span class='session-chip'><span class='session-dot'></span>{sid_text}</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        with right:
            # Pipeline status strip — compact dots in the top bar
            data_done     = bool(astate.get("data_type"))
            analysis_done = astate.get("n_significant") is not None
            enrich_done   = bool(astate.get("pathways"))
            plots_done    = bool(astate.get("plot_paths"))

            def _dot(label: str, icon: str, done: bool) -> str:
                cls = "done" if done else "idle"
                return (
                    f"<span class='pstep {cls}'>"
                    f"<span class='pdot {cls}'></span>{icon} {label}"
                    f"</span>"
                )

            st.markdown(
                "<div class='pstrip'>"
                + _dot("Data", "📂", data_done)
                + _dot("Analysis", "🔬", analysis_done)
                + _dot("Enrichment", "🧬", enrich_done)
                + _dot("Plots", "📊", plots_done)
                + "</div>",
                unsafe_allow_html=True,
            )

        # New conversation button — full row below
        if st.button("＋  New Conversation", key="new_conv"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("<div style='border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:8px'></div>",
                    unsafe_allow_html=True)


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
        )

    if result:
        returned_sid = result.get("session_id") or session_id
        st.session_state["session_id"]     = returned_sid
        st.session_state["upload_result"]  = result
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
    sample_cols = result.get("sample_columns") or []

    lines = [f"**Data loaded** — {n_p} proteins · {n_s} samples · `{dtype}`"]

    if is_pooled and label_map:
        groups = ", ".join(f"`{k}` → **{v}**" for k, v in label_map.items())
        lines += [
            f"\nPooled design detected: {groups}",
            "\nType **run analysis** to compute log₂ fold-changes across all groups.",
        ]
    else:
        if sample_cols:
            preview = ", ".join(f"`{c}`" for c in sample_cols[:8])
            more    = f" … (+{len(sample_cols)-8} more)" if len(sample_cols) > 8 else ""
            lines.append(f"\nSample columns: {preview}{more}")
        lines.append(
            "\nTell me which groups to compare — for example:\n"
            "> *\"Compare Control_1, Control_2, Control_3 vs Disease_1, Disease_2, Disease_3\"*\n\n"
            "Or type **run all comparisons** to auto-detect groups and analyse everything."
        )
    return "\n".join(lines)


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

    cols = st.columns(3)
    for i, (_, title, _desc) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(title, key=f"sug_{i}", use_container_width=True, help=_desc):
                return title

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Inline plot renderer  (PNG grid + interactive HTML expander)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_file(session_id: str, path: str) -> bytes | None:
    try:
        r = requests.get(
            f"{API_BASE}/results/{session_id}/file",
            params={"path": path},
            timeout=20,
        )
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def _render_inline_plots(session_id: str, plot_paths: list[str]) -> None:
    """Render PNG thumbnail grid inside a chat message bubble."""
    if not plot_paths:
        return

    # Only PNG paths (skip any HTML paths that ended up in the list)
    png_paths = [p for p in plot_paths if str(p).endswith(".png")]
    if not png_paths:
        return

    st.markdown(
        "<div style='margin:14px 0 8px;font-size:0.82rem;color:#475569;"
        "font-weight:600;letter-spacing:0.4px;text-transform:uppercase;'>"
        f"📊 {len(png_paths)} plot{'s' if len(png_paths) != 1 else ''} generated</div>",
        unsafe_allow_html=True,
    )

    cols_per_row = 2
    rows = [png_paths[i:i + cols_per_row] for i in range(0, len(png_paths), cols_per_row)]

    for row_paths in rows:
        cols = st.columns(len(row_paths))
        for col, path in zip(cols, row_paths):
            label = (
                Path(path).stem
                .split("_", 1)[-1]          # strip "stem_" prefix
                .replace("_", " ")
                .title()
            )
            img_bytes = _fetch_file(session_id, path)
            if img_bytes:
                with col:
                    st.image(img_bytes, caption=label, use_container_width=True)


def _render_interactive_plots(session_id: str, plot_paths: list[str]) -> None:
    """Render interactive Plotly charts in a standalone expander OUTSIDE chat messages."""
    png_paths = [p for p in plot_paths if str(p).endswith(".png")]
    if not png_paths:
        return

    with st.expander("🔬 Explore plots interactively  (zoom · hover · pan)", expanded=False):
        for path in png_paths:
            label = (
                Path(path).stem
                .split("_", 1)[-1]
                .replace("_", " ")
                .title()
            )
            json_path = path.replace(".png", ".json")
            json_bytes = _fetch_file(session_id, json_path)
            if not json_bytes:
                continue
            try:
                fig = _go.Figure(_json.loads(json_bytes.decode("utf-8")))
                st.markdown(
                    f"<div style='font-size:0.82rem;font-weight:600;color:#94a3b8;"
                    f"margin:16px 0 4px;'>📊 {label}</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{Path(path).stem}")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Quick action chips
# ══════════════════════════════════════════════════════════════════════════════

def _render_quick_actions(session_id: str, astate: dict) -> str | None:
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
        actions = [
            "Run all comparisons",
            "Describe the dataset",
            "What groups are in my data?",
        ]
    else:
        return None

    # Excel download — shown alongside quick actions when available
    excel_path = astate.get("excel_path")
    if excel_path and session_id:
        dl_col, chips_col = st.columns([1, 4])
        with dl_col:
            try:
                r = requests.get(f"{API_BASE}/results/{session_id}/excel", timeout=20)
                if r.status_code == 200:
                    st.download_button(
                        "⬇ Download Excel",
                        data=r.content,
                        file_name=f"biomarkers_{session_id[:8]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception:
                pass
        with chips_col:
            cols = st.columns(len(actions))
            for col, action in zip(cols, actions):
                if col.button(action, key=f"qa_{action[:18]}", use_container_width=True):
                    return action
    else:
        cols = st.columns(len(actions))
        for col, action in zip(cols, actions):
            if col.button(action, key=f"qa_{action[:18]}", use_container_width=True):
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
            "<div style='text-align:center; color:#334155; padding-top:100px; font-size:0.9rem;'>"
            "⚠ Cannot reach the API server.<br><br>"
            "<code>uvicorn api.main:app --reload --port 8000</code>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.get("api_error"):
        err = st.session_state.pop("api_error")
        st.error(err)

    _, chat_col, _ = st.columns([1, 10, 1])

    with chat_col:
        messages = st.session_state.get("messages") or []
        astate   = st.session_state.get("analysis_state") or {}

        # ── Top bar (logo + pipeline status + new chat) ────────────────────────
        _render_topbar(session_id, astate)

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
        plot_paths   = astate.get("plot_paths") or []
        show_interactive = False

        for m in messages:
            role    = m.get("role", "assistant")
            content = m.get("content", "")
            with st.chat_message(role, avatar="🧬" if role == "assistant" else None):
                st.markdown(content)
                if role == "assistant" and m.get("has_plots") and plot_paths:
                    _render_inline_plots(session_id, plot_paths)
                    show_interactive = True

        # ── Interactive plots expander (must be OUTSIDE st.chat_message) ──────
        if show_interactive and plot_paths:
            _render_interactive_plots(session_id, plot_paths)

        # ── Quick action chips ─────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        quick_action = _render_quick_actions(session_id, astate)

        # ── File attachment ────────────────────────────────────────────────────
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
                new_sid = resp.get("session_id")
                if new_sid and new_sid != session_id:
                    st.session_state["session_id"]     = new_sid
                    st.session_state["upload_result"]  = {}
                    new_astate = {}
                else:
                    new_astate = _api_fetch_state(session_id)
                st.session_state["analysis_state"] = new_astate

                # Detect whether this response generated new plots
                old_plots = set(astate.get("plot_paths") or [])
                new_plots = set(new_astate.get("plot_paths") or [])
                has_plots = bool(new_plots - old_plots)   # True only when brand-new plots appear

                # Clear has_plots from all previous messages so plots attach
                # to exactly the message that produced them
                if has_plots:
                    for m in st.session_state["messages"]:
                        m.pop("has_plots", None)

                st.session_state["messages"].append({
                    "role":      "assistant",
                    "content":   resp["response"],
                    "has_plots": has_plots,
                })
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session()
    _ensure_session()
    _render_main()


main()
