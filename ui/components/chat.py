"""
ui/components/chat.py
Reusable chat message rendering helpers for Streamlit.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


# ─── Message bubble renderer ──────────────────────────────────────────────────

def render_messages(messages: list[dict[str, Any]]) -> None:
    """Render the full message history using st.chat_message."""
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role, avatar="🧬" if role == "assistant" else "👤"):
            st.markdown(content)


def render_welcome() -> None:
    """Show welcome message when there are no messages yet."""
    with st.chat_message("assistant", avatar="🧬"):
        st.markdown(
            "**Welcome to the Proteomics Biomarker Discovery Platform.**\n\n"
            "I can help you:\n"
            "- **Load** your CSV or Excel proteomics data (Olink NPX, MS, or generic)\n"
            "- **Run QC** — missing value filter, CV cutoff, outlier detection\n"
            "- **Differential expression** — limma, DEP, or MSstats\n"
            "- **Pathway enrichment** — KEGG and GO via clusterProfiler\n"
            "- **Generate reports** — volcano plots, heatmaps, ranking tables\n\n"
            "**Start by uploading your data in the sidebar**, then describe what you'd like to do."
        )


# ─── Chat input helper ────────────────────────────────────────────────────────

def render_chat_input(placeholder: str = "Ask about your proteomics data …") -> str | None:
    """
    Render a chat input box.
    Returns the submitted text, or None if nothing submitted yet.
    """
    return st.chat_input(placeholder)


# ─── Suggested prompts ────────────────────────────────────────────────────────

def render_suggested_prompts(data_loaded: bool, qc_done: bool, dea_done: bool) -> str | None:
    """
    Show clickable suggested prompts appropriate to the current pipeline stage.
    Returns the selected prompt text, or None.
    """
    suggestions: list[str] = []

    if not data_loaded:
        suggestions = [
            "Load my data and run quality control",
        ]
    elif not qc_done:
        suggestions = [
            "Run quality control on my data",
            "What QC filters do you apply?",
        ]
    elif not dea_done:
        suggestions = [
            "Run differential expression analysis",
            "Compare Disease vs Control groups",
        ]
    else:
        suggestions = [
            "Run pathway enrichment analysis",
            "Generate a volcano plot",
            "Show me the top biomarkers",
            "Create a full analysis report",
        ]

    if not suggestions:
        return None

    st.caption("💡 Try:")
    cols = st.columns(len(suggestions))
    for col, prompt in zip(cols, suggestions):
        if col.button(prompt, use_container_width=True, key=f"suggest_{prompt[:20]}"):
            return prompt

    return None
