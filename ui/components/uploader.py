"""
ui/components/uploader.py
File upload component — renders the sidebar uploader and sends the file
to the FastAPI /upload/ endpoint.
"""
from __future__ import annotations

import io
from typing import Any

import requests
import streamlit as st


def render_uploader(
    api_base: str,
    session_id: str,
    disease_program: str = "FA",
    organism: str = "human",
    widget_key: str = "proteomics_file_uploader",
) -> dict[str, Any] | None:
    """
    Render the proteomics file uploader widget.

    Returns the upload response dict on success, or None if no file has been
    uploaded yet.  The caller is responsible for updating st.session_state.
    """
    uploaded_file = st.file_uploader(
        "Proteomics data (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        help=(
            "Upload your proteomics intensity matrix.\n"
            "- **Rows** = proteins / features\n"
            "- **Columns** = samples\n"
            "Supported: Olink NPX, MS label-free (LFQ), TMT, or any generic matrix."
        ),
        key=widget_key,
    )

    if uploaded_file is None:
        return None

    # Only re-upload if the file name changed (avoid re-uploading on every rerun)
    if st.session_state.get("_last_uploaded_file") == uploaded_file.name:
        return st.session_state.get("_upload_result")

    with st.spinner(f"Uploading **{uploaded_file.name}** …"):
        try:
            response = requests.post(
                f"{api_base}/upload/",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
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

    if response.status_code in (200, 201):
        result = response.json()
        st.session_state["_last_uploaded_file"] = uploaded_file.name
        st.session_state["_upload_result"] = result
        return result
    else:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        st.error(f"Upload failed ({response.status_code}): {detail}")
        return None


def render_upload_success(result: dict[str, Any]) -> None:
    """Render a concise success card after a successful upload."""
    st.success("Data loaded successfully")
    cols = st.columns(2)
    cols[0].metric("Proteins", result.get("n_proteins", "–"))
    cols[1].metric("Samples", result.get("n_samples", "–"))
    st.caption(
        f"Type: **{result.get('data_type', 'unknown')}** · "
        f"Format: **{result.get('data_format', '').upper()}**"
    )
