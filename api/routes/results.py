"""
api/routes/results.py
GET  /results/{session_id}      – return full analysis state for a session
GET  /results/{session_id}/file – serve a generated output file (plots, CSVs)
"""

import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Response schemas ──────────────────────────────────────────────────────────

class TopProtein(BaseModel):
    protein: str
    logFC: float
    adj_pval: float
    direction: str


class TopPathway(BaseModel):
    pathway: str
    p_adjust: float
    gene_count: int
    source: str = "KEGG"


class AnalysisStateResponse(BaseModel):
    session_id: str
    disease_program: Optional[str] = None
    organism: Optional[str] = None

    # Data layer
    data_type: Optional[str] = None
    data_format: Optional[str] = None
    n_proteins: Optional[int] = None
    n_samples: Optional[int] = None

    # QC
    qc_passed: Optional[bool] = None
    qc_report_path: Optional[str] = None

    # Analysis
    sample_group_col: Optional[str] = None
    contrast_groups: Optional[List[str]] = None
    dea_result_path: Optional[str] = None
    top_proteins: Optional[List[TopProtein]] = None
    n_significant: Optional[int] = None

    # Enrichment
    enrichment_result_path: Optional[str] = None
    pathways: Optional[List[TopPathway]] = None

    # Output
    plot_paths: Optional[List[str]] = None
    report_path: Optional[str] = None

    status: Optional[str] = None
    error_message: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=AnalysisStateResponse)
def get_analysis_state(session_id: str):
    """Return the current analysis state for a session."""
    try:
        state = SessionManager.get_session(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    # Compute n_proteins / n_samples from data_path if not already stored
    n_proteins, n_samples = _get_data_shape(state.get("data_path"))

    top_proteins = None
    if state.get("top_proteins"):
        top_proteins = [
            TopProtein(
                protein=p["protein"],
                logFC=round(p["logFC"], 4),
                adj_pval=p["adj_pval"],
                direction=p.get("direction", "up" if p["logFC"] > 0 else "down"),
            )
            for p in state["top_proteins"]
        ]

    top_pathways = None
    if state.get("pathways"):
        top_pathways = [
            TopPathway(
                pathway=p["pathway"],
                p_adjust=p["p_adjust"],
                gene_count=p.get("gene_count", 0),
                source=p.get("source", "KEGG"),
            )
            for p in state["pathways"]
        ]

    return AnalysisStateResponse(
        session_id=session_id,
        disease_program=state.get("disease_program"),
        data_type=state.get("data_type"),
        data_format=state.get("data_format"),
        n_proteins=n_proteins,
        n_samples=n_samples,
        qc_passed=state.get("qc_passed"),
        qc_report_path=state.get("qc_report_path"),
        sample_group_col=state.get("sample_group_col"),
        contrast_groups=state.get("contrast_groups"),
        dea_result_path=state.get("dea_result_path"),
        top_proteins=top_proteins,
        n_significant=len(top_proteins) if top_proteins else None,
        enrichment_result_path=state.get("enrichment_result_path"),
        pathways=top_pathways,
        plot_paths=state.get("plot_paths"),
        report_path=state.get("report_path"),
        status=state.get("status"),
        error_message=state.get("error_message"),
    )


@router.get("/{session_id}/file")
def download_output_file(
    session_id: str,
    path: str = Query(..., description="Relative or absolute path to the output file"),
):
    """
    Serve a generated output file (plot PNG, results CSV, etc.).
    Only serves files from the outputs/ directory for safety.
    """
    file_path = Path(path)

    if not file_path.is_absolute():
        file_path = Path("outputs") / file_path

    # Security: only serve from outputs/
    try:
        file_path.resolve().relative_to(Path("outputs").resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this path is not permitted.",
        )

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {path}",
        )

    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(path=str(file_path), media_type=media_type or "application/octet-stream")


# ── Utility ───────────────────────────────────────────────────────────────────

def _get_data_shape(data_path: Optional[str]):
    if not data_path or not Path(data_path).exists():
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(data_path, index_col=0)
        return df.shape[0], df.shape[1]
    except Exception:
        return None, None
