"""
api/routes/results.py
GET  /results/{session_id}          – full analysis state
GET  /results/{session_id}/excel    – download formatted Excel file
GET  /results/{session_id}/file     – serve any generated output file
"""
import logging
import mimetypes
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from core.backend_service import BackendError
from core.backend_service import get_analysis_state as _get_analysis_state
from core.backend_service import get_download_payload as _get_download_payload
from core.backend_service import resolve_output_path as _resolve_output_path

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QCSummary(BaseModel):
    proteins_input:    Optional[int]   = None
    proteins_after_qc: Optional[int]   = None
    proteins_removed:  Optional[int]   = None
    samples_input:     Optional[int]   = None
    samples_after_qc:  Optional[int]   = None
    log2_transformed:  Optional[bool]  = None
    missing_threshold: Optional[float] = None


class TopBiomarker(BaseModel):
    rank:             Optional[int]   = None
    protein:          str
    log2_fold_change: Optional[float] = None
    p_value:          Optional[float] = None
    adj_p_value:      Optional[float] = None
    significance:     Optional[str]   = None
    cv_percent:       Optional[float] = None


class AnalysisStateResponse(BaseModel):
    session_id:    str
    disease_program: Optional[str] = None

    # Data layer
    data_type:    Optional[str] = None
    data_format:  Optional[str] = None
    n_proteins:   Optional[int] = None
    n_samples:    Optional[int] = None
    omic_type:    Optional[str] = None
    sample_columns:   Optional[List[str]] = None
    metadata_columns: Optional[List[str]] = None

    # Groups
    group1_label:   Optional[str]       = None
    group2_label:   Optional[str]       = None
    group1_samples: Optional[List[str]] = None
    group2_samples: Optional[List[str]] = None
    analysis_mode:  Optional[str]       = None

    # QC
    qc_passed:    Optional[bool]        = None
    qc_summary:   Optional[Dict]        = None

    # Results
    n_significant:   Optional[int]  = None
    top_biomarkers:  Optional[List[Dict[str, Any]]] = None
    excel_path:      Optional[str]  = None
    analysis_summary: Optional[str] = None

    # Plots — collected from both BiomarkerAgent (qc_summary) and VisualizationAgent
    plot_paths:   Optional[List[str]] = None

    # Enrichment
    pathways:               Optional[List[Dict[str, Any]]] = None
    enrichment_result_path: Optional[str] = None

    status:         Optional[str] = None
    error_message:  Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=AnalysisStateResponse)
def get_analysis_state(session_id: str):
    """Return the current analysis state for a session."""
    try:
        return AnalysisStateResponse(**_get_analysis_state(session_id))
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("/{session_id}/excel")
def download_excel(session_id: str):
    """Download the latest results file (enrichment CSV if available, else biomarker Excel)."""
    try:
        payload = _get_download_payload(session_id)
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    return Response(
        content=payload["content"],
        media_type=payload["mime"],
        headers={"Content-Disposition": f'attachment; filename="{payload["filename"]}"'},
    )


@router.get("/{session_id}/file")
def download_output_file(
    session_id: str,
    path: str = Query(..., description="Relative path inside outputs/"),
):
    """Serve any generated output file (plots, CSVs, etc.)."""
    try:
        resolved = _resolve_output_path(path)
    except BackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    media_type, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(path=str(resolved), media_type=media_type or "application/octet-stream")
