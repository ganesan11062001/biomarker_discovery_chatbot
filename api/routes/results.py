"""
api/routes/results.py
GET  /results/{session_id}          – full analysis state
GET  /results/{session_id}/excel    – download formatted Excel file
GET  /results/{session_id}/file     – serve any generated output file
"""
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.settings import get_settings
from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

# Resolve once at import so path-traversal checks are CWD-independent.
_OUTPUT_BASE: Path = Path(settings.output_dir).resolve()


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
        state = SessionManager.get_session(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    # Collect plot paths from BiomarkerAgent (inside qc_summary) + VisualizationAgent
    qc = state.get("qc_summary") or {}
    all_plots = list(state.get("plot_paths") or [])
    for p in (qc.get("plot_paths") or []):
        if p and p not in all_plots:
            all_plots.append(p)

    return AnalysisStateResponse(
        session_id=session_id,
        disease_program=state.get("disease_program"),
        data_type=state.get("data_type"),
        data_format=state.get("data_format"),
        n_proteins=state.get("n_proteins"),
        n_samples=state.get("n_samples"),
        omic_type=state.get("omic_type"),
        sample_columns=state.get("sample_columns"),
        metadata_columns=state.get("metadata_columns"),
        group1_label=state.get("group1_label"),
        group2_label=state.get("group2_label"),
        group1_samples=state.get("group1_samples"),
        group2_samples=state.get("group2_samples"),
        analysis_mode=state.get("analysis_mode"),
        qc_passed=state.get("qc_passed"),
        qc_summary=qc,
        n_significant=state.get("n_significant"),
        top_biomarkers=state.get("top_biomarkers"),
        excel_path=state.get("excel_path"),
        analysis_summary=state.get("analysis_summary"),
        plot_paths=all_plots or None,
        pathways=state.get("pathways"),
        enrichment_result_path=state.get("enrichment_result_path"),
        status=state.get("status"),
        error_message=state.get("error_message"),
    )


@router.get("/{session_id}/excel")
def download_excel(session_id: str):
    """Download the latest results file (enrichment CSV if available, else biomarker Excel)."""
    try:
        state = SessionManager.get_session(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    enrichment_path = state.get("enrichment_result_path")
    if enrichment_path and Path(enrichment_path).exists():
        return FileResponse(
            path=enrichment_path,
            media_type="text/csv",
            filename=f"enrichment_{session_id[:8]}.csv",
        )

    excel_path = state.get("excel_path")
    if not excel_path or not Path(excel_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Excel file available. Run the analysis first.",
        )

    return FileResponse(
        path=excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"biomarkers_{session_id[:8]}.xlsx",
    )


@router.get("/{session_id}/file")
def download_output_file(
    session_id: str,
    path: str = Query(..., description="Relative path inside outputs/"),
):
    """Serve any generated output file (plots, CSVs, etc.)."""
    candidate = Path(path)
    if candidate.is_absolute():
        # Absolute path — still validated against _OUTPUT_BASE below
        pass
    elif candidate.parts and candidate.parts[0] == "outputs":
        # Stored paths typically include the "outputs/" prefix; strip it and
        # re-anchor under the resolved output base.
        candidate = _OUTPUT_BASE.joinpath(*candidate.parts[1:])
    else:
        candidate = _OUTPUT_BASE / candidate

    # Security: resolved path must be inside the configured output directory.
    # Use the pre-resolved _OUTPUT_BASE so this is independent of CWD.
    try:
        resolved = candidate.resolve()
        resolved.relative_to(_OUTPUT_BASE)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    media_type, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(path=str(resolved), media_type=media_type or "application/octet-stream")
