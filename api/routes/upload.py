"""
api/routes/upload.py
POST /upload/
Accepts a proteomics CSV or Excel file, runs DataLoadingSkill,
and returns dataset metadata including detected sample columns.
"""
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from agents.ingestion_agent import IngestionAgent
from config.settings import get_settings
from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

_ingestion_agent = IngestionAgent()

_ALLOWED_EXT = {".csv", ".xlsx", ".xls"}
_MAX_BYTES   = settings.max_file_size_mb * 1024 * 1024


# ── Schema ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id:       str
    file_id:          str
    filename:         str
    data_type:        Optional[str]             = None
    data_format:      Optional[str]             = None
    n_proteins:       Optional[int]             = None
    n_samples:        Optional[int]             = None
    sample_columns:   Optional[List[str]]       = None
    metadata_columns: Optional[List[str]]       = None
    is_pooled_design: bool                      = False
    label_map:        Optional[dict]            = None
    status:           str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_proteomics_file(
    file:             UploadFile       = File(...),
    session_id:       Optional[str]    = Form(None),
    disease_program:  Optional[str]    = Form("FA"),
    organism:         Optional[str]    = Form("human"),
):
    """
    Upload a proteomics matrix (CSV / Excel).
    Returns dataset shape and the list of detected sample columns
    so the client can present a group-assignment UI.
    """
    suffix = Path(file.filename or "data.csv").suffix.lower()
    if suffix not in _ALLOWED_EXT:
        hint = ""
        if suffix in (".txt", ".tsv"):
            hint = " Rename to .csv if your file is tab/comma-separated."
        elif suffix in (".ods", ".xlsm", ".xlsb"):
            hint = " Please export as .xlsx from Excel/LibreOffice."
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'.{hint} Accepted formats: .csv, .xlsx, .xls.",
        )

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb} MB limit.")

    # Resolve or create session
    if session_id:
        try:
            SessionManager.get_session(session_id)
        except KeyError:
            session_id = None

    if not session_id:
        session_id = SessionManager.create_session(disease_program=disease_program)
        logger.info("New session %s created for upload.", session_id)

    # Persist raw file
    file_id  = uuid.uuid4().hex
    raw_dir  = Path(settings.data_raw_dir) / session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{file_id}{suffix}"
    raw_path.write_bytes(content)
    logger.info("Saved %s (%d bytes) → %s", file.filename, len(content), raw_path)

    data_format = "excel" if suffix in (".xlsx", ".xls") else "csv"
    SessionManager.update_session(
        session_id,
        {"file_id": file_id, "data_path": str(raw_path), "data_format": data_format},
    )

    # Run ingestion agent
    state   = SessionManager.get_session(session_id)
    updated = _ingestion_agent.run(state)
    SessionManager.update_session(session_id, updated)

    if updated.get("status") == "error":
        raise HTTPException(422, updated.get("error_message", "Ingestion failed."))

    return UploadResponse(
        session_id       = session_id,
        file_id          = file_id,
        filename         = file.filename or raw_path.name,
        data_type        = updated.get("data_type"),
        data_format      = updated.get("data_format"),
        n_proteins       = updated.get("n_proteins"),
        n_samples        = updated.get("n_samples"),
        sample_columns   = updated.get("sample_columns"),
        metadata_columns = updated.get("metadata_columns"),
        is_pooled_design = bool(updated.get("is_pooled_design", False)),
        label_map        = updated.get("label_map"),
        status           = updated.get("status", "data_loaded"),
    )
