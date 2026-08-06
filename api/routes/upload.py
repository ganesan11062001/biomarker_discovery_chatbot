"""
api/routes/upload.py
POST /upload/
Accepts a proteomics CSV or Excel file, runs DataLoadingSkill,
and returns dataset metadata including detected sample columns.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from config.settings import get_settings
from core.backend_service import BackendError
from core.backend_service import upload_file as _upload_file

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


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
    inferred_groups:  Optional[dict]            = None
    message:          Optional[str]             = None
    status:           str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_proteomics_file(
    request:          Request,
    file:             UploadFile       = File(...),
    session_id:       Optional[str]    = Form(None),
    disease_program:  Optional[str]    = Form(None),
    organism:         Optional[str]    = Form(None),
):
    """
    Upload a proteomics matrix (CSV / Excel).
    Returns dataset shape and the list of detected sample columns
    so the client can present a group-assignment UI.
    """
    # Reject oversized requests before reading the body into memory.
    # Content-Length is advisory (clients can omit it) but the hard size
    # gate inside upload_file() after read() catches the rest.
    content_length = request.headers.get("content-length")
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb} MB limit.")

    content = await file.read()

    try:
        result = _upload_file(
            content,
            file.filename or "data.csv",
            session_id=session_id,
            disease_program=disease_program,
            organism=organism,
        )
    except BackendError as exc:
        raise HTTPException(exc.status_code, exc.message)

    return UploadResponse(**result)

