"""
api/routes/upload.py
POST /upload/
Accepts a proteomics CSV or Excel file, persists it, runs the IngestionAgent
to validate & normalise, and returns dataset metadata.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from agents.ingestion_agent import IngestionAgent
from config.settings import get_settings
from core.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

_ingestion_agent = IngestionAgent()

# ── Schemas ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    file_id: str
    filename: str
    data_type: Optional[str] = None
    data_format: Optional[str] = None
    n_proteins: Optional[int] = None
    n_samples: Optional[int] = None
    status: str


# ── Helpers ───────────────────────────────────────────────────────────────────

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


def _validate_extension(filename: str) -> str:
    """Return lower-case extension or raise 400."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Upload CSV or Excel (.xlsx/.xls).",
        )
    return suffix


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_proteomics_file(
    file: UploadFile = File(..., description="Proteomics intensity matrix (CSV or Excel)"),
    session_id: Optional[str] = Form(None, description="Existing session ID (optional)"),
    disease_program: Optional[str] = Form("FA"),
    organism: Optional[str] = Form("human"),
):
    """
    Upload a proteomics data file.

    - Rows = proteins/features, columns = samples  (or transposed — auto-detected)
    - Returns dataset metadata (n_proteins, n_samples, data_type)
    - Creates a new session if session_id is not provided
    """
    suffix = _validate_extension(file.filename or "file.csv")

    # Read and size-check
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.max_file_size_mb} MB.",
        )

    # Resolve or create session
    if session_id:
        try:
            SessionManager.get_session(session_id)
        except KeyError:
            session_id = None

    if not session_id:
        session_id = SessionManager.create_session(disease_program=disease_program)
        logger.info("Created new session %s for upload.", session_id)

    # Persist raw file
    file_id = uuid.uuid4().hex
    raw_dir = Path(settings.data_raw_dir) / session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{file_id}{suffix}"
    raw_path = raw_dir / safe_name

    raw_path.write_bytes(content)
    logger.info("Saved upload → %s (%d bytes)", raw_path, len(content))

    # Update session state with file info
    data_format = "excel" if suffix in (".xlsx", ".xls") else "csv"
    SessionManager.update_session(
        session_id,
        {
            "file_id": file_id,
            "data_path": str(raw_path),
            "data_format": data_format,
        },
    )

    # Run ingestion agent (validate & normalise)
    state = SessionManager.get_session(session_id)
    updated = _ingestion_agent.run(state)
    SessionManager.update_session(session_id, updated)

    if updated.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=updated.get("error_message", "Data ingestion failed."),
        )

    return UploadResponse(
        session_id=session_id,
        file_id=file_id,
        filename=file.filename or safe_name,
        data_type=updated.get("data_type"),
        data_format=updated.get("data_format"),
        n_proteins=updated.get("n_proteins"),
        n_samples=updated.get("n_samples"),
        status=updated.get("status", "data_loaded"),
    )
