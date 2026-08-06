"""
core/backend_service.py
Framework-agnostic service layer for the biomarker chat/upload/results flow.

Extracted from api/routes/{chat,upload,results,sessions}.py so the same
logic can be called either:
  1. From FastAPI route handlers (HTTP deployment), or
  2. Directly, in-process, from ui/app.py (single-app Streamlit deployment).

No FastAPI/Starlette types appear in this module's public functions.
"""
from __future__ import annotations

import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

from agents.ingestion_agent import IngestionAgent
from config.settings import get_settings
from core.langgraph_workflow import get_workflow
from core.session_manager import SessionManager

logger = logging.getLogger(__name__)
settings = get_settings()

_ingestion_agent = IngestionAgent()

_ALLOWED_EXT = {".csv", ".xlsx", ".xls"}
_MAX_BYTES = settings.max_file_size_mb * 1024 * 1024
_OUTPUT_BASE: Path = Path(settings.output_dir).resolve()

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


class BackendError(Exception):
    """Raised for expected, user-facing failures (bad input, missing session, etc.)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(disease_program: Optional[str] = None, organism: Optional[str] = None) -> str:
    return SessionManager.create_session(disease_program=disease_program, organism=organism)


def delete_session(session_id: str) -> None:
    try:
        SessionManager.get_session(session_id)
    except KeyError:
        raise BackendError(f"Session {session_id!r} not found", 404)
    SessionManager.delete_session(session_id)
    logger.info("Session %s deleted by client.", session_id)


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_file(
    file_bytes: bytes,
    filename: str,
    session_id: Optional[str] = None,
    disease_program: Optional[str] = None,
    organism: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist an uploaded proteomics matrix and run ingestion. Returns an
    UploadResponse-shaped dict."""
    suffix = Path(filename or "data.csv").suffix.lower()
    if suffix not in _ALLOWED_EXT:
        hint = ""
        if suffix in (".txt", ".tsv"):
            hint = " Rename to .csv if your file is tab/comma-separated."
        elif suffix in (".ods", ".xlsm", ".xlsb"):
            hint = " Please export as .xlsx from Excel/LibreOffice."
        raise BackendError(
            f"Unsupported file type '{suffix}'.{hint} Accepted formats: .csv, .xlsx, .xls.", 400,
        )

    if len(file_bytes) > _MAX_BYTES:
        raise BackendError(f"File exceeds {settings.max_file_size_mb} MB limit.", 413)

    if session_id:
        try:
            SessionManager.get_session(session_id)
        except KeyError:
            session_id = None

    if not session_id:
        session_id = SessionManager.create_session(disease_program=disease_program)
        logger.info("New session %s created for upload.", session_id)

    file_id = uuid.uuid4().hex
    raw_dir = Path(settings.data_raw_dir) / session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{file_id}{suffix}"
    raw_path.write_bytes(file_bytes)
    logger.info("Saved %s (%d bytes) -> %s", filename, len(file_bytes), raw_path)

    data_format = "excel" if suffix in (".xlsx", ".xls") else "csv"
    SessionManager.update_session(
        session_id,
        {"file_id": file_id, "data_path": str(raw_path), "data_format": data_format},
    )

    state = SessionManager.get_session(session_id)
    updated = _ingestion_agent.run(state)
    SessionManager.update_session(session_id, updated)

    if updated.get("status") == "error":
        raise BackendError(updated.get("error_message", "Ingestion failed."), 422)

    msgs = updated.get("messages") or []
    last_assistant_msg = next(
        (m.get("content") for m in reversed(msgs)
         if isinstance(m, dict) and m.get("role") == "assistant"),
        None,
    )

    g1_label = updated.get("group1_label")
    g2_label = updated.get("group2_label")
    inferred_groups: Optional[dict] = None
    if g1_label and g2_label:
        inferred_groups = {
            g1_label: updated.get("group1_samples") or [],
            g2_label: updated.get("group2_samples") or [],
        }

    return {
        "session_id": session_id,
        "file_id": file_id,
        "filename": filename or raw_path.name,
        "data_type": updated.get("data_type"),
        "data_format": updated.get("data_format"),
        "n_proteins": updated.get("n_proteins"),
        "n_samples": updated.get("n_samples"),
        "sample_columns": updated.get("sample_columns"),
        "metadata_columns": updated.get("metadata_columns"),
        "is_pooled_design": bool(updated.get("is_pooled_design", False)),
        "label_map": updated.get("label_map"),
        "inferred_groups": inferred_groups,
        "message": last_assistant_msg,
        "status": updated.get("status", "data_loaded"),
    }


# ── Chat / workflow ───────────────────────────────────────────────────────────

def _flatten_plot_paths(plot_paths: Any) -> List[str]:
    if not plot_paths:
        return []
    out: List[str] = []
    seen: set = set()

    def _push(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        if value in seen:
            return
        seen.add(value)
        out.append(value)

    if isinstance(plot_paths, list):
        for v in plot_paths:
            if isinstance(v, str):
                _push(v)
            elif isinstance(v, dict):
                for inner in v.values():
                    _push(inner)
    elif isinstance(plot_paths, dict):
        for v in plot_paths.values():
            if isinstance(v, str):
                _push(v)
            elif isinstance(v, dict):
                for inner in v.values():
                    _push(inner)
            elif isinstance(v, list):
                for inner in v:
                    _push(inner)
    return out


def _humanise_plot_title(path: str) -> str:
    stem = Path(path).stem
    stem = stem.replace("plot_", "").replace("_plot", "")
    return stem.replace("_", " ").strip().title() or "Plot"


def build_plot_artifacts(session_id: str, paths: List[str]) -> List[Dict[str, Any]]:
    """Convert a list of plot file paths into PlotArtifact dicts (used by the
    SSE path in api/routes/chat.py; harmless to call in-process too)."""
    if not paths:
        return []

    base_url = f"/results/{session_id}/file?path="
    by_stem: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for raw in paths:
        p = Path(raw)
        suffix = p.suffix.lower()
        try:
            rel = str(p.resolve().relative_to(Path("outputs").resolve()))
            file_param = f"outputs/{rel}"
        except (ValueError, FileNotFoundError, RuntimeError):
            file_param = str(p)

        stem_key = p.stem
        if stem_key not in by_stem:
            order.append(stem_key)
            by_stem[stem_key] = {
                "id": f"plot-{session_id[:8]}-{stem_key}",
                "kind": "plot",
                "title": _humanise_plot_title(raw),
                "createdAt": now,
            }
        entry = by_stem[stem_key]
        full_url = base_url + quote(file_param, safe="/:?=&")
        if suffix == ".html":
            entry["htmlUrl"] = full_url
        elif suffix in _IMAGE_SUFFIXES:
            entry["imageUrl"] = full_url
        else:
            entry.setdefault("imageUrl", full_url)

    return [by_stem[k] for k in order]


def run_chat_turn(
    session_id: str,
    message: str,
    *,
    sample_group_col: Optional[str] = None,
    contrast_groups: Optional[List[str]] = None,
    disease_program: Optional[str] = None,
    organism: Optional[str] = None,
    group1_samples: Optional[List[str]] = None,
    group2_samples: Optional[List[str]] = None,
    group1_label: Optional[str] = None,
    group2_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one LangGraph turn for a session. Returns
    ``{session_id, new_assistant_messages, intent, status, expired, new_plot_paths}``.
    """
    try:
        state = SessionManager.get_session(session_id)
        expired = False
    except KeyError:
        logger.warning("Session '%s' not found - creating replacement.", session_id)
        new_sid = SessionManager.create_session(
            disease_program=disease_program,
            organism=organism,
        )
        state = SessionManager.get_session(new_sid)
        state["session_id"] = new_sid
        expired_msg = (
            "⚠️ **Session expired** — the server was restarted and in-memory "
            "session data was lost. Please re-upload your data file to continue."
        )
        state["messages"].append({"role": "assistant", "content": expired_msg})
        SessionManager.update_session(new_sid, state)
        return {
            "session_id": new_sid,
            "new_assistant_messages": [{"role": "assistant", "content": expired_msg}],
            "intent": None,
            "status": "session_expired",
            "expired": True,
            "new_plot_paths": [],
        }

    overrides: dict = {"user_query": message}
    if sample_group_col: overrides["sample_group_col"] = sample_group_col
    if contrast_groups:  overrides["contrast_groups"]  = contrast_groups
    if disease_program:  overrides["disease_program"]  = disease_program
    if organism:         overrides["organism"]         = organism
    if group1_samples:   overrides["group1_samples"]   = group1_samples
    if group2_samples:   overrides["group2_samples"]   = group2_samples
    if group1_label:     overrides["group1_label"]     = group1_label
    if group2_label:     overrides["group2_label"]     = group2_label
    state.update(overrides)

    # Decouple messages list — passing the live list causes LangGraph's
    # add_messages reducer to double every message.
    state["messages"] = list(state.get("messages") or [])
    n_msgs_before = len(state["messages"])
    plots_before = set(_flatten_plot_paths(state.get("plot_paths")))

    workflow = get_workflow()
    try:
        updated_state = workflow.invoke(state)
    except Exception as exc:
        logger.exception("Workflow error session=%s: %s", session_id, exc)
        raise BackendError(f"Analysis pipeline error: {exc}", 500)

    delta_msgs = (updated_state.get("messages") or [])[n_msgs_before:]
    state_to_store = {**updated_state, "messages": delta_msgs}
    SessionManager.update_session(updated_state["session_id"], state_to_store)

    plots_after = _flatten_plot_paths(updated_state.get("plot_paths"))
    new_plot_paths = [p for p in plots_after if p not in plots_before]

    new_assistant_messages: List[dict] = []
    for m in delta_msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            new_assistant_messages.append(m)
        elif hasattr(m, "content"):
            msg_type = getattr(m, "type", "") or type(m).__name__.lower()
            if "ai" in msg_type or "assistant" in msg_type:
                new_assistant_messages.append({
                    "role": "assistant",
                    "content": str(m.content),
                })

    return {
        "session_id": updated_state["session_id"],
        "new_assistant_messages": new_assistant_messages,
        "intent": updated_state.get("intent"),
        "status": updated_state.get("status", "ok"),
        "expired": False,
        "new_plot_paths": new_plot_paths,
    }


# ── Results / state ───────────────────────────────────────────────────────────

def get_analysis_state(session_id: str) -> Dict[str, Any]:
    try:
        state = SessionManager.get_session(session_id)
    except KeyError:
        raise BackendError(f"Session '{session_id}' not found.", 404)

    qc = state.get("qc_summary") or {}
    all_plots = list(state.get("plot_paths") or [])
    for p in (qc.get("plot_paths") or []):
        if p and p not in all_plots:
            all_plots.append(p)

    return {
        "session_id": session_id,
        "disease_program": state.get("disease_program"),
        "data_type": state.get("data_type"),
        "data_format": state.get("data_format"),
        "n_proteins": state.get("n_proteins"),
        "n_samples": state.get("n_samples"),
        "omic_type": state.get("omic_type"),
        "sample_columns": state.get("sample_columns"),
        "metadata_columns": state.get("metadata_columns"),
        "group1_label": state.get("group1_label"),
        "group2_label": state.get("group2_label"),
        "group1_samples": state.get("group1_samples"),
        "group2_samples": state.get("group2_samples"),
        "analysis_mode": state.get("analysis_mode"),
        "qc_passed": state.get("qc_passed"),
        "qc_summary": qc,
        "n_significant": state.get("n_significant"),
        "top_biomarkers": state.get("top_biomarkers"),
        "excel_path": state.get("excel_path"),
        "analysis_summary": state.get("analysis_summary"),
        "plot_paths": all_plots or None,
        "pathways": state.get("pathways"),
        "enrichment_result_path": state.get("enrichment_result_path"),
        "status": state.get("status"),
        "error_message": state.get("error_message"),
    }


def resolve_output_path(path: str) -> Path:
    """Resolve a stored (possibly 'outputs/'-prefixed) path to an absolute
    path inside the configured output directory. Raises BackendError on
    path-traversal attempts or missing files."""
    candidate = Path(path)
    if candidate.is_absolute():
        pass
    elif candidate.parts and candidate.parts[0] == "outputs":
        candidate = _OUTPUT_BASE.joinpath(*candidate.parts[1:])
    else:
        candidate = _OUTPUT_BASE / candidate

    try:
        resolved = candidate.resolve()
        resolved.relative_to(_OUTPUT_BASE)
    except ValueError:
        raise BackendError("Access denied.", 403)

    if not resolved.exists():
        raise BackendError(f"File not found: {path}", 404)
    return resolved


def get_download_payload(session_id: str) -> Dict[str, Any]:
    """Return the best available results download as
    ``{content, filename, mime}`` (enrichment CSV if present, else biomarker Excel)."""
    try:
        state = SessionManager.get_session(session_id)
    except KeyError:
        raise BackendError(f"Session '{session_id}' not found.", 404)

    enrichment_path = state.get("enrichment_result_path")
    if enrichment_path and Path(enrichment_path).exists():
        p = Path(enrichment_path)
        return {
            "content": p.read_bytes(),
            "filename": f"enrichment_{session_id[:8]}.csv",
            "mime": "text/csv",
        }

    excel_path = state.get("excel_path")
    if not excel_path or not Path(excel_path).exists():
        raise BackendError("No Excel file available. Run the analysis first.", 404)

    p = Path(excel_path)
    return {
        "content": p.read_bytes(),
        "filename": f"biomarkers_{session_id[:8]}.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
