import uuid
import threading
from typing import Dict, Optional
from core.state import BiomarkerState


class SessionManager:
    """Thread-safe in-memory session store."""

    _sessions: Dict[str, BiomarkerState] = {}
    _lock = threading.RLock()

    @classmethod
    def create_session(cls, disease_program: str = "FA", organism: str = "human") -> str:
        session_id = str(uuid.uuid4())
        with cls._lock:
            cls._sessions[session_id] = BiomarkerState(
                messages=[],
                session_id=session_id,
                user_query="",
                intent=None,
                active_agent=None,
                # Data ingestion
                file_id=None,
                data_path=None,
                data_type=None,
                data_format=None,
                n_proteins=None,
                n_samples=None,
                sample_columns=None,
                metadata_columns=None,
                # Analysis config
                sample_group_col=None,
                contrast_groups=None,
                disease_program=disease_program,
                group1_samples=None,
                group2_samples=None,
                group1_label=None,
                group2_label=None,
                analysis_mode=None,
                # QC
                qc_report_path=None,
                qc_passed=None,
                qc_summary=None,
                # Analysis results
                dea_result_path=None,
                top_proteins=None,
                top_biomarkers=None,
                n_significant=None,
                excel_path=None,
                # Enrichment
                enrichment_result_path=None,
                pathways=None,
                # Output
                plot_paths=None,
                report_path=None,
                analysis_summary=None,
                status="ready",
                error_message=None,
            )
        return session_id

    @classmethod
    def get_session(cls, session_id: str) -> BiomarkerState:
        with cls._lock:
            if session_id not in cls._sessions:
                raise KeyError(f"Session '{session_id}' not found.")
            return cls._sessions[session_id]

    @classmethod
    def update_session(cls, session_id: str, updates: dict) -> None:
        with cls._lock:
            if session_id not in cls._sessions:
                raise KeyError(f"Session '{session_id}' not found.")
            cls._sessions[session_id].update(updates)

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(session_id, None)

    @classmethod
    def list_sessions(cls) -> list:
        with cls._lock:
            return list(cls._sessions.keys())
