import uuid
from typing import Dict
from core.state import BiomarkerState

class SessionManager:
    _sessions: Dict[str, BiomarkerState] = {}

    @classmethod
    def create_session(cls, disease_program: str = None) -> str:
        session_id = str(uuid.uuid4())
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
            # Analysis config
            sample_group_col=None,
            contrast_groups=None,
            disease_program=disease_program,
            # QC
            qc_report_path=None,
            qc_passed=None,
            # DEA
            dea_result_path=None,
            top_proteins=None,
            # Enrichment
            enrichment_result_path=None,
            pathways=None,
            # Output
            plot_paths=None,
            report_path=None,
            status="ready",
            error_message=None
        )
        return session_id

    @classmethod
    def get_session(cls, session_id: str) -> BiomarkerState:
        if session_id not in cls._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        return cls._sessions[session_id]

    @classmethod
    def update_session(cls, session_id: str, updates: dict):
        if session_id not in cls._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        cls._sessions[session_id].update(updates)

    @classmethod
    def delete_session(cls, session_id: str):
        cls._sessions.pop(session_id, None)
