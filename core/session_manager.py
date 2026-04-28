"""
core/session_manager.py
Thread-safe in-memory session store.

Every field defined in BiomarkerState is initialised so agents can
do state.get("field") safely without KeyError or unexpected None.
"""
import threading
import time
import uuid
from typing import Dict, Optional

from core.state import BiomarkerState

_SESSION_TTL_SECONDS = 4 * 60 * 60   # 4-hour expiry (idle)


class SessionManager:
    _sessions:       Dict[str, BiomarkerState] = {}
    _last_accessed:  Dict[str, float]          = {}
    _lock = threading.RLock()

    # ── Create ────────────────────────────────────────────────────────────────

    @classmethod
    def create_session(
        cls,
        disease_program: str = "General",
        organism: str = "human",
    ) -> str:
        session_id = str(uuid.uuid4())
        state = BiomarkerState(
            # ── LangGraph ────────────────────────────────────────────────────
            messages       = [],
            # ── Session ──────────────────────────────────────────────────────
            session_id     = session_id,
            user_query     = "",
            intent         = None,
            active_agent   = None,
            # ── Omic routing ─────────────────────────────────────────────────
            omic_type      = None,
            # ── Data ingestion ────────────────────────────────────────────────
            file_id              = None,
            raw_data_path        = None,
            data_path            = None,
            data_type            = None,
            data_format          = None,
            n_proteins           = None,
            n_samples            = None,
            sample_columns       = None,
            metadata_columns     = None,
            label_map            = None,
            is_pooled_design     = False,
            identifier_info      = None,
            all_sheets           = None,
            # ── Analysis config ───────────────────────────────────────────────
            disease_program      = disease_program,
            organism             = organism,
            sample_group_col     = None,
            contrast_groups      = None,
            group1_samples       = None,
            group2_samples       = None,
            group1_label         = None,
            group2_label         = None,
            analysis_mode        = None,
            # ── QC ────────────────────────────────────────────────────────────
            qc_passed            = None,
            qc_summary           = None,
            # ── Analysis results ──────────────────────────────────────────────
            top_biomarkers       = None,
            top_proteins         = None,
            n_significant        = None,
            excel_path           = None,
            analysis_summary     = None,
            analysis_code        = None,
            dea_result_path      = None,
            # ── Enrichment ────────────────────────────────────────────────────
            enrichment_result_path = None,
            pathways               = None,
            # ── Visualisation ─────────────────────────────────────────────────
            plot_paths           = None,
            report_path          = None,
            # ── Status ────────────────────────────────────────────────────────
            status               = "ready",
            error_message        = None,
        )
        with cls._lock:
            cls._sessions[session_id]      = state
            cls._last_accessed[session_id] = time.time()
        return session_id

    # ── Read ──────────────────────────────────────────────────────────────────

    @classmethod
    def get_session(cls, session_id: str) -> BiomarkerState:
        with cls._lock:
            if session_id not in cls._sessions:
                raise KeyError(f"Session '{session_id}' not found.")
            cls._last_accessed[session_id] = time.time()
            return cls._sessions[session_id]

    # ── Update ────────────────────────────────────────────────────────────────

    @classmethod
    def update_session(cls, session_id: str, updates: dict) -> None:
        with cls._lock:
            if session_id not in cls._sessions:
                raise KeyError(f"Session '{session_id}' not found.")
            # Merge: for messages use extend so we never lose history
            existing = cls._sessions[session_id]
            new_msgs = updates.get("messages")
            if new_msgs is not None:
                existing_msgs = list(existing.get("messages") or [])
                # Normalise to plain dicts before storing
                for m in new_msgs:
                    plain = _to_plain_dict(m)
                    if plain:
                        existing_msgs.append(plain)
                updates = {k: v for k, v in updates.items() if k != "messages"}
                existing.update(updates)
                existing["messages"] = existing_msgs
            else:
                existing.update(updates)
            cls._last_accessed[session_id] = time.time()

    # ── Delete ────────────────────────────────────────────────────────────────

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(session_id, None)
            cls._last_accessed.pop(session_id, None)

    # ── Housekeeping ──────────────────────────────────────────────────────────

    @classmethod
    def expire_old_sessions(cls) -> int:
        """Remove sessions idle longer than TTL. Returns count removed."""
        cutoff = time.time() - _SESSION_TTL_SECONDS
        to_remove = []
        with cls._lock:
            for sid, last in cls._last_accessed.items():
                if last < cutoff:
                    to_remove.append(sid)
            for sid in to_remove:
                cls._sessions.pop(sid, None)
                cls._last_accessed.pop(sid, None)
        return len(to_remove)

    @classmethod
    def list_sessions(cls) -> list:
        with cls._lock:
            return list(cls._sessions.keys())

    @classmethod
    def session_count(cls) -> int:
        with cls._lock:
            return len(cls._sessions)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_plain_dict(msg) -> Optional[dict]:
    """Normalise a LangChain message object or plain dict to {role, content, ...extras}."""
    if isinstance(msg, dict):
        role    = msg.get("role") or msg.get("type", "assistant")
        content = msg.get("content", "")
        # Preserve any extra metadata keys (e.g. has_plots)
        extras  = {k: v for k, v in msg.items() if k not in ("role", "type", "content")}
        return {"role": role, "content": str(content), **extras}
    # LangChain BaseMessage duck-type
    if hasattr(msg, "content"):
        msg_type = getattr(msg, "type", "") or type(msg).__name__.lower()
        role = "assistant" if "ai" in msg_type or "assistant" in msg_type else "user"
        return {"role": role, "content": str(msg.content)}
    return None
