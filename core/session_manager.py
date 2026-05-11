"""
core/session_manager.py
Thread-safe session store with optional disk checkpointing.

Every field defined in BiomarkerState is initialised so agents can
do state.get("field") safely without KeyError or unexpected None.

Disk checkpoints:
  After every update_session() and create_session() the state is also
  serialised to JSON under data/sessions/<session_id>.json. On process
  start, _load_from_disk() rehydrates all non-expired sessions so the
  user can resume after a server restart.
"""
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from core.state import BiomarkerState

logger = logging.getLogger(__name__)

_SESSION_TTL_SECONDS = 4 * 60 * 60   # 4-hour expiry (idle)
_CHECKPOINT_DIR      = Path(os.environ.get("SESSION_CHECKPOINT_DIR",
                                            "data/sessions"))


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert state values into JSON-serialisable primitives.

    DataFrames are dropped from the on-disk checkpoint (they're large and
    can be reloaded from the source CSV/Excel). Everything else is rendered
    via best-effort conversion.
    """
    # Lazy pandas import — keep core import cheap
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            # Skip DataFrames — they reload from the file via load_data
            return f"<DataFrame {obj.shape[0]}x{obj.shape[1]}>"
        if isinstance(obj, pd.Series):
            return obj.to_list()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy scalars / other types — coerce via str
    return str(obj)


class SessionManager:
    _sessions:       Dict[str, BiomarkerState] = {}
    _last_accessed:  Dict[str, float]          = {}
    _lock = threading.RLock()
    _checkpoint_enabled = True

    # ── Disk checkpoint helpers ───────────────────────────────────────────────

    @classmethod
    def _checkpoint_path(cls, session_id: str) -> Path:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        return _CHECKPOINT_DIR / f"{session_id}.json"

    @classmethod
    def _save_to_disk(cls, session_id: str) -> None:
        if not cls._checkpoint_enabled:
            return
        try:
            state = cls._sessions.get(session_id)
            if state is None:
                return
            path = cls._checkpoint_path(session_id)
            payload = {
                "session_id":    session_id,
                "last_accessed": cls._last_accessed.get(session_id, time.time()),
                "state":         _to_json_safe(dict(state)),
            }
            # Atomic write: write to .tmp then rename
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            logger.debug("Session checkpoint save failed for %s: %s",
                         session_id, exc)

    @classmethod
    def load_from_disk(cls) -> int:
        """Rehydrate non-expired sessions from disk. Returns count loaded.

        Called once at API startup (api/main.py:lifespan)."""
        if not _CHECKPOINT_DIR.exists():
            return 0

        cutoff = time.time() - _SESSION_TTL_SECONDS
        loaded = 0
        with cls._lock:
            for path in sorted(_CHECKPOINT_DIR.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    sid           = payload.get("session_id")
                    last_accessed = payload.get("last_accessed", 0)
                    if not sid or last_accessed < cutoff:
                        # Expired — remove the file
                        try:
                            path.unlink()
                        except Exception:
                            pass
                        continue
                    cls._sessions[sid]      = payload.get("state") or {}
                    cls._last_accessed[sid] = last_accessed
                    loaded += 1
                except Exception as exc:
                    logger.debug("Skipping unreadable checkpoint %s: %s", path, exc)
        if loaded:
            logger.info("SessionManager: rehydrated %d session(s) from %s",
                        loaded, _CHECKPOINT_DIR)
        return loaded

    # ── Create ────────────────────────────────────────────────────────────────

    @classmethod
    def create_session(
        cls,
        disease_program: Optional[str] = None,
        organism: Optional[str] = None,
    ) -> str:
        """Create a new session.

        `disease_program` and `organism` are left None when not provided —
        the IngestionAgent will detect organism from protein-name suffixes
        (`OS=...`) at upload time, and disease_program stays as the user's
        free-form label (or None if never set). No hardcoded biology defaults.
        """
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
            test_method          = None,
            is_paired            = False,
            all_groups           = None,
            analysis_params      = None,
            tmt_batches          = None,
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
        cls._save_to_disk(session_id)
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
        cls._save_to_disk(session_id)

    # ── Delete ────────────────────────────────────────────────────────────────

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(session_id, None)
            cls._last_accessed.pop(session_id, None)
        # Remove checkpoint file too
        try:
            cls._checkpoint_path(session_id).unlink(missing_ok=True)
        except Exception:
            pass

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
        # Clean up checkpoint files
        for sid in to_remove:
            try:
                cls._checkpoint_path(sid).unlink(missing_ok=True)
            except Exception:
                pass
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
