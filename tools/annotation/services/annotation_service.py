from __future__ import annotations

from typing import Optional, Dict, Any, List

from core.session_manager import SessionManager
from db.connection import get_connection
from db.schema import init_db
from db.repository import (
    get_session_db_id,
    get_frame_db_id,
    ensure_membership,
    set_annotation_status,
    upsert_regression,
    upsert_single_label,
    replace_multilabel_set,
    list_frames_with_annotations,
    set_annotation_frame_settings as repo_set_annotation_frame_settings,
)
from services.settings_service import SettingsService


class AnnotationService:
    """DB-backed CRUD for annotations and per-frame override settings."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager
        # Lazily initialize DB per call to avoid long-lived connections in Flask workers
        self._settings = SettingsService()

    # No filesystem-based methods remain

    # ---------------------------
    # DB-backed annotation methods
    # ---------------------------

    def _conn(self):
        conn = get_connection()
        init_db(conn)
        return conn

    def _resolve_ids(self, conn, session_id: str, frame_id: str) -> tuple[int, int]:
        sid = get_session_db_id(conn, session_id)
        if sid is None:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        fid = get_frame_db_id(conn, sid, str(frame_id))
        if fid is None:
            raise FileNotFoundError(f"Frame not found: session_id={session_id} frame_id={frame_id}")
        return sid, fid

    def get_annotation_db(self, session_id: str, dataset_id: int, frame_id: str) -> Optional[Dict[str, Any]]:
        """Return unified view for a single frame: status + payloads + effective settings."""
        conn = self._conn()
        sid, fid = self._resolve_ids(conn, session_id, frame_id)
        # Reuse list query for single item
        rows = list_frames_with_annotations(conn, dataset_id, sid, labeled_only=False)
        row = next((r for r in rows if str(r['frame_id']) == str(frame_id)), None)
        if not row:
            return None
        eff = self._settings.get_effective_settings(dataset_id, session_id, frame_db_id=fid)
        row['effective_settings'] = eff
        return row

    def list_annotations_for_session(self, session_id: str, dataset_id: int, labeled_only: bool = False) -> List[Dict[str, Any]]:
        conn = self._conn()
        sid = get_session_db_id(conn, session_id)
        if sid is None:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        return list_frames_with_annotations(conn, dataset_id, sid, labeled_only=labeled_only)

    def save_regression(self, session_id: str, dataset_id: int, frame_id: str, value: float, override_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conn = self._conn()
        sid, fid = self._resolve_ids(conn, session_id, frame_id)
        ensure_membership(conn, dataset_id, fid)
        # Optional: enforce bounds if provided in effective settings
        try:
            eff = self._settings.get_effective_settings(dataset_id, session_id, frame_db_id=fid)
            reg = eff.get('regression') if isinstance(eff, dict) else None
            if isinstance(reg, dict):
                vmin = reg.get('min')
                vmax = reg.get('max')
                if vmin is not None and value < float(vmin):
                    raise ValueError(f"regression value {value} < min {vmin}")
                if vmax is not None and value > float(vmax):
                    raise ValueError(f"regression value {value} > max {vmax}")
        except Exception:
            # Do not fail on settings retrieval errors; only enforce when valid bounds are present
            pass
        upsert_regression(conn, dataset_id, fid, float(value))
        if override_settings is not None:
            repo_set_annotation_frame_settings(conn, dataset_id, fid, override_settings)
        conn.commit()
        return self.get_annotation_db(session_id, dataset_id, frame_id) or {}

    def save_single_label(self, session_id: str, dataset_id: int, frame_id: str, class_id: int, override_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conn = self._conn()
        sid, fid = self._resolve_ids(conn, session_id, frame_id)
        ensure_membership(conn, dataset_id, fid)
        upsert_single_label(conn, dataset_id, fid, int(class_id))
        if override_settings is not None:
            repo_set_annotation_frame_settings(conn, dataset_id, fid, override_settings)
        conn.commit()
        return self.get_annotation_db(session_id, dataset_id, frame_id) or {}

    def save_multilabel(self, session_id: str, dataset_id: int, frame_id: str, class_ids: List[int], override_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conn = self._conn()
        sid, fid = self._resolve_ids(conn, session_id, frame_id)
        ensure_membership(conn, dataset_id, fid)
        replace_multilabel_set(conn, dataset_id, fid, [int(c) for c in class_ids])
        if override_settings is not None:
            repo_set_annotation_frame_settings(conn, dataset_id, fid, override_settings)
        conn.commit()
        return self.get_annotation_db(session_id, dataset_id, frame_id) or {}

    def set_status(self, session_id: str, dataset_id: int, frame_id: str, status: str) -> None:
        conn = self._conn()
        sid, fid = self._resolve_ids(conn, session_id, frame_id)
        ensure_membership(conn, dataset_id, fid)
        set_annotation_status(conn, dataset_id, fid, status)
        conn.commit()

    def set_frame_override_settings(self, session_id: str, dataset_id: int, frame_id: str, settings: Optional[Dict[str, Any]]) -> None:
        conn = self._conn()
        sid, fid = self._resolve_ids(conn, session_id, frame_id)
        ensure_membership(conn, dataset_id, fid)
        repo_set_annotation_frame_settings(conn, dataset_id, fid, settings)
        conn.commit()
