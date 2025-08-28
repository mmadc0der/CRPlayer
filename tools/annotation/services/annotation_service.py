from __future__ import annotations

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

from core.session_manager import SessionManager
from core.path_resolver import resolve_session_dir
from db.connection import get_connection
from db.schema import init_db
from db.repository import (
    get_session_db_id,
    get_frame_db_id,
    ensure_membership,
    set_annotation_status,
    upsert_regression,
    delete_regression,
    upsert_single_label,
    delete_single_label,
    add_multilabel,
    remove_multilabel,
    replace_multilabel_set,
    list_frames_with_annotations,
    set_annotation_frame_settings as repo_set_annotation_frame_settings,
)
from services.settings_service import SettingsService


class AnnotationService:
    """Stateless CRUD for annotations by session_id/project_name with per-frame JSON files under data/annotated."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager
        # Lazily initialize DB per call to avoid long-lived connections in Flask workers
        self._settings = SettingsService()

    def _ensure_ann_project_dirs(self, session_id: str, project_name: str) -> Path:
        """Ensure annotated directories exist and return frames dir path."""
        # Ensure base annotated session dir exists
        base = self.sm.annotated_dir / session_id
        (base).mkdir(parents=True, exist_ok=True)

        # Project directories
        project_root = base / 'annotations' / project_name
        frames_dir = project_root / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    def _progress_file(self, session_id: str, project_name: str) -> Path:
        return self.sm.annotated_dir / session_id / 'annotations' / project_name / 'progress.json'

    def get_annotation(self, session_id: str, project_name: str, frame_id: str) -> Optional[Dict[str, Any]]:
        frames_dir = self._ensure_ann_project_dirs(session_id, project_name)
        fpath = frames_dir / f"{frame_id}.json"
        if not fpath.exists():
            return None
        try:
            with open(fpath, 'r') as f:
                return json.load(f)
        except Exception:
            return None

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

    def save_annotation(
        self,
        session_id: str,
        project_name: str,
        frame_id: str,
        annotations: Dict[str, Any],
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Ensure dirs
        frames_dir = self._ensure_ann_project_dirs(session_id, project_name)

        # Write per-frame JSON atomically
        payload = {
            'session_id': session_id,
            'project_name': project_name,
            'frame_id': str(frame_id),
            'annotations': annotations,
            'confidence': confidence if confidence is not None else 1.0,
            'annotated_at': datetime.now().isoformat(),
        }
        tmp = frames_dir / f"{frame_id}.json.tmp"
        out = frames_dir / f"{frame_id}.json"
        with open(tmp, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        Path(tmp).replace(out)

        # Update progress.json (recompute annotated_count via filesystem)
        try:
            annotated_count = len([p for p in frames_dir.glob('*.json')])
        except Exception:
            annotated_count = 0

        # Total frames from session metadata
        info = self.sm.find_session_by_id(session_id)
        total_frames = len(info['metadata'].get('frames', [])) if info else 0
        progress = {
            'session_id': session_id,
            'project_name': project_name,
            'annotated_frames': annotated_count,
            'total_frames': total_frames,
            'progress_percent': (annotated_count / total_frames * 100) if total_frames > 0 else 0,
            'updated_at': datetime.now().isoformat(),
        }
        pfile = self._progress_file(session_id, project_name)
        ptmp = pfile.with_suffix('.json.tmp')
        pfile.parent.mkdir(parents=True, exist_ok=True)
        with open(ptmp, 'w') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        Path(ptmp).replace(pfile)

        return payload
