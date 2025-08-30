from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from core.session_manager import SessionManager
from db.connection import get_connection
from db.schema import init_db


class SessionService:
    """Thin service over SessionManager for stateless access patterns using session_id."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager

    def resolve_session_dir_by_id(self, session_id: str) -> Path:
        p = self.sm.get_session_path_by_id(session_id)
        if not p:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        return Path(p)

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        info = self.sm.find_session_by_id(session_id)
        if not info:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        return info

    def _get_frame_row_by_idx(self, session_id: str, idx: int) -> Tuple[str, Optional[int]]:
        """Return (frame_id, ts_ms) for the idx-th frame of a session using DB ordering.

        Frames are ordered chronologically by timestamp when available, falling back to
        frame_id for deterministic behavior. This makes index navigation reflect the
        actual capture order.
        """
        if idx < 0:
            raise IndexError("frame_idx must be >= 0")
        sid_db = self.sm.get_session_db_id(session_id)
        if sid_db is None:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        conn = get_connection()
        init_db(conn)
        row = conn.execute(
            "SELECT frame_id, ts_ms FROM frames WHERE session_id = ? "
            "ORDER BY COALESCE(ts_ms, frame_id), frame_id LIMIT 1 OFFSET ?",
            (sid_db, int(idx)),
        ).fetchone()
        if not row:
            # Determine range for clearer error
            total = conn.execute("SELECT COUNT(1) FROM frames WHERE session_id = ?", (sid_db,)).fetchone()[0]
            raise IndexError(f"frame_idx {idx} out of range [0, {max(0,total-1)}]")
        return str(row[0]), (int(row[1]) if row[1] is not None else None)

    def get_frame_by_idx(self, session_id: str, idx: int) -> Dict[str, Any]:
        """DB-backed frame fetch with enrichment from session metadata_json.

        Returns dict with at least: frame_id, filename (if available), timestamp (seconds, if derivable).
        """
        frame_id, ts_ms = self._get_frame_row_by_idx(session_id, idx)
        info = self.get_session_info(session_id)
        md_frames = info['metadata'].get('frames', []) or []
        found: Optional[Dict[str, Any]] = None
        for fr in md_frames:
            if str(fr.get('frame_id')) == frame_id:
                found = fr
                break
        if found is None:
            # Minimal payload from DB
            payload: Dict[str, Any] = {'frame_id': frame_id}
            if ts_ms is not None:
                payload['timestamp'] = float(ts_ms) / 1000.0
            return payload
        # Ensure types/fields
        res = dict(found)
        res['frame_id'] = str(found.get('frame_id', frame_id))
        if 'timestamp' not in res and ts_ms is not None:
            res['timestamp'] = float(ts_ms) / 1000.0
        return res

    def get_frame_for_image(self, session_id: str, idx: int) -> Tuple[Path, Dict[str, Any]]:
        """Resolve absolute image path and frame dict for given idx.

        Raises FileNotFoundError or IndexError on errors.
        """
        frame = self.get_frame_by_idx(session_id, idx)
        info = self.get_session_info(session_id)
        session_dir = Path(info['session_dir'])
        filename = frame.get('filename') or frame.get('path') or frame.get('name')
        if not filename:
            # As a last resort, use frame_id as filename candidate
            filename = str(frame.get('frame_id'))
        from core.path_resolver import resolve_frame_absolute_path
        abs_path = resolve_frame_absolute_path(self.sm, session_dir, info['metadata'].get('session_id', session_id), filename)
        if not abs_path:
            # Try basename variant
            from pathlib import Path as _P
            abs_path = resolve_frame_absolute_path(self.sm, session_dir, info['metadata'].get('session_id', session_id), _P(filename).name)
        if not abs_path or not abs_path.exists():
            raise FileNotFoundError(f"Image not found for idx={idx}")
        return abs_path, frame

    def list_projects(self, session_id: str) -> Dict[str, Any]:
        info = self.get_session_info(session_id)
        return info.get('projects', {})
