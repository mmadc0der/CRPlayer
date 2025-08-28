"""
DB-backed dataset export preview helpers.

This module replaces legacy filesystem-based dataset building. It does not
write manifests. Callers should use DB queries to fetch labeled samples and
perform any serialization themselves.
"""

from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path

from db.connection import get_connection
from db.schema import init_db
from db.datasets import list_labeled
from core.session_manager import SessionManager
from core.path_resolver import resolve_session_dir, resolve_frame_relative_path


class DatasetBuilder:
    """Produce labeled rows from DB for export; no filesystem manifests."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def fetch_labeled(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Return labeled items for a dataset using DB view.

        Output row keys match `dataset_labeled_view` plus a resolved relative
        path (`frame_path_rel`) when possible.
        """
        conn = get_connection()
        init_db(conn)
        rows = list_labeled(conn, dataset_id)
        # Best-effort relative path resolution using metadata filenames
        results: List[Dict[str, Any]] = []
        for r in rows:
            # We may not have filename/saved_id in frames table yet; derive via metadata when needed.
            try:
                session_info = self.session_manager.find_session_by_id(r['session_id'])
                session_dir = resolve_session_dir(self.session_manager, r['session_id']) or session_info and Path(session_info['session_dir'])
                filename = None
                if session_info and session_info.get('metadata'):
                    frames_md = session_info['metadata'].get('frames', [])
                    # Attempt simple lookup by frame_id equality
                    for fr in frames_md:
                        if str(fr.get('frame_id')) == str(r['frame_id']):
                            filename = fr.get('filename')
                            break
                frame_path_rel = None
                if filename and session_dir:
                    frame_path_rel = resolve_frame_relative_path(self.session_manager, session_dir, r['session_id'], filename)
                r2 = dict(r)
                r2['frame_path_rel'] = frame_path_rel
                results.append(r2)
            except Exception:
                results.append(dict(r))
        return results

