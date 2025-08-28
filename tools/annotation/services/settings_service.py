from __future__ import annotations

import sqlite3
from typing import Optional, Dict, Any

from db.connection import get_connection
from db.schema import init_db
from db.repository import (
    get_session_db_id,
    upsert_dataset_session_settings as repo_upsert_dataset_session_settings,
    get_dataset_session_settings as repo_get_dataset_session_settings,
    delete_dataset_session_settings as repo_delete_dataset_session_settings,
    get_effective_settings as repo_get_effective_settings,
)


class SettingsService:
    """Manage dataset-session baseline settings and per-frame overrides resolution."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None):
        self._external_conn = conn

    def _conn(self) -> sqlite3.Connection:
        if self._external_conn is not None:
            return self._external_conn
        conn = get_connection()
        init_db(conn)
        return conn

    # ------------- dataset-session ---------------
    def upsert_dataset_session_settings(self, dataset_id: int, session_id_str: str, settings: Dict[str, Any]) -> None:
        conn = self._conn()
        sid = get_session_db_id(conn, session_id_str)
        if sid is None:
            raise ValueError(f"Unknown session_id: {session_id_str}")
        repo_upsert_dataset_session_settings(conn, dataset_id, sid, settings)
        if self._external_conn is None:
            conn.commit()

    def get_dataset_session_settings(self, dataset_id: int, session_id_str: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        sid = get_session_db_id(conn, session_id_str)
        if sid is None:
            return None
        return repo_get_dataset_session_settings(conn, dataset_id, sid)

    def clear_dataset_session_settings(self, dataset_id: int, session_id_str: str) -> None:
        conn = self._conn()
        sid = get_session_db_id(conn, session_id_str)
        if sid is None:
            return
        repo_delete_dataset_session_settings(conn, dataset_id, sid)
        if self._external_conn is None:
            conn.commit()

    # ------------- effective resolution ---------------
    def get_effective_settings(self, dataset_id: int, session_id_str: str, frame_db_id: Optional[int] = None) -> Dict[str, Any]:
        conn = self._conn()
        sid = get_session_db_id(conn, session_id_str)
        if sid is None:
            return {}
        return repo_get_effective_settings(conn, dataset_id, sid, frame_db_id)
