"""
Session management for annotation tool (DB-backed).

Reads sessions and frames from SQLite. Metadata is sourced from the
authoritative metadata.json captured under data/raw and stored in the DB by
the indexer. This module does not write to metadata.json.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
import logging

from db.connection import get_connection
from db.schema import init_db


class SessionManager:
  """Manages sessions using DB as the source of truth."""

  def __init__(self, data_root: str = "data", conn: Optional[sqlite3.Connection] = None):
    self.data_root = Path(data_root)
    self.raw_dir = self.data_root / "raw"
    self._external_conn = conn
    self._log = logging.getLogger("annotation.core.SessionManager")

  def _conn(self) -> sqlite3.Connection:
    if self._external_conn is not None:
      return self._external_conn
    conn = get_connection()
    init_db(conn)
    return conn

  def discover_sessions(self) -> List[Dict[str, Any]]:
    """List sessions from DB with frame counts.

        - path is taken from sessions.root_path
        - frames_count from frames table
        - metadata fields (game_name, start_time) parsed from sessions.metadata_json if present
        - status is read from status.json on disk if available (read-only)
        """
    self._log.debug("discover_sessions")
    conn = self._conn()
    cur = conn.execute("SELECT id, session_id, root_path, metadata_json FROM sessions")
    rows = cur.fetchall()
    results: List[Dict[str, Any]] = []
    for sid_db, session_id, root_path, md_json in rows:
      try:
        md = json.loads(md_json) if md_json else {}
      except Exception:
        self._log.debug("invalid metadata_json for session_id=%s", session_id, exc_info=True)
        md = {}
      # frames count via DB
      cnt = conn.execute("SELECT COUNT(1) FROM frames WHERE session_id = ?", (sid_db, )).fetchone()[0]
      session_dir = Path(root_path)
      status_data = self._get_session_status(session_dir)
      session_info = {
        "session_id": session_id,
        "path": root_path,
        "game_name": md.get("game_name", "Unknown"),
        "frames_count": int(cnt),
        "start_time": md.get("start_time", "unknown"),
        "projects": [],
      }
      # Merge status data into session info
      session_info.update(status_data)
      results.append(session_info)
    return results

  def get_session_path_by_id(self, session_id: str) -> Optional[str]:
    """Resolve a session directory by session_id from DB."""
    self._log.debug("get_session_path_by_id session_id=%s", session_id)
    conn = self._conn()
    row = conn.execute(
      "SELECT root_path FROM sessions WHERE session_id = ?",
      (session_id, ),
    ).fetchone()
    if not row or not row[0]:
      return None
    return str(row[0])

  def find_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
    """Load session info by session_id from DB.

        Returns metadata stored in DB (captured from metadata.json by the indexer).
        Frames are not expanded here; callers should query frames table if needed.
        """
    self._log.debug("find_session_by_id session_id=%s", session_id)
    conn = self._conn()
    row = conn.execute(
      "SELECT id, root_path, metadata_json FROM sessions WHERE session_id = ?",
      (session_id, ),
    ).fetchone()
    if not row:
      return None
    sid_db, root_path, md_json = row
    try:
      metadata = json.loads(md_json) if md_json else {}
    except Exception:
      metadata = {}
    return {
      "session_id":
      session_id,
      "session_dir":
      str(root_path),
      "metadata":
      metadata,
      "frames_count":
      int((self._conn().execute("SELECT COUNT(1) FROM frames WHERE session_id = ?", (sid_db, )).fetchone()
           or (0, ))[0]),
      "projects": [],
    }

  def get_session_db_id(self, session_id: str) -> Optional[int]:
    """Return internal sessions.id for a given session_id string."""
    self._log.debug("get_session_db_id session_id=%s", session_id)
    conn = self._conn()
    row = conn.execute(
      "SELECT id FROM sessions WHERE session_id = ?",
      (session_id, ),
    ).fetchone()
    return int(row[0]) if row else None

  def get_frames_for_session(self, session_id: str) -> List[Dict[str, Any]]:
    """Get all frames for a session."""
    self._log.debug("get_frames_for_session session_id=%s", session_id)
    conn = self._conn()
    cur = conn.execute(
      """
            SELECT f.frame_id, f.ts_ms 
            FROM frames f
            JOIN sessions s ON s.id = f.session_id
            WHERE s.session_id = ?
            ORDER BY f.ts_ms, f.frame_id
        """,
      (session_id, ),
    )
    rows = cur.fetchall()
    return [{"frame_id": row[0], "ts_ms": row[1]} for row in rows]

  def load_session(self, session_path: str) -> Dict[str, Any]:
    """Load session by path (read-only).

        This remains as a helper for tools operating directly on data/raw. It does
        not write to metadata.json and is not used for persistence.
        """
    self._log.debug("load_session session_path=%s", session_path)
    session_dir = Path(session_path)
    metadata_file = session_dir / "metadata.json"
    if not metadata_file.exists():
      raise FileNotFoundError(f"No metadata.json found in {session_dir}")
    with open(metadata_file, "r", encoding="utf-8") as f:
      metadata = json.load(f)
    return {
      "session_id": metadata.get("session_id", session_dir.name),
      "session_dir": str(session_dir),
      "metadata": metadata,
      "frames": metadata.get("frames", []),
      "projects": [],
    }

  def _load_session_projects(self, session_dir: Path) -> Dict[str, Any]:
    """DEPRECATED: Projects are not used in DB-backed flow."""
    warnings.warn(
      "_load_session_projects is deprecated in DB-backed flow.",
      DeprecationWarning,
      stacklevel=2,
    )
    return {}

  def save_session_projects(self, session_dir: str, projects: Dict[str, Any]):
    """DEPRECATED: Projects persistence removed."""
    warnings.warn(
      "save_session_projects is deprecated and has no effect in DB-backed flow.",
      DeprecationWarning,
      stacklevel=2,
    )

  def create_project(self, session_dir: str, project_name: str, annotation_type: str) -> Dict[str, Any]:
    """DEPRECATED: Project creation is not supported in DB-backed flow."""
    warnings.warn(
      "create_project is deprecated and not supported.",
      DeprecationWarning,
      stacklevel=2,
    )
    return {}

  def get_project(self, session_dir: str, project_name: str) -> Optional[Dict[str, Any]]:
    """DEPRECATED: Projects are not supported in DB-backed flow."""
    warnings.warn(
      "get_project is deprecated and returns None.",
      DeprecationWarning,
      stacklevel=2,
    )
    return None

  def update_project(self, session_dir: str, project: Dict[str, Any]):
    """DEPRECATED: Projects are not supported in DB-backed flow."""
    warnings.warn(
      "update_project is deprecated and has no effect.",
      DeprecationWarning,
      stacklevel=2,
    )

  def _get_session_status(self, session_dir: Path) -> Dict[str, Any]:
    """Get current status of a session."""
    status_file = session_dir / "status.json"
    if status_file.exists():
      try:
        with open(status_file, "r") as f:
          status_data = json.load(f)
        return status_data
      except Exception:
        self._log.debug("failed to read status.json at %s", status_file, exc_info=True)
    return {"status": "captured", "collection_status": "unknown", "annotation_status": "not_started"}
