from __future__ import annotations

import json
import sqlite3
from typing import Optional, Tuple


def upsert_session(conn: sqlite3.Connection, session_id: str, root_path: str, metadata: dict) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO sessions(session_id, root_path, metadata_json)
        VALUES (?, ?, ?)
        """,
        (session_id, root_path, json.dumps(metadata, ensure_ascii=False)),
    )
    if cur.rowcount == 0:
        # Update root_path/metadata if changed
        cur.execute(
            """
            UPDATE sessions SET root_path = ?, metadata_json = ? WHERE session_id = ?
            """,
            (root_path, json.dumps(metadata, ensure_ascii=False), session_id),
        )
    # Fetch id
    cur.execute("SELECT id FROM sessions WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("failed to upsert session")
    return int(row[0])


def upsert_frame(conn: sqlite3.Connection, session_db_id: int, frame_id: str, ts_ms: Optional[int]) -> int:
    cur = conn.cursor()
    # Try insert
    cur.execute(
        """
        INSERT OR IGNORE INTO frames(session_id, frame_id, ts_ms)
        VALUES (?, ?, ?)
        """,
        (session_db_id, frame_id, ts_ms),
    )
    if cur.rowcount == 0 and ts_ms is not None:
        # Optionally update timestamp if provided
        cur.execute(
            """
            UPDATE frames SET ts_ms = COALESCE(?, ts_ms)
            WHERE session_id = ? AND frame_id = ?
            """,
            (ts_ms, session_db_id, frame_id),
        )
    # Fetch id
    cur.execute(
        "SELECT id FROM frames WHERE session_id = ? AND frame_id = ?",
        (session_db_id, frame_id),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError("failed to upsert frame")
    return int(row[0])
