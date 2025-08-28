from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
import sqlite3

from .connection import get_connection
from .schema import init_db
from .repository import upsert_session, upsert_frame


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_json_stable(path: Path, retries: int = 3, sleep_s: float = 0.05) -> Optional[dict]:
    """Attempt to read JSON atomically by verifying file didn't change during read."""
    for _ in range(retries):
        try:
            stat_before = path.stat()
            with open(path, 'rb') as f:
                data = f.read()
            stat_after = path.stat()
            if (stat_before.st_mtime_ns, stat_before.st_size) == (stat_after.st_mtime_ns, stat_after.st_size):
                return json.loads(data.decode('utf-8'))
        except Exception:
            pass
        time.sleep(sleep_s)
    return _safe_load_json(path)


def reindex_sessions(conn: Optional[sqlite3.Connection], data_root: Path) -> dict:
    """
    Scan only data/raw for sessions and frames and populate the DB.
    Deterministic rules:
      - Use session_id from metadata.json; session dir name is fallback only if missing.
      - Use frame_id from each frame entry; frames without frame_id are skipped.
      - Timestamp field is strictly ts_ms (optional). Strings are coerced to int when possible.
    Metadata reads are stabilized to avoid mid-write inconsistencies.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        init_db(conn)
        should_close = True

    raw_dir = data_root / 'raw'

    # Map session_id -> session_dir (raw only)
    session_map: dict[str, Path] = {}

    if raw_dir.exists():
        for session_dir in raw_dir.iterdir():
            if not session_dir.is_dir():
                continue
            metadata_file = session_dir / 'metadata.json'
            if not metadata_file.exists():
                continue
            md = _load_json_stable(metadata_file) or {}
            sid = md.get('session_id') or session_dir.name
            session_map[sid] = session_dir

    inserted_sessions = 0
    inserted_frames = 0

    for sid, sdir in session_map.items():
        md = _load_json_stable(sdir / 'metadata.json') or {}
        session_db_id = upsert_session(conn, sid, str(sdir), md)
        if session_db_id:
            inserted_sessions += 1
        frames = md.get('frames', [])
        for fr in frames:
            # Deterministic: require frame_id; derive nothing
            frame_id = fr.get('frame_id')
            if not frame_id:
                continue
            ts = fr.get('ts_ms')
            if isinstance(ts, str):
                try:
                    ts = int(ts)
                except Exception:
                    ts = None
            upsert_frame(conn, session_db_id, str(frame_id), ts)
            inserted_frames += 1

    conn.commit()
    if should_close:
        conn.close()

    return {
        'sessions_indexed': inserted_sessions,
        'frames_indexed': inserted_frames,
    }
