from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import Optional
import threading
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


def _derive_ts_ms(frame_entry: dict) -> Optional[int]:
    """Return ts_ms if present else convert 'timestamp' seconds to ms; otherwise None.

    - Accepts numeric or string types for ts_ms/timestamp.
    - Rounds to nearest millisecond when converting seconds.
    """
    ts = frame_entry.get('ts_ms')
    if ts is not None:
        try:
            return int(ts)
        except Exception:
            pass
    sec = frame_entry.get('timestamp')
    if sec is not None:
        try:
            # Support string or float seconds
            sec_f = float(sec)
            return int(round(sec_f * 1000.0))
        except Exception:
            return None
    return None


def reindex_sessions(conn: Optional[sqlite3.Connection], data_root: Path) -> dict:
    """
    Scan data/raw for sessions and frames and populate the DB.
    Deterministic rules:
      - Use session_id from metadata.json; fall back to session dir name if missing.
      - Prefer frames listed in metadata.json and use their frame_id; frames without frame_id are skipped.
      - If metadata has no frames list, enumerate files under <session>/frames (preferred) and <session> as frames.
        In this case, use the filename as frame_id (stable identifier) and leave ts_ms NULL.
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
            try:
                # Ignore system/hidden folders
                if session_dir.name.startswith('.'):
                    continue
            except Exception:
                continue
            try:
                metadata_file = session_dir / 'metadata.json'
                md = _load_json_stable(metadata_file) if metadata_file.exists() else {}
                sid = md.get('session_id') or session_dir.name
                try:
                    sid = str(sid).strip()
                except Exception:
                    pass
                session_map[sid] = session_dir
            except Exception:
                continue

    inserted_sessions = 0
    inserted_frames = 0
    details: list[dict] = []

    for sid, sdir in session_map.items():
        md = _load_json_stable(sdir / 'metadata.json') or {}
        session_db_id = upsert_session(conn, sid, str(sdir), md)
        if session_db_id:
            inserted_sessions += 1
        frames = md.get('frames', [])
        if frames and isinstance(frames, list):
            sample_ids = []
            for fr in frames:
                # Deterministic: require frame_id from metadata; derive nothing else except ts
                frame_id = fr.get('frame_id')
                if not frame_id:
                    continue
                ts = _derive_ts_ms(fr)
                upsert_frame(conn, session_db_id, str(frame_id), ts)
                inserted_frames += 1
                if len(sample_ids) < 5:
                    sample_ids.append(str(frame_id))
            try:
                print(f"[indexer][session={sid}] frames_from=metadata count={len(frames)} sample={sample_ids}", flush=True)
            except Exception:
                pass
            details.append({
                'session_id': sid,
                'source': 'metadata',
                'count': len(frames),
                'sample': sample_ids,
            })
        else:
            # Filesystem fallback: enumerate frames by filename
            candidates = []
            frames_dir = sdir / 'frames'
            exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
            try:
                if frames_dir.exists():
                    for p in sorted(frames_dir.iterdir()):
                        if p.is_file() and p.suffix.lower() in exts:
                            candidates.append(p.name)
                else:
                    # Fallback to session root
                    for p in sorted(sdir.iterdir()):
                        if p.is_file() and p.suffix.lower() in exts:
                            candidates.append(p.name)
            except Exception:
                candidates = []
            try:
                origin = str(frames_dir) if frames_dir.exists() else str(sdir)
                print(f"[indexer][session={sid}] frames_from=filesystem dir={origin} count={len(candidates)} sample={candidates[:5]}", flush=True)
            except Exception:
                pass
            details.append({
                'session_id': sid,
                'source': 'filesystem',
                'dir': str(frames_dir) if frames_dir.exists() else str(sdir),
                'count': len(candidates),
                'sample': candidates[:5],
            })
            for fname in candidates:
                # Use filename as stable frame_id
                upsert_frame(conn, session_db_id, fname, None)
                inserted_frames += 1

    conn.commit()
    if should_close:
        conn.close()

    return {
        'sessions_indexed': inserted_sessions,
        'frames_indexed': inserted_frames,
        'details': details,
    }


def run_indexer_loop(
    data_root: Path,
    interval_s: float = 5.0,
    jitter_s: float = 1.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the reindexer periodically in the background.

    - Read-only access to metadata.json under data/raw/*.
    - Idempotent DB upserts; safe to run continuously.
    - Lightweight logging via print; replace with app logger if available.
    - Optional stop_event to terminate loop gracefully.
    """
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        t0 = time.time()
        try:
            stats = reindex_sessions(None, data_root)
            print(f"[indexer] sessions={stats.get('sessions_indexed',0)} frames={stats.get('frames_indexed',0)}", flush=True)
        except Exception as e:
            print(f"[indexer][error] {e}", flush=True)
        # Sleep with small jitter to avoid sync with writers
        elapsed = time.time() - t0
        delay = max(0.5, interval_s - elapsed) + (random.random() * jitter_s)
        # Allow prompt shutdown
        if stop_event is not None:
            # Wait in small increments so we can react to stop_event
            waited = 0.0
            step = 0.25
            while waited < delay:
                if stop_event.is_set():
                    return
                time.sleep(min(step, delay - waited))
                waited += step
        else:
            time.sleep(delay)
