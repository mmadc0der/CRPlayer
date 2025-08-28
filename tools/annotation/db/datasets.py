from __future__ import annotations

import sqlite3
from typing import Dict, Any, List, Optional
from .repository import (
    get_session_db_id,
    upsert_dataset_session_settings,
)


def enroll_session_frames(
    conn: sqlite3.Connection,
    dataset_id: int,
    session_id_str: str,
    default_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Create membership rows (annotations) for all frames of the given session in the dataset.
    - status defaults to 'unlabeled'
    - settings_json can be prefilled with default settings (categories/hotkeys), else NULL
    Idempotent: uses INSERT OR IGNORE on (dataset_id, frame_id) PK.
    """
    # Resolve DB ids
    session_db_id = get_session_db_id(conn, session_id_str)
    if session_db_id is None:
        raise ValueError(f"Unknown session_id: {session_id_str}")

    # Frames for session
    cur = conn.execute("SELECT id FROM frames WHERE session_id = ?", (session_db_id,))
    frame_ids = [int(r[0]) for r in cur.fetchall()]

    # Store baseline settings once per (dataset, session)
    if default_settings is not None:
        upsert_dataset_session_settings(conn, dataset_id, session_db_id, default_settings)

    inserted = 0
    for fid in frame_ids:
        conn.execute(
            """
            INSERT OR IGNORE INTO annotations(dataset_id, frame_id, status)
            VALUES (?, ?, 'unlabeled')
            """,
            (dataset_id, fid),
        )
        if conn.total_changes:
            inserted += 1

    conn.commit()
    return {"frames": len(frame_ids), "inserted": inserted}


def list_labeled(conn: sqlite3.Connection, dataset_id: int) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT dataset_id, session_id, frame_id, value_real, single_label_class_id, multilabel_class_ids_csv
        FROM dataset_labeled_view
        WHERE dataset_id = ?
        ORDER BY session_id, frame_id
        """,
        (dataset_id,),
    )
    return [dict(row) for row in cur.fetchall()]
