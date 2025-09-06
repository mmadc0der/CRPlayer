from __future__ import annotations

import json
import sqlite3
from typing import Optional, Tuple, List, Dict, Any
from contextlib import contextmanager


@contextmanager
def transaction(conn: sqlite3.Connection):
  """Context manager for database transactions with proper error handling."""
  try:
    yield conn
    conn.commit()
  except Exception:
    conn.rollback()
    raise


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
  cur.execute("SELECT id FROM sessions WHERE session_id = ?", (session_id, ))
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
      UPDATE frames SET ts_ms = COALESCE(ts_ms, ?)
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


# -----------------------------
# Dataset-session settings CRUD
# -----------------------------


def upsert_dataset_session_settings(
  conn: sqlite3.Connection,
  dataset_id: int,
  session_db_id: int,
  settings: dict,
) -> None:
  cur = conn.cursor()
  payload = json.dumps(settings, ensure_ascii=False)
  # Use INSERT OR REPLACE to honor WITHOUT ROWID composite PK
  cur.execute(
    """
    INSERT INTO dataset_session_settings(dataset_id, session_id, settings_json)
    VALUES (?, ?, ?)
    ON CONFLICT(dataset_id, session_id) DO UPDATE SET
      settings_json = excluded.settings_json
    """,
    (dataset_id, session_db_id, payload),
  )


def get_dataset_session_settings(
  conn: sqlite3.Connection,
  dataset_id: int,
  session_db_id: int,
) -> Optional[dict]:
  cur = conn.execute(
    """
    SELECT settings_json
    FROM dataset_session_settings
    WHERE dataset_id = ? AND session_id = ?
    """,
    (dataset_id, session_db_id),
  )
  row = cur.fetchone()
  if not row or row[0] is None:
    return None
  try:
    return json.loads(row[0])
  except Exception:
    return None


def delete_dataset_session_settings(
  conn: sqlite3.Connection,
  dataset_id: int,
  session_db_id: int,
) -> None:
  conn.execute(
    "DELETE FROM dataset_session_settings WHERE dataset_id = ? AND session_id = ?",
    (dataset_id, session_db_id),
  )


# ---------------------------------
# Per-frame override settings (JSON)
# ---------------------------------


def set_annotation_frame_settings(
  conn: sqlite3.Connection,
  dataset_id: int,
  frame_db_id: int,
  settings: Optional[dict],
) -> None:
  payload = json.dumps(settings, ensure_ascii=False) if settings is not None else None
  conn.execute(
    """
    UPDATE annotations
    SET settings_json = ?
    WHERE dataset_id = ? AND frame_id = ?
    """,
    (payload, dataset_id, frame_db_id),
  )


def get_annotation_frame_settings(
  conn: sqlite3.Connection,
  dataset_id: int,
  frame_db_id: int,
) -> Optional[dict]:
  cur = conn.execute(
    "SELECT settings_json FROM annotations WHERE dataset_id = ? AND frame_id = ?",
    (dataset_id, frame_db_id),
  )
  row = cur.fetchone()
  if not row or row[0] is None:
    return None
  try:
    return json.loads(row[0])
  except Exception:
    return None


# -------------------------------------------
# Effective settings resolution (shallow merge)
# -------------------------------------------


def get_effective_settings(
  conn: sqlite3.Connection,
  dataset_id: int,
  session_db_id: int,
  frame_db_id: Optional[int] = None,
) -> dict:
  """
    Resolve baseline settings for (dataset, session) and optionally merge
    per-frame overrides if frame_db_id is provided. Shallow merge: frame overrides win.
    """
  base = get_dataset_session_settings(conn, dataset_id, session_db_id) or {}
  if frame_db_id is None:
    return base
  over = get_annotation_frame_settings(conn, dataset_id, frame_db_id) or {}
  if not base:
    return over
  if not over:
    return base
  merged = {**base, **over}
  return merged


# -----------------
# ID resolver utils
# -----------------


def get_session_db_id(conn: sqlite3.Connection, session_id_str: str) -> Optional[int]:
  row = conn.execute("SELECT id FROM sessions WHERE session_id = ?", (session_id_str, )).fetchone()
  return int(row[0]) if row else None


def get_frame_db_id(conn: sqlite3.Connection, session_db_id: int, frame_id_str: str) -> Optional[int]:
  row = conn.execute(
    "SELECT id FROM frames WHERE session_id = ? AND frame_id = ?",
    (session_db_id, frame_id_str),
  ).fetchone()
  return int(row[0]) if row else None


# ----------------------------
# Annotation membership/status
# ----------------------------


def ensure_membership(conn: sqlite3.Connection,
                      dataset_id: int,
                      frame_db_id: int,
                      default_status: str = "unlabeled") -> None:
  conn.execute(
    """
        INSERT OR IGNORE INTO annotations(dataset_id, frame_id, status)
        VALUES (?, ?, ?)
        """,
    (dataset_id, frame_db_id, default_status),
  )


def set_annotation_status(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int, status: str) -> None:
  conn.execute(
    "UPDATE annotations SET status = ? WHERE dataset_id = ? AND frame_id = ?",
    (status, dataset_id, frame_db_id),
  )


# -----------------------
# Regression CRUD helpers
# -----------------------


def upsert_regression(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int, value_real: float) -> dict:
  ensure_membership(conn, dataset_id, frame_db_id)
  conn.execute(
    """
        INSERT INTO regression_annotations(dataset_id, frame_id, value_real)
        VALUES (?, ?, ?)
        ON CONFLICT(dataset_id, frame_id) DO UPDATE SET value_real = excluded.value_real
        """,
    (dataset_id, frame_db_id, float(value_real)),
  )
  set_annotation_status(conn, dataset_id, frame_db_id, "labeled")

  # Return the created/updated annotation data
  return {"regression_value": float(value_real), "status": "labeled"}


def delete_regression(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int) -> None:
  conn.execute(
    "DELETE FROM regression_annotations WHERE dataset_id = ? AND frame_id = ?",
    (dataset_id, frame_db_id),
  )


# -----------------------------
# Single-label CRUD helpers
# -----------------------------


def upsert_single_label(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int, class_id: int) -> dict:
  ensure_membership(conn, dataset_id, frame_db_id)
  conn.execute(
    """
        INSERT INTO classification_annotations(dataset_id, frame_id, class_id)
        VALUES (?, ?, ?)
        ON CONFLICT(dataset_id, frame_id) DO UPDATE SET class_id = excluded.class_id
        """,
    (dataset_id, frame_db_id, int(class_id)),
  )
  set_annotation_status(conn, dataset_id, frame_db_id, "labeled")

  # Return the created/updated annotation data
  return {"single_label_class_id": int(class_id), "status": "labeled"}


def delete_single_label(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int) -> None:
  conn.execute(
    "DELETE FROM classification_annotations WHERE dataset_id = ? AND frame_id = ?",
    (dataset_id, frame_db_id),
  )


# --------------------------
# Multilabel CRUD helpers
# --------------------------


def add_multilabel(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int, class_id: int) -> None:
  ensure_membership(conn, dataset_id, frame_db_id)
  conn.execute(
    """
        INSERT OR IGNORE INTO annotation_labels(dataset_id, frame_id, class_id)
        VALUES (?, ?, ?)
        """,
    (dataset_id, frame_db_id, int(class_id)),
  )
  set_annotation_status(conn, dataset_id, frame_db_id, "labeled")


def remove_multilabel(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int, class_id: int) -> None:
  conn.execute(
    "DELETE FROM annotation_labels WHERE dataset_id = ? AND frame_id = ? AND class_id = ?",
    (dataset_id, frame_db_id, int(class_id)),
  )


def replace_multilabel_set(conn: sqlite3.Connection, dataset_id: int, frame_db_id: int, class_ids: List[int]) -> None:
  ensure_membership(conn, dataset_id, frame_db_id)
  conn.execute(
    "DELETE FROM annotation_labels WHERE dataset_id = ? AND frame_id = ?",
    (dataset_id, frame_db_id),
  )
  if class_ids:
    conn.executemany(
      "INSERT OR IGNORE INTO annotation_labels(dataset_id, frame_id, class_id) VALUES (?, ?, ?)",
      [(dataset_id, frame_db_id, int(cid)) for cid in class_ids],
    )
    set_annotation_status(conn, dataset_id, frame_db_id, "labeled")


# -------------------------------------------------------
# List frames + annotations for a given session + dataset
# -------------------------------------------------------


def list_frames_with_annotations(
  conn: sqlite3.Connection,
  dataset_id: int,
  session_db_id: int,
  labeled_only: bool = False,
) -> List[Dict[str, Any]]:
  """
    Returns rows for the selected session and dataset.
    Includes frame identity, timestamps, status and payloads using annotations_view.
    """
  sql = ("""
        SELECT
          f.id   AS frame_db_id,
          f.frame_id AS frame_id,
          f.ts_ms AS ts_ms,
          a.status AS status,
          av.value_real AS value_real,
          av.single_label_class_id AS single_label_class_id,
          av.multilabel_class_ids_csv AS multilabel_class_ids_csv
        FROM frames f
        LEFT JOIN annotations a
          ON a.frame_id = f.id AND a.dataset_id = ?
        LEFT JOIN annotations_view av
          ON av.dataset_id = a.dataset_id AND av.frame_id = a.frame_id
        WHERE f.session_id = ? AND (? = 0 OR a.status = 'labeled')
        ORDER BY f.frame_id
    """)
  cur = conn.execute(sql, (dataset_id, session_db_id, 1 if labeled_only else 0))
  return [dict(row) for row in cur.fetchall()]
