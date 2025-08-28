from __future__ import annotations

import sqlite3
from typing import List, Optional, Dict, Any


def list_projects(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.execute("SELECT id, name, description, created_at FROM projects ORDER BY id DESC")
    return [dict(row) for row in cur.fetchall()]


def create_project(conn: sqlite3.Connection, name: str, description: Optional[str]) -> int:
    cur = conn.execute(
        "INSERT INTO projects(name, description) VALUES (?, ?)",
        (name, description),
    )
    return int(cur.lastrowid)


def list_datasets(conn: sqlite3.Connection, project_id: int) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT d.id, d.name, d.description, d.target_type_id, d.created_at
        FROM datasets d
        WHERE d.project_id = ?
        ORDER BY d.id DESC
        """,
        (project_id,),
    )
    return [dict(row) for row in cur.fetchall()]


def create_dataset(
    conn: sqlite3.Connection,
    project_id: int,
    name: str,
    description: Optional[str],
    target_type_id: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO datasets(project_id, name, description, target_type_id)
        VALUES (?, ?, ?, ?)
        """,
        (project_id, name, description, target_type_id),
    )
    return int(cur.lastrowid)


def get_dataset(conn: sqlite3.Connection, dataset_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, project_id, name, description, target_type_id, created_at FROM datasets WHERE id = ?",
        (dataset_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def dataset_progress(conn: sqlite3.Connection, dataset_id: int) -> Dict[str, int]:
    total = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()[0]
    labeled = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE dataset_id = ? AND status = 'labeled'",
        (dataset_id,),
    ).fetchone()[0]
    return {"total": int(total), "labeled": int(labeled), "unlabeled": int(total - labeled)}
