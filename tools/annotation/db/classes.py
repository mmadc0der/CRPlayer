from __future__ import annotations

import sqlite3
from typing import List, Dict, Any


def list_dataset_classes(conn: sqlite3.Connection, dataset_id: int) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, name, idx
        FROM dataset_classes
        WHERE dataset_id = ?
        ORDER BY (idx IS NULL), idx, id
        """,
        (dataset_id,),
    )
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        # Support both sqlite3.Row and tuple
        try:
            out.append({"id": int(r[0] if isinstance(r, tuple) else r["id"]),
                        "name": (r[1] if isinstance(r, tuple) else r["name"]),
                        "idx": (r[2] if isinstance(r, tuple) else r["idx"])})
        except Exception:
            out.append({"id": int(r[0]), "name": r[1], "idx": r[2]})
    return out


def sync_dataset_classes(conn: sqlite3.Connection, dataset_id: int, names: List[str]) -> List[Dict[str, Any]]:
    """
    Ensure that each provided class name exists for the dataset and assign stable ordering by list position.
    - Inserts any missing names.
    - Updates idx to match provided order.
    - Does NOT delete extra classes not in names, but preserves their existing idx values after the provided ones.
    Returns the full ordered list after sync.
    """
    # Normalize and deduplicate while preserving order
    seen = set()
    ordered = []
    for n in names:
        if not isinstance(n, str):
            continue
        s = n.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        ordered.append(s)

    cur = conn.cursor()
    # Fetch existing
    existing_rows = cur.execute(
        "SELECT id, name, idx FROM dataset_classes WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchall()
    existing_by_name = {r[1]: {"id": int(r[0]), "name": r[1], "idx": r[2]} for r in existing_rows}

    # Insert missing with temporary idx (will update below)
    for name in ordered:
        if name not in existing_by_name:
            cur.execute(
                """
                INSERT INTO dataset_classes(dataset_id, name, idx)
                VALUES (?, ?, NULL)
                """,
                (dataset_id, name),
            )
            new_id = int(cur.lastrowid)
            existing_by_name[name] = {"id": new_id, "name": name, "idx": None}

    # Assign idx sequentially based on provided order
    for pos, name in enumerate(ordered):
        row = existing_by_name[name]
        if row.get("idx") != pos:
            cur.execute(
                "UPDATE dataset_classes SET idx = ? WHERE dataset_id = ? AND id = ?",
                (pos, dataset_id, row["id"]),
            )
            row["idx"] = pos

    # Return full ordered list: provided ones first by idx, then the rest by existing idx/id
    cur2 = conn.execute(
        """
        SELECT id, name, idx
        FROM dataset_classes
        WHERE dataset_id = ?
        ORDER BY (idx IS NULL), idx, id
        """,
        (dataset_id,),
    )
    rows2 = cur2.fetchall()
    out2: List[Dict[str, Any]] = []
    for r in rows2:
        try:
            out2.append({"id": int(r[0] if isinstance(r, tuple) else r["id"]),
                         "name": (r[1] if isinstance(r, tuple) else r["name"]),
                         "idx": (r[2] if isinstance(r, tuple) else r["idx"])})
        except Exception:
            out2.append({"id": int(r[0]), "name": r[1], "idx": r[2]})
    return out2
