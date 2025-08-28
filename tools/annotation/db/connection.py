from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# Determine default DB path relative to project root: <repo_root>/data/models/annotation.db
_DEF_DB_PATH = (
    Path(__file__).resolve().parents[2]  # repo root (tools/annotation/db -> tools/annotation -> tools)
    / 'data' / 'models' / 'annotation.db'
)


def get_db_path(custom_path: Optional[Path] = None) -> Path:
    path = Path(custom_path) if custom_path else _DEF_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(custom_path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = get_db_path(custom_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Ensure foreign keys
    conn.execute('PRAGMA foreign_keys = ON;')
    return conn
