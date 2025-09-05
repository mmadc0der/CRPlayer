from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

# Determine default DB path relative to project root: <repo_root>/data/models/annotation.db
# This must work both when running from the source tree and inside the Docker image
# where the application is copied to /app and the data volume is mounted at /app/data.

_DEFAULT_DB_REL = Path('data') / 'annotated' / 'annotated.db'

def _detect_repo_root() -> Path:
    """Heuristically detect the project root.

    - In source tree: tools/annotation/db/ -> parents[3] == repo root
    - In Docker image: /app/db/ -> parents[1] == /app
    """
    here = Path(__file__).resolve()
    candidates = []
    try:
        candidates.append(here.parents[3])  # repo root in source checkout
    except IndexError:
        pass
    try:
        candidates.append(here.parents[1])  # /app in Docker image
    except IndexError:
        pass

    for base in candidates:
        try:
            if (base / 'data').exists() or base.name in ('CRPlayer', 'app'):
                return base
        except Exception:
            continue
    # Fallback to immediate parent
    return here.parent


def get_db_path(custom_path: Optional[Path] = None) -> Path:
    # 1) explicit custom path
    if custom_path is not None:
        path = Path(custom_path)
    else:
        # 2) env override
        env_path = os.getenv('ANNOTATION_DB_PATH')
        if env_path:
            path = Path(env_path)
        else:
            # 3) detected repo root + default relative path
            base = _detect_repo_root()
            new_path = base / _DEFAULT_DB_REL
            # Migrate legacy DB if found and new DB doesn't exist yet
            try:
                if not new_path.exists():
                    new_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            path = new_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(custom_path: Optional[Path] = None) -> sqlite3.Connection:
    try:
        db_path = get_db_path(custom_path)
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Ensure foreign keys and optimize for concurrent access
        conn.execute('PRAGMA foreign_keys = ON;')
        conn.execute('PRAGMA journal_mode = WAL;')  # Better concurrency
        conn.execute('PRAGMA synchronous = NORMAL;')  # Balance safety/speed
        conn.execute('PRAGMA cache_size = -64000;')  # 64MB cache
        conn.execute('PRAGMA temp_store = MEMORY;')
        return conn
<<<<<<< Current (Your changes)
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to connect to database at {db_path}: {e}") from e
=======
    except Exception as e:
        raise RuntimeError(f"Failed to connect to database: {e}") from e
>>>>>>> Incoming (Background Agent changes)
