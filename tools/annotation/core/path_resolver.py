"""
Deterministic helpers to resolve session and frame paths.
Avoids expensive recursive scans and enforces canonical layout:
- Sessions live under data/annotated/<session_id> or data/raw/<session_id>
- Frames are stored under a frames/ subdirectory (preferred) or at session root as fallback
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .session_manager import SessionManager


def resolve_session_dir(session_manager: SessionManager, session_id: str, explicit_session_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a session directory by id with clear precedence.
    Precedence:
    1) explicit_session_dir if it exists
    2) data/annotated/<session_id>
    3) data/raw/<session_id>
    """
    if explicit_session_dir is not None and Path(explicit_session_dir).exists():
        return Path(explicit_session_dir)

    ann = session_manager.annotated_dir / session_id
    if ann.exists():
        return ann

    raw = session_manager.raw_dir / session_id
    if raw.exists():
        return raw

    return None


def resolve_frame_relative_path(
    session_manager: SessionManager,
    session_dir: Path,
    session_id: str,
    filename: str,
) -> Optional[str]:
    """Resolve a frame's relative path inside dataset manifests using a deterministic search order.

    Search order (absolute -> dataset-relative returned string):
    - <session_dir>/<filename>
    - <session_dir>/frames/<basename>
    - data/annotated/<session_id>/<filename>
    - data/annotated/<session_id>/frames/<basename>
    - data/raw/<session_id>/<filename>
    - data/raw/<session_id>/frames/<basename>

    Returns a dataset-relative path like ../../annotated/<session_id>/frames/<basename>
    or ../../raw/<session_id>/frames/<basename> depending on the match location.
    """
    fname = Path(filename)
    basename = fname.name

    ann_root = session_manager.annotated_dir / session_id
    raw_root = session_manager.raw_dir / session_id

    # Candidate list: (absolute_path, dataset_relative)
    rel_base_session = None
    try:
        # Determine if session_dir is under annotated or raw to build relative base accordingly
        sd_str = str(session_dir)
        if str(ann_root) in sd_str:
            rel_base_session = f"../../annotated/{session_id}"
        elif str(raw_root) in sd_str:
            rel_base_session = f"../../raw/{session_id}"
    except Exception:
        rel_base_session = None

    candidates: list[Tuple[Path, str]] = []

    # Prefer explicit session_dir
    if rel_base_session:
        candidates.extend([
            (session_dir / fname, f"{rel_base_session}/{fname.as_posix()}"),
            (session_dir / 'frames' / basename, f"{rel_base_session}/frames/{basename}"),
            (session_dir / basename, f"{rel_base_session}/{basename}"),
        ])

    # Then annotated tree
    candidates.extend([
        (ann_root / fname, f"../../annotated/{session_id}/{fname.as_posix()}"),
        (ann_root / 'frames' / basename, f"../../annotated/{session_id}/frames/{basename}"),
        (ann_root / basename, f"../../annotated/{session_id}/{basename}"),
    ])

    # Then raw tree
    candidates.extend([
        (raw_root / fname, f"../../raw/{session_id}/{fname.as_posix()}"),
        (raw_root / 'frames' / basename, f"../../raw/{session_id}/frames/{basename}"),
        (raw_root / basename, f"../../raw/{session_id}/{basename}"),
    ])

    for abs_path, rel in candidates:
        try:
            if abs_path.exists():
                return rel
        except Exception:
            continue

    # No match found
    return None


def resolve_frame_absolute_path(
    session_manager: SessionManager,
    session_dir: Path,
    session_id: str,
    filename: str,
) -> Optional[Path]:
    """Resolve a frame to an absolute Path using the same deterministic order as relative resolver."""
    fname = Path(filename)
    basename = fname.name

    ann_root = session_manager.annotated_dir / session_id
    raw_root = session_manager.raw_dir / session_id

    candidates = [
        session_dir / fname,
        session_dir / 'frames' / basename,
        session_dir / basename,
        ann_root / fname,
        ann_root / 'frames' / basename,
        ann_root / basename,
        raw_root / fname,
        raw_root / 'frames' / basename,
        raw_root / basename,
    ]

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue

    return None
