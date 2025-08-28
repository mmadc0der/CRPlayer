from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from core.session_manager import SessionManager


class SessionService:
    """Thin service over SessionManager for stateless access patterns."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager

    def resolve_session_dir(self, session_path: Optional[str] = None) -> Path:
        if not session_path:
            raise ValueError("session_path is required")
        p = Path(session_path)
        if not p.exists():
            raise FileNotFoundError(f"Session path not found: {session_path}")
        return p

    def get_session_info(self, session_path: str) -> Dict[str, Any]:
        return self.sm.load_session(session_path)

    def get_frame_by_idx(self, session_path: str, idx: int) -> Dict[str, Any]:
        info = self.get_session_info(session_path)
        frames = info['metadata'].get('frames', [])
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"frame_idx {idx} out of range [0, {len(frames)-1}]")
        return frames[idx]

    def list_projects(self, session_path: str) -> Dict[str, Any]:
        info = self.get_session_info(session_path)
        return info.get('projects', {})
