from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from core.session_manager import SessionManager


class SessionService:
    """Thin service over SessionManager for stateless access patterns using session_id."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager

    def resolve_session_dir_by_id(self, session_id: str) -> Path:
        p = self.sm.get_session_path_by_id(session_id)
        if not p:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        return Path(p)

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        info = self.sm.find_session_by_id(session_id)
        if not info:
            raise FileNotFoundError(f"Session not found by id: {session_id}")
        return info

    def get_frame_by_idx(self, session_id: str, idx: int) -> Dict[str, Any]:
        info = self.get_session_info(session_id)
        frames = info['metadata'].get('frames', [])
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"frame_idx {idx} out of range [0, {len(frames)-1}]")
        return frames[idx]

    def list_projects(self, session_id: str) -> Dict[str, Any]:
        info = self.get_session_info(session_id)
        return info.get('projects', {})
