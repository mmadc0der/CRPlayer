from __future__ import annotations

from typing import List, Optional, Dict, Any
import warnings

from core.session_manager import SessionManager
from core.dataset_builder import DatasetBuilder
from core.path_resolver import resolve_session_dir, resolve_frame_relative_path


class DatasetService:
    """Orchestrates dataset build using DatasetBuilder and resolvers."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager
        self.builder = DatasetBuilder(session_manager)

    def fetch_labeled(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Return labeled items for a dataset (DB-backed)."""
        return self.builder.fetch_labeled(dataset_id)

    def build_from_sessions(self, project_name: str, session_ids: Optional[List[str]] = None):
        """DEPRECATED: Legacy filesystem dataset builds are not supported."""
        warnings.warn(
            "build_from_sessions is deprecated. Use DB-backed dataset endpoints or DatasetService.fetch_labeled.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("Use DB-backed dataset flows")
