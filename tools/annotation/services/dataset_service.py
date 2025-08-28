from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from core.session_manager import SessionManager
from core.dataset_builder import DatasetBuilder
from core.path_resolver import resolve_session_dir, resolve_frame_relative_path
from models.dataset import DatasetManifest


class DatasetService:
    """Orchestrates dataset build using DatasetBuilder and resolvers."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager
        self.builder = DatasetBuilder(session_manager)

    def build_from_sessions(self, project_name: str, session_ids: Optional[List[str]] = None) -> DatasetManifest:
        return self.builder.build_dataset(project_name, session_ids=session_ids)
