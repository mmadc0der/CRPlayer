from __future__ import annotations

from typing import Optional, Dict, Any
from pathlib import Path

from core.session_manager import SessionManager
from models.annotation import AnnotationProject


class AnnotationService:
    """Stateless CRUD for annotations given session_path and project_name."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager

    def _load_projects(self, session_dir: Path) -> Dict[str, AnnotationProject]:
        # Using internal loader for now; could be exposed as public later
        return self.sm._load_session_projects(session_dir)

    def _save_projects(self, session_dir: Path, projects: Dict[str, AnnotationProject]) -> None:
        self.sm.save_session_projects(str(session_dir), projects)

    def get_or_create_project(self, session_path: str, project_name: str, annotation_type: str = 'classification') -> AnnotationProject:
        session_dir = Path(session_path)
        projects = self._load_projects(session_dir)
        project = projects.get(project_name)
        if not project:
            project = self.sm.create_project(str(session_dir), project_name, annotation_type)
            projects[project_name] = project
            self._save_projects(session_dir, projects)
        return project

    def get_annotation(self, session_path: str, project_name: str, frame_id: str) -> Optional[Dict[str, Any]]:
        session_dir = Path(session_path)
        projects = self._load_projects(session_dir)
        project = projects.get(project_name)
        if not project:
            return None
        fa = project.get_annotation(str(frame_id))
        return fa.to_dict() if fa else None

    def save_annotation(self, session_path: str, project_name: str, frame_id: str, annotations: Dict[str, Any], confidence: Optional[float] = None) -> Dict[str, Any]:
        session_dir = Path(session_path)
        projects = self._load_projects(session_dir)
        project = projects.get(project_name)
        if not project:
            project = self.sm.create_project(str(session_dir), project_name, 'classification')
            projects[project_name] = project
        # Upsert annotation
        project.add_annotation(str(frame_id), annotations, confidence)
        # Persist atomically
        self._save_projects(session_dir, projects)
        return project.get_annotation(str(frame_id)).to_dict()
