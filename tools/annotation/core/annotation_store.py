"""
Annotation storage and retrieval operations.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from core.session_manager import SessionManager
from models.annotation import AnnotationProject, FrameAnnotation


class AnnotationStore:
    """Handles annotation CRUD operations."""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.current_session_dir: Optional[str] = None
        self.current_project: Optional[AnnotationProject] = None
    
    def load_session_project(self, session_dir: str, project_name: str) -> AnnotationProject:
        """Load or create a project in a session."""
        self.current_session_dir = session_dir
        
        # Try to load existing project
        project = self.session_manager.get_project(session_dir, project_name)
        if project is None:
            # Create new project - default to classification for now
            project = self.session_manager.create_project(session_dir, project_name, 'classification')
        
        self.current_project = project
        return project
    
    def save_annotation(self, frame_id: str, annotation_data: Dict[str, Any], confidence: float = 1.0):
        """Save annotation for current project."""
        if not self.current_project or not self.current_session_dir:
            raise ValueError("No project loaded")
        
        self.current_project.add_annotation(frame_id, annotation_data, confidence)
        self.session_manager.update_project(self.current_session_dir, self.current_project)
    
    def get_annotation(self, frame_id: str) -> Optional[FrameAnnotation]:
        """Get annotation for a frame."""
        if not self.current_project:
            return None
        return self.current_project.get_annotation(frame_id)
    
    def add_category(self, category: str):
        """Add a new category to current project."""
        if not self.current_project:
            raise ValueError("No project loaded")
        
        if category not in self.current_project.categories:
            self.current_project.categories.append(category)
            self.session_manager.update_project(self.current_session_dir, self.current_project)
    
    def save_project(self, session_dir: str, project_name: str, project: AnnotationProject):
        """Save a project to session."""
        self.session_manager.update_project(session_dir, project)
    
    def get_project_stats(self, total_frames: int) -> Dict[str, Any]:
        """Get statistics for current project."""
        if not self.current_project:
            return {}
        return self.current_project.get_progress(total_frames)
