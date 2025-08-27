"""
Session management for annotation tool.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from models.annotation import AnnotationProject


class SessionManager:
    """Manages annotation sessions and projects."""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.annotated_dir = self.data_root / "annotated"
    
    def discover_sessions(self) -> List[Dict[str, Any]]:
        """Discover available annotation sessions."""
        sessions = []
        
        search_dirs = [self.raw_dir, self.annotated_dir]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for session_dir in search_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                    
                metadata_file = session_dir / "metadata.json"
                if not metadata_file.exists():
                    continue
                
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Load existing projects
                    projects = self._load_session_projects(session_dir)
                    
                    session_info = {
                        'session_id': metadata.get('session_id', session_dir.name),
                        'path': str(session_dir),
                        'game_name': metadata.get('game_name', 'unknown'),
                        'frames_count': len(metadata.get('frames', [])),
                        'start_time': metadata.get('start_time', 'unknown'),
                        'projects': list(projects.keys()),
                        'status': self._get_session_status(session_dir)
                    }
                    sessions.append(session_info)
                    
                except Exception as e:
                    print(f"[ERROR] Error reading session {session_dir}: {e}")
                    continue
        
        return sessions
    
    def load_session(self, session_path: str) -> Dict[str, Any]:
        """Load session metadata and frame information."""
        session_dir = Path(session_path)
        metadata_file = session_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.json found in {session_dir}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            'session_id': metadata.get('session_id', session_dir.name),
            'session_dir': str(session_dir),
            'metadata': metadata,
            'frames': metadata.get('frames', []),
            'projects': self._load_session_projects(session_dir)
        }
    
    def _load_session_projects(self, session_dir: Path) -> Dict[str, AnnotationProject]:
        """Load all annotation projects for a session."""
        projects = {}
        annotations_file = session_dir / "annotations.json"
        
        if not annotations_file.exists():
            return projects
        
        try:
            with open(annotations_file, 'r') as f:
                data = json.load(f)
            
            projects_data = data.get('projects', {})
            for project_name, project_data in projects_data.items():
                projects[project_name] = AnnotationProject.from_dict(project_data)
                
        except Exception as e:
            print(f"[ERROR] Error loading projects from {annotations_file}: {e}")
        
        return projects
    
    def save_session_projects(self, session_dir: str, projects: Dict[str, AnnotationProject]):
        """Save all annotation projects for a session."""
        session_path = Path(session_dir)
        annotations_file = session_path / "annotations.json"
        
        # Load existing data or create new
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'session_id': session_path.name,
                'updated_at': datetime.now().isoformat(),
                'projects': {}
            }
        
        # Update projects data
        data['projects'] = {name: project.to_dict() for name, project in projects.items()}
        data['updated_at'] = datetime.now().isoformat()
        
        # Save to file
        with open(annotations_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_project(self, session_dir: str, project_name: str, annotation_type: str) -> AnnotationProject:
        """Create a new annotation project in a session."""
        # Load existing projects
        projects = self._load_session_projects(Path(session_dir))
        
        # Create new project
        project = AnnotationProject(
            name=project_name,
            annotation_type=annotation_type
        )
        
        # Add to projects and save
        projects[project_name] = project
        self.save_session_projects(session_dir, projects)
        
        return project
    
    def get_project(self, session_dir: str, project_name: str) -> Optional[AnnotationProject]:
        """Get a specific project from a session."""
        projects = self._load_session_projects(Path(session_dir))
        return projects.get(project_name)
    
    def update_project(self, session_dir: str, project: AnnotationProject):
        """Update a project in a session."""
        projects = self._load_session_projects(Path(session_dir))
        projects[project.name] = project
        self.save_session_projects(session_dir, projects)
    
    def _get_session_status(self, session_dir: Path) -> str:
        """Get current status of a session."""
        status_file = session_dir / "status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                return status_data.get('status', 'unknown')
            except:
                pass
        return 'captured'
