"""
Dataset creation and export functionality.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from core.session_manager import SessionManager
from models.dataset import DatasetManifest, DatasetSample
from models.annotation import AnnotationProject


class DatasetBuilder:
    """Builds datasets from annotated sessions."""
    
    def __init__(self, session_manager: SessionManager, datasets_dir: str = "data/datasets"):
        self.session_manager = session_manager
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, dataset_id: str, project_name: str, session_ids: List[str], 
                      splits: Dict[str, float] = None) -> DatasetManifest:
        """Create a new dataset from annotated sessions."""
        if splits is None:
            splits = {'train': 0.8, 'val': 0.2}
        
        # Create dataset manifest
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            annotation_type='classification',  # Will be updated from first project
            project_name=project_name,
            created_at=datetime.now(),
            splits=splits
        )
        
        # Collect samples from sessions
        for session_id in session_ids:
            self._add_session_to_dataset(manifest, session_id, project_name)
        
        # Save manifest
        self._save_manifest(manifest)
        return manifest
    
    def _add_session_to_dataset(self, manifest: DatasetManifest, session_id: str, project_name: str) -> int:
        """Add all annotated frames from a session to the dataset. Returns number of samples added."""
        # Find session directory
        session_dir = None
        for search_dir in [self.session_manager.raw_dir, self.session_manager.annotated_dir]:
            potential_path = search_dir / session_id
            if potential_path.exists():
                session_dir = potential_path
                break
        
        if not session_dir:
            print(f"[WARNING] Session {session_id} not found")
            return 0
        
        # Load session data
        try:
            session_data = self.session_manager.load_session(str(session_dir))
            projects = session_data['projects']
            
            if project_name not in projects:
                print(f"[WARNING] Project {project_name} not found in session {session_id}")
                return 0
            
            project = projects[project_name]
            
            # Update manifest metadata from first project
            if not manifest.samples:
                manifest.annotation_type = project.annotation_type
                manifest.categories = project.categories.copy()
                # Some projects may not define structured targets; keep empty if absent
                manifest.targets = getattr(project, 'targets', {}).copy() if hasattr(project, 'targets') else {}
            
            # Add samples
            frames = session_data['frames']
            added = 0
            for frame_info in frames:
                frame_id = str(frame_info['frame_id'])
                annotation = project.get_annotation(frame_id)
                
                if annotation is None:
                    continue  # Skip unannotated frames
                
                # Determine frame path searching both raw and annotated roots
                frame_filename = frame_info['filename']
                raw_root = self.session_manager.raw_dir / session_id
                ann_root = self.session_manager.annotated_dir / session_id
                candidates = [
                    (raw_root / 'frames' / frame_filename, f"../../raw/{session_id}/frames/{frame_filename}"),
                    (raw_root / frame_filename, f"../../raw/{session_id}/{frame_filename}"),
                    (ann_root / 'frames' / frame_filename, f"../../annotated/{session_id}/frames/{frame_filename}"),
                    (ann_root / frame_filename, f"../../annotated/{session_id}/{frame_filename}")
                ]
                relative_path = None
                for abs_path, rel in candidates:
                    if abs_path.exists():
                        relative_path = rel
                        break
                if not relative_path:
                    print(f"[WARNING] Frame {frame_filename} not found for session {session_id} in raw/ or annotated/")
                    continue
                
                # Extract label based on annotation type
                if project.annotation_type == 'classification':
                    # Prefer new key 'category', fall back to legacy 'game_state'
                    label = annotation.annotations.get('category') or annotation.annotations.get('game_state', '')
                    if not label:
                        # Skip if still empty
                        continue
                elif project.annotation_type == 'regression':
                    label = annotation.annotations
                else:
                    label = annotation.annotations
                
                sample = DatasetSample(
                    frame_path=relative_path,
                    label=label,
                    session_id=session_id,
                    frame_id=frame_id,
                    confidence=annotation.confidence,
                    metadata={
                        'timestamp': frame_info.get('timestamp'),
                        'shape': frame_info.get('shape'),
                        'annotated_at': annotation.annotated_at.isoformat() if annotation.annotated_at else None
                    }
                )
                
                manifest.add_sample(sample)
                added += 1
            return added
            
        except Exception as e:
            print(f"[ERROR] Error processing session {session_id}: {e}")
            return added
    
    def _save_manifest(self, manifest: DatasetManifest):
        """Save dataset manifest to file."""
        dataset_dir = self.datasets_dir / manifest.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_file = dataset_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        print(f"[INFO] Dataset manifest saved to {manifest_file}")
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        datasets = []
        
        if not self.datasets_dir.exists():
            return datasets
        
        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            manifest_file = dataset_dir / "manifest.json"
            if not manifest_file.exists():
                continue
            
            try:
                with open(manifest_file, 'r') as f:
                    manifest_data = json.load(f)
                
                datasets.append({
                    'dataset_id': manifest_data['dataset_id'],
                    'annotation_type': manifest_data['annotation_type'],
                    'project_name': manifest_data['project_name'],
                    'created_at': manifest_data['created_at'],
                    'statistics': manifest_data.get('statistics', {}),
                    'path': str(dataset_dir)
                })
                
            except Exception as e:
                print(f"[ERROR] Error reading dataset {dataset_dir}: {e}")
        
        return datasets
