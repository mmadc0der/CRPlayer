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
                      splits: Dict[str, float] = None, session_paths: List[str] = None) -> DatasetManifest:
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
        if session_paths:
            for spath in session_paths:
                self._add_session_to_dataset(manifest, self._resolve_session_id(spath), project_name, explicit_session_dir=Path(spath))
        else:
            for session_id in session_ids:
                self._add_session_to_dataset(manifest, session_id, project_name)
        
        # Save manifest
        self._save_manifest(manifest)
        return manifest
    
    def _add_session_to_dataset(self, manifest: DatasetManifest, session_id: str, project_name: str, explicit_session_dir: Path = None) -> int:
        """Add all annotated frames from a session to the dataset. Returns number of samples added."""
        # Find session directory
        session_dir = explicit_session_dir
        if session_dir is not None and not session_dir.exists():
            session_dir = None
        for search_dir in [self.session_manager.raw_dir, self.session_manager.annotated_dir]:
            potential_path = search_dir / session_id
            if potential_path.exists():
                session_dir = potential_path
                break
        
        # Fallback: scan for a session whose metadata.json has matching session_id
        if not session_dir:
            for search_dir in [self.session_manager.raw_dir, self.session_manager.annotated_dir]:
                if not search_dir.exists():
                    continue
                for candidate in search_dir.iterdir():
                    if not candidate.is_dir():
                        continue
                    meta = candidate / 'metadata.json'
                    try:
                        if meta.exists():
                            with open(meta, 'r') as f:
                                md = json.load(f)
                            if md.get('session_id') == session_id:
                                session_dir = candidate
                                break
                    except Exception:
                        continue
                if session_dir:
                    break
        
        if not session_dir:
            print(f"[WARNING] Session {session_id} not found by name or metadata scan")
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
            # Diagnostics for this session
            session_diag = {
                'session_id': session_id,
                'frames_total': len(frames),
                'annotations_found': 0,
                'labels_present': 0,
                'path_found': 0,
                'missing_path': 0,
                'missing_label': 0,
                'missing_annotation': 0,
                'missing_path_examples': [],
                'raw_root_exists': False,
                'ann_root_exists': False
            }

            # Migrate legacy index-keyed annotations -> frame_id (in-memory, persist if possible)
            try:
                total = len(frames)
                idx_keys = [k for k in list(project.annotations.keys()) if isinstance(k, str) and k.isdigit() and int(k) < total]
                migrated = 0
                for k in idx_keys:
                    idx = int(k)
                    frame_id_m = str(frames[idx]['frame_id'])
                    if frame_id_m not in project.annotations:
                        project.annotations[frame_id_m] = project.annotations[k]
                        project.annotations[frame_id_m].frame_id = frame_id_m
                        del project.annotations[k]
                        migrated += 1
                if migrated:
                    # Try to persist so future calls see consistent keys
                    try:
                        self.session_manager.update_project(str(session_dir), project)
                    except Exception:
                        pass
            except Exception:
                pass

            added = 0
            # Derive folder-based session name and root kind for relative paths
            folder_session_name = session_dir.name
            root_kind = 'raw'
            try:
                # Decide whether explicit dir is under raw/ or annotated/
                if str(session_dir).startswith(str(self.session_manager.annotated_dir)):
                    root_kind = 'annotated'
            except Exception:
                pass

            for frame_info in frames:
                frame_id = str(frame_info['frame_id'])
                annotation = project.get_annotation(frame_id)
                
                if annotation is None:
                    session_diag['missing_annotation'] += 1
                    continue  # Skip unannotated frames
                session_diag['annotations_found'] += 1
                
                # Determine frame path searching both raw and annotated roots
                frame_filename = frame_info['filename']
                raw_root = self.session_manager.raw_dir / session_id
                ann_root = self.session_manager.annotated_dir / session_id
                session_diag['raw_root_exists'] = session_diag['raw_root_exists'] or raw_root.exists()
                session_diag['ann_root_exists'] = session_diag['ann_root_exists'] or ann_root.exists()
                # Support both cases: filename may already include 'frames/' or other subdirs
                from pathlib import Path as _P
                fname_path = _P(frame_filename)
                basename = fname_path.name
                # Build candidate pairs (absolute path, relative path string in dataset)
                base_root = session_dir  # prefer actual folder location
                rel_base = f"../../{root_kind}/{folder_session_name}"
                candidates = [
                    # prefer explicit session_dir
                    (base_root / fname_path, f"{rel_base}/{fname_path.as_posix()}"),
                    (base_root / 'frames' / basename, f"{rel_base}/frames/{basename}"),
                    (base_root / basename, f"{rel_base}/{basename}"),
                    # raw, keep original relative if present
                    (raw_root / fname_path, f"../../raw/{session_id}/{fname_path.as_posix()}"),
                    (raw_root / 'frames' / basename, f"../../raw/{session_id}/frames/{basename}"),
                    (raw_root / basename, f"../../raw/{session_id}/{basename}"),
                    # annotated
                    (ann_root / fname_path, f"../../annotated/{session_id}/{fname_path.as_posix()}"),
                    (ann_root / 'frames' / basename, f"../../annotated/{session_id}/frames/{basename}"),
                    (ann_root / basename, f"../../annotated/{session_id}/{basename}")
                ]
                relative_path = None
                for abs_path, rel in candidates:
                    if abs_path.exists():
                        relative_path = rel
                        break
                if not relative_path:
                    # Try alternate extensions
                    alt_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
                    stem = basename.rsplit('.', 1)[0] if '.' in basename else basename
                    alt_candidates = []
                    for ext in alt_exts:
                        alt_name = stem + ext
                        alt_candidates.extend([
                            (base_root / 'frames' / alt_name, f"{rel_base}/frames/{alt_name}"),
                            (base_root / alt_name, f"{rel_base}/{alt_name}"),
                            (raw_root / 'frames' / alt_name, f"../../raw/{session_id}/frames/{alt_name}"),
                            (raw_root / alt_name, f"../../raw/{session_id}/{alt_name}"),
                            (ann_root / 'frames' / alt_name, f"../../annotated/{session_id}/frames/{alt_name}"),
                            (ann_root / alt_name, f"../../annotated/{session_id}/{alt_name}")
                        ])
                    for abs_path, rel in alt_candidates:
                        if abs_path.exists():
                            relative_path = rel
                            break
                if not relative_path:
                    # Last resort: recursive search for basename under session roots
                    try:
                        found_abs = None
                        target_names = {basename.lower()}
                        if '.' in basename:
                            target_names.add(basename.rsplit('.', 1)[0].lower())
                        for root in [base_root, raw_root, ann_root]:
                            if root.exists():
                                for p in root.rglob('*'):
                                    if not p.is_file():
                                        continue
                                    name_lower = p.name.lower()
                                    stem_lower = p.stem.lower()
                                    if name_lower in target_names or stem_lower in target_names:
                                        found_abs = p
                                        break
                            if found_abs:
                                break
                        if found_abs:
                            # Compute relative_path based on which root matched
                            try:
                                if str(found_abs).startswith(str(raw_root)):
                                    rel = found_abs.relative_to(raw_root).as_posix()
                                    relative_path = f"../../raw/{session_id}/{rel}"
                                else:
                                    rel = found_abs.relative_to(ann_root).as_posix()
                                    relative_path = f"../../annotated/{session_id}/{rel}"
                            except Exception:
                                relative_path = None
                    except Exception:
                        pass
                if not relative_path:
                    print(f"[WARNING] Frame {frame_filename} not found for session {session_id} in raw/ or annotated/")
                    session_diag['missing_path'] += 1
                    if len(session_diag['missing_path_examples']) < 5:
                        session_diag['missing_path_examples'].append({
                            'filename': frame_filename,
                            'tried': [str(p) for p, _ in candidates]
                        })
                    continue
                
                # Extract label based on annotation type
                if project.annotation_type == 'classification':
                    # Prefer new key 'category', fall back to legacy 'game_state'
                    label = annotation.annotations.get('category') or annotation.annotations.get('game_state', '')
                    if not label:
                        # Skip if still empty
                        session_diag['missing_label'] += 1
                        continue
                    else:
                        session_diag['labels_present'] += 1
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
                session_diag['path_found'] += 1
            # Store diagnostics
            try:
                if 'session_diagnostics' not in manifest.metadata:
                    manifest.metadata['session_diagnostics'] = []
                manifest.metadata['session_diagnostics'].append(session_diag)
            except Exception:
                pass
            return added
            
        except Exception as e:
            print(f"[ERROR] Error processing session {session_id}: {e}")
            return added

    def _resolve_session_id(self, session_path: str) -> str:
        """Extract session_id from an absolute session path by reading metadata.json or folder name."""
        try:
            p = Path(session_path)
            meta = p / 'metadata.json'
            if meta.exists():
                with open(meta, 'r') as f:
                    md = json.load(f)
                return md.get('session_id', p.name)
            return p.name
        except Exception:
            return Path(session_path).name
    
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
