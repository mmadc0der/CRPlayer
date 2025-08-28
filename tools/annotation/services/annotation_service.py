from __future__ import annotations

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json

from core.session_manager import SessionManager
from core.path_resolver import resolve_session_dir


class AnnotationService:
    """Stateless CRUD for annotations by session_id/project_name with per-frame JSON files under data/annotated."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager

    def _ensure_ann_project_dirs(self, session_id: str, project_name: str) -> Path:
        """Ensure annotated directories exist and return frames dir path."""
        # Ensure base annotated session dir exists
        base = self.sm.annotated_dir / session_id
        (base).mkdir(parents=True, exist_ok=True)

        # Project directories
        project_root = base / 'annotations' / project_name
        frames_dir = project_root / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    def _progress_file(self, session_id: str, project_name: str) -> Path:
        return self.sm.annotated_dir / session_id / 'annotations' / project_name / 'progress.json'

    def get_annotation(self, session_id: str, project_name: str, frame_id: str) -> Optional[Dict[str, Any]]:
        frames_dir = self._ensure_ann_project_dirs(session_id, project_name)
        fpath = frames_dir / f"{frame_id}.json"
        if not fpath.exists():
            return None
        try:
            with open(fpath, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def save_annotation(
        self,
        session_id: str,
        project_name: str,
        frame_id: str,
        annotations: Dict[str, Any],
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Ensure dirs
        frames_dir = self._ensure_ann_project_dirs(session_id, project_name)

        # Write per-frame JSON atomically
        payload = {
            'session_id': session_id,
            'project_name': project_name,
            'frame_id': str(frame_id),
            'annotations': annotations,
            'confidence': confidence if confidence is not None else 1.0,
            'annotated_at': datetime.now().isoformat(),
        }
        tmp = frames_dir / f"{frame_id}.json.tmp"
        out = frames_dir / f"{frame_id}.json"
        with open(tmp, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        Path(tmp).replace(out)

        # Update progress.json (recompute annotated_count via filesystem)
        try:
            annotated_count = len([p for p in frames_dir.glob('*.json')])
        except Exception:
            annotated_count = 0

        # Total frames from session metadata
        info = self.sm.find_session_by_id(session_id)
        total_frames = len(info['metadata'].get('frames', [])) if info else 0
        progress = {
            'session_id': session_id,
            'project_name': project_name,
            'annotated_frames': annotated_count,
            'total_frames': total_frames,
            'progress_percent': (annotated_count / total_frames * 100) if total_frames > 0 else 0,
            'updated_at': datetime.now().isoformat(),
        }
        pfile = self._progress_file(session_id, project_name)
        ptmp = pfile.with_suffix('.json.tmp')
        pfile.parent.mkdir(parents=True, exist_ok=True)
        with open(ptmp, 'w') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        Path(ptmp).replace(pfile)

        return payload
