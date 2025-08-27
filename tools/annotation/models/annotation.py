"""
Annotation data models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


@dataclass
class FrameAnnotation:
    """Single frame annotation data."""
    frame_id: str
    annotations: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    annotated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_id': self.frame_id,
            'annotations': self.annotations,
            'confidence': self.confidence,
            'annotated_at': self.annotated_at.isoformat() if self.annotated_at else None
        }


class AnnotationProject:
    """Represents an annotation project with its settings and annotations."""
    
    def __init__(self, name: str, annotation_type: str = 'classification', categories: List[str] = None):
        self.name = name
        self.annotation_type = annotation_type  # 'classification', 'regression', 'object_detection'
        self.categories = categories or []
        self.hotkeys: Dict[str, str] = {}  # category -> hotkey mapping
        self.annotations: Dict[str, FrameAnnotation] = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_annotation(self, frame_id: str, annotation_data: Dict[str, Any], confidence: float = 1.0):
        """Add or update frame annotation."""
        self.annotations[frame_id] = FrameAnnotation(
            frame_id=frame_id,
            annotations=annotation_data,
            confidence=confidence,
            annotated_at=datetime.now()
        )
        self.updated_at = datetime.now()
    
    def get_annotation(self, frame_id: str) -> Optional[FrameAnnotation]:
        """Get frame annotation."""
        return self.annotations.get(frame_id)
    
    def get_progress(self, total_frames: int) -> Dict[str, Any]:
        """Get annotation progress statistics."""
        annotated_count = len(self.annotations)
        return {
            'annotated_frames': annotated_count,
            'total_frames': total_frames,
            'progress_percent': (annotated_count / total_frames * 100) if total_frames > 0 else 0,
            'completion_rate': annotated_count / total_frames if total_frames > 0 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'annotation_type': self.annotation_type,
            'categories': self.categories,
            'hotkeys': self.hotkeys,
            'annotations': {k: v.to_dict() for k, v in self.annotations.items()},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationProject':
        """Create from dictionary."""
        project = cls(
            name=data['name'],
            annotation_type=data['annotation_type'],
            categories=data.get('categories', [])
        )
        
        project.hotkeys = data.get('hotkeys', {})
        project.created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        project.updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        
        # Load annotations
        annotations_data = data.get('annotations', {})
        for frame_id, ann_data in annotations_data.items():
            project.annotations[frame_id] = FrameAnnotation(
                frame_id=ann_data['frame_id'],
                annotations=ann_data['annotations'],
                confidence=ann_data.get('confidence', 1.0),
                annotated_at=datetime.fromisoformat(ann_data['annotated_at']) if ann_data.get('annotated_at') else None
            )
        
        return project
