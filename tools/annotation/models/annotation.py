"""
DEPRECATED: Legacy filesystem-based annotation models.

These classes were used for per-frame JSON storage and in-memory aggregation.
The system now uses a DB-backed flow via repository/services. Do not use these
models for new code. They are kept only to avoid breaking imports in older modules.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings


@dataclass
class FrameAnnotation:
  """DEPRECATED: Single frame annotation data (legacy)."""

  frame_id: str
  annotations: Dict[str, Any] = field(default_factory=dict)
  confidence: float = 1.0
  annotated_at: Optional[datetime] = None

  def to_dict(self) -> Dict[str, Any]:
    return {
      "frame_id": self.frame_id,
      "annotations": self.annotations,
      "confidence": self.confidence,
      "annotated_at": self.annotated_at.isoformat() if self.annotated_at else None,
    }


class AnnotationProject:
  """DEPRECATED legacy in-memory project. Use DB-backed services instead.

    This class remains only for backward compatibility. Instantiating it will
    emit a deprecation warning. All functionality here is considered legacy.
    """

  def __init__(self, name: str, annotation_type: str = "classification", categories: List[str] = None):
    warnings.warn(
      "AnnotationProject is deprecated. Use DB-backed services (AnnotationService, SettingsService).",
      DeprecationWarning,
      stacklevel=2,
    )
    self.name = name
    self.annotation_type = annotation_type  # 'classification', 'regression', 'object_detection'
    self.categories = categories or []
    self.hotkeys: Dict[str, str] = {}  # category -> hotkey mapping
    self.annotations: Dict[str, FrameAnnotation] = {}
    self.created_at = datetime.now()
    self.updated_at = datetime.now()

  def add_annotation(self, frame_id: str, annotation_data: Dict[str, Any], confidence: float = 1.0):
    """DEPRECATED: Adds or updates a legacy in-memory annotation."""
    self.annotations[frame_id] = FrameAnnotation(frame_id=frame_id,
                                                 annotations=annotation_data,
                                                 confidence=confidence,
                                                 annotated_at=datetime.now())
    self.updated_at = datetime.now()

  def get_annotation(self, frame_id: str) -> Optional[FrameAnnotation]:
    """DEPRECATED: Get legacy in-memory frame annotation."""
    return self.annotations.get(frame_id)

  def get_progress(self, total_frames: int) -> Dict[str, Any]:
    """DEPRECATED: Get progress based on legacy in-memory annotations."""
    annotated_count = len(self.annotations)
    return {
      "annotated_frames": annotated_count,
      "total_frames": total_frames,
      "progress_percent": (annotated_count / total_frames * 100) if total_frames > 0 else 0,
      "completion_rate": annotated_count / total_frames if total_frames > 0 else 0,
    }

  def to_dict(self) -> Dict[str, Any]:
    """DEPRECATED: Serialize legacy model."""
    return {
      "name": self.name,
      "annotation_type": self.annotation_type,
      "categories": self.categories,
      "hotkeys": self.hotkeys,
      "annotations": {
        k: v.to_dict()
        for k, v in self.annotations.items()
      },
      "created_at": self.created_at.isoformat() if self.created_at else None,
      "updated_at": self.updated_at.isoformat() if self.updated_at else None,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "AnnotationProject":
    """DEPRECATED: Deserialize legacy model."""
    project = cls(name=data["name"], annotation_type=data["annotation_type"], categories=data.get("categories", []))

    project.hotkeys = data.get("hotkeys", {})
    project.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
    project.updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None

    # Load annotations (legacy)
    annotations_data = data.get("annotations", {})
    for frame_id, ann_data in annotations_data.items():
      project.annotations[frame_id] = FrameAnnotation(
        frame_id=ann_data["frame_id"],
        annotations=ann_data["annotations"],
        confidence=ann_data.get("confidence", 1.0),
        annotated_at=datetime.fromisoformat(ann_data["annotated_at"]) if ann_data.get("annotated_at") else None,
      )

    warnings.warn(
      "Loading legacy annotations. Use DB-backed services (AnnotationService, SettingsService) instead.",
      DeprecationWarning,
      stacklevel=2,
    )

    return project
