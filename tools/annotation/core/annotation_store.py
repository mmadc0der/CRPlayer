"""
DEPRECATED: Filesystem-based AnnotationStore.

This module previously handled annotation CRUD via annotations.json on disk.
The project is now fully DB-backed. Use `services.annotation_service.AnnotationService` instead.
All methods here either no-op or raise to prevent accidental filesystem writes.
"""

from typing import Dict, Any, Optional
import warnings

from core.session_manager import SessionManager


class AnnotationStore:
  """DEPRECATED shim for legacy filesystem-based annotation store.

    Use `services.annotation_service.AnnotationService` for DB-backed CRUD.
    """

  def __init__(self, session_manager: SessionManager):
    warnings.warn(
      "AnnotationStore is deprecated. Use services.annotation_service.AnnotationService instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    self.session_manager = session_manager

  def load_session_project(self, session_dir: str, project_name: str) -> Dict[str, Any]:
    warnings.warn(
      "load_session_project is deprecated and no longer supported in DB-backed flow.",
      DeprecationWarning,
      stacklevel=2,
    )
    raise NotImplementedError("Use Dataset + AnnotationService instead of projects")

  def save_annotation(self, frame_id: str, annotation_data: Dict[str, Any], confidence: float = 1.0):
    raise NotImplementedError("Use AnnotationService.save_* endpoints (DB-backed)")

  def get_annotation(self, frame_id: str) -> Optional[Dict[str, Any]]:
    warnings.warn(
      "get_annotation (filesystem) is deprecated.",
      DeprecationWarning,
      stacklevel=2,
    )
    return None

  def add_category(self, category: str):
    raise NotImplementedError("Categories via projects are deprecated. Use dataset metadata if needed.")

  def save_project(self, session_dir: str, project_name: str, project: Dict[str, Any]):
    warnings.warn(
      "save_project is deprecated and has no effect.",
      DeprecationWarning,
      stacklevel=2,
    )

  def get_project_stats(self, total_frames: int) -> Dict[str, Any]:
    warnings.warn(
      "get_project_stats is deprecated. Use dataset progress endpoints.",
      DeprecationWarning,
      stacklevel=2,
    )
    return {}
