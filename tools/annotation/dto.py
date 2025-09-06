from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel

try:
  # Pydantic v2
  from pydantic import field_validator, model_validator  # type: ignore
except Exception:  # pragma: no cover
  # Fallback for v1 compatibility
  from pydantic import validator as field_validator  # type: ignore

  def model_validator(*args, **kwargs):  # type: ignore

    def deco(fn):
      return fn

    return deco


class ErrorResponse(BaseModel):
  code: str
  message: str
  details: Optional[Dict[str, Any]] = None


class FrameQuery(BaseModel):
  session_id: str
  idx: int

  @field_validator("session_id", mode="before")
  def _trim_session_id(cls, v):
    return v.strip() if isinstance(v, str) else v


class ImageQuery(BaseModel):
  session_id: str
  idx: int

  @field_validator("session_id", mode="before")
  def _trim_session_id(cls, v):
    return v.strip() if isinstance(v, str) else v


# ---------------- DB-backed annotation DTOs ----------------


class SaveRegressionRequest(BaseModel):
  session_id: str
  dataset_id: int
  frame_id: Optional[str] = None
  frame_idx: Optional[int] = None
  value: float
  override_settings: Optional[Dict[str, Any]] = None

  @model_validator(mode="after")
  def _validate_frame_id_or_idx(self):
    if (self.frame_id is None) and (self.frame_idx is None):
      raise ValueError("Either frame_id or frame_idx must be provided")
    return self

  @field_validator("session_id", mode="before")
  def _trim_session_id(cls, v):
    return v.strip() if isinstance(v, str) else v


class SaveSingleLabelRequest(BaseModel):
  session_id: str
  dataset_id: int
  frame_id: Optional[str] = None
  frame_idx: Optional[int] = None
  class_id: Optional[int] = None
  category_name: Optional[str] = None
  override_settings: Optional[Dict[str, Any]] = None

  @model_validator(mode="after")
  def _validate_frame_id_or_idx(self):
    if (self.frame_id is None) and (self.frame_idx is None):
      raise ValueError("Either frame_id or frame_idx must be provided")
    return self

  @model_validator(mode="after")
  def _validate_class_or_category(self):
    if self.class_id is None and (not self.category_name or not str(self.category_name).strip()):
      raise ValueError("Either class_id or category_name must be provided")
    return self


class SaveMultilabelRequest(BaseModel):
  session_id: str
  dataset_id: int
  frame_id: Optional[str] = None
  frame_idx: Optional[int] = None
  class_ids: List[int]
  category_names: Optional[List[str]] = None
  override_settings: Optional[Dict[str, Any]] = None

  @model_validator(mode="after")
  def _validate_frame_id_or_idx(self):
    if (self.frame_id is None) and (self.frame_idx is None):
      raise ValueError("Either frame_id or frame_idx must be provided")
    return self


class UpsertDatasetSessionSettingsRequest(BaseModel):
  dataset_id: int
  session_id: str
  settings: Dict[str, Any]

  @field_validator("session_id", mode="before")
  def _trim_session_id(cls, v):
    return v.strip() if isinstance(v, str) else v
