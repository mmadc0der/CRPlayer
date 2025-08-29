from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, validator


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class FrameQuery(BaseModel):
    session_id: str
    idx: int

    @validator('session_id', pre=True)
    def _trim_session_id(cls, v):
        return v.strip() if isinstance(v, str) else v


class ImageQuery(BaseModel):
    session_id: str
    idx: int

    @validator('session_id', pre=True)
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

    @validator('frame_id', always=True)
    def _validate_frame_id_or_idx(cls, v, values):
        if (v is None) and (values.get('frame_idx') is None):
            raise ValueError('Either frame_id or frame_idx must be provided')
        return v

    @validator('session_id', pre=True)
    def _trim_session_id(cls, v):
        return v.strip() if isinstance(v, str) else v


class SaveSingleLabelRequest(BaseModel):
    session_id: str
    dataset_id: int
    frame_id: Optional[str] = None
    frame_idx: Optional[int] = None
    class_id: int
    override_settings: Optional[Dict[str, Any]] = None

    @validator('frame_id', always=True)
    def _validate_frame_id_or_idx(cls, v, values):
        if (v is None) and (values.get('frame_idx') is None):
            raise ValueError('Either frame_id or frame_idx must be provided')
        return v


class SaveMultilabelRequest(BaseModel):
    session_id: str
    dataset_id: int
    frame_id: Optional[str] = None
    frame_idx: Optional[int] = None
    class_ids: List[int]
    override_settings: Optional[Dict[str, Any]] = None

    @validator('frame_id', always=True)
    def _validate_frame_id_or_idx(cls, v, values):
        if (v is None) and (values.get('frame_idx') is None):
            raise ValueError('Either frame_id or frame_idx must be provided')
        return v


class UpsertDatasetSessionSettingsRequest(BaseModel):
    dataset_id: int
    session_id: str
    settings: Dict[str, Any]

    @validator('session_id', pre=True)
    def _trim_session_id(cls, v):
        return v.strip() if isinstance(v, str) else v
