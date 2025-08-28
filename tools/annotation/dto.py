from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class FrameQuery(BaseModel):
    session_path: str
    project_name: Optional[str] = Field(default='default')
    idx: int


class ImageQuery(BaseModel):
    session_path: str
    idx: int


class SaveAnnotationRequest(BaseModel):
    session_path: str
    project_name: str = Field(default='default')
    frame_id: Optional[str] = None
    frame_idx: Optional[int] = None
    annotations: Dict[str, Any]
    confidence: Optional[float] = None

    @validator('frame_id', always=True)
    def validate_frame_id_or_idx(cls, v, values):
        if (v is None) and (values.get('frame_idx') is None):
            raise ValueError('Either frame_id or frame_idx must be provided')
        return v
