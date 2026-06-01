from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClassroomCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ClassroomOut(BaseModel):
    id: int
    name: str
    created_at: datetime


class StudentCreate(BaseModel):
    student_no: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=64)


class StudentOut(BaseModel):
    id: int
    student_no: str
    name: str
    created_at: datetime


class SessionCreate(BaseModel):
    classroom_id: int
    title: str = Field(min_length=1, max_length=128)


class SessionOut(BaseModel):
    id: int
    classroom_id: int
    title: str
    started_at: datetime
    ended_at: Optional[datetime]


AttendanceStatus = Literal["present", "late", "absent"]


class AttendanceUpsert(BaseModel):
    session_id: int
    student_id: int
    status: AttendanceStatus = "present"
    ts: Optional[datetime] = None


class AttentionIn(BaseModel):
    session_id: int
    student_id: int
    score_attention: float = Field(ge=0, le=100)
    score_expression: float = Field(ge=0, le=100)
    score_headpose: float = Field(ge=0, le=100)
    score_behavior: float = Field(ge=0, le=100)

    ear: Optional[float] = None
    mar: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    ts: Optional[datetime] = None

    # MiniFASNet 活体检测结果
    is_live: Optional[bool] = True    # True=真人, False=假脸(照片/视频)
    live_score: Optional[float] = None  # 活体置信度 0.0~1.0


class AttentionOut(BaseModel):
    id: int
    session_id: int
    student_id: int
    score_attention: float
    score_expression: float
    score_headpose: float
    score_behavior: float
    ear: Optional[float]
    mar: Optional[float]
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    ts: datetime

    # MiniFASNet 活体检测结果
    is_live: Optional[bool] = True
    live_score: Optional[float] = None


class WsEvent(BaseModel):
    type: Literal["attention", "attendance", "session"]
    payload: dict
    ts: datetime

