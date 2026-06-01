from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Classroom(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_no: str = Field(index=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class SessionRecord(SQLModel, table=True):
    """A teaching session / class period."""

    id: Optional[int] = Field(default=None, primary_key=True)
    classroom_id: int = Field(index=True)
    title: str
    started_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    ended_at: Optional[datetime] = Field(default=None, index=True)


class Attendance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True)
    student_id: int = Field(index=True)
    status: str = Field(default="present", index=True)  # present/late/absent
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)


class AttentionSample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True)
    student_id: int = Field(index=True)

    # Scores are 0-100.
    score_attention: float = Field(index=True)
    score_expression: float
    score_headpose: float
    score_behavior: float

    # Evidence for explainability / charts
    ear: Optional[float] = None
    mar: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    # MiniFASNet 活体检测结果
    is_live: Optional[bool] = Field(default=True)
    live_score: Optional[float] = Field(default=None)

    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

