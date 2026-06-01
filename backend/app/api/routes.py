from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, desc, select

from app.api.ws_hub import ws_hub
from app.services.attendance_finalize import finalize_session_attendance
from app.schemas.dto import (
    AttentionIn,
    AttentionOut,
    AttendanceUpsert,
    ClassroomCreate,
    ClassroomOut,
    SessionCreate,
    SessionOut,
    StudentCreate,
    StudentOut,
)
from app.storage.db import get_session
from app.storage.models import Attendance, AttentionSample, Classroom, SessionRecord, Student


router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/classrooms", response_model=ClassroomOut)
def create_classroom(payload: ClassroomCreate, db: Session = Depends(get_session)) -> Classroom:
    c = Classroom(name=payload.name)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@router.get("/classrooms", response_model=list[ClassroomOut])
def list_classrooms(db: Session = Depends(get_session)) -> list[Classroom]:
    return list(db.exec(select(Classroom).order_by(desc(Classroom.created_at))).all())


@router.post("/students", response_model=StudentOut)
def create_student(payload: StudentCreate, db: Session = Depends(get_session)) -> Student:
    existing = db.exec(select(Student).where(Student.student_no == payload.student_no)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"学号已存在: {payload.student_no}")
    s = Student(student_no=payload.student_no, name=payload.name)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


@router.get("/students", response_model=list[StudentOut])
def list_students(db: Session = Depends(get_session)) -> list[Student]:
    return list(db.exec(select(Student).order_by(desc(Student.created_at))).all())


@router.post("/sessions", response_model=SessionOut)
async def create_session(payload: SessionCreate, db: Session = Depends(get_session)) -> SessionRecord:
    classroom = db.get(Classroom, payload.classroom_id)
    if not classroom:
        raise HTTPException(status_code=404, detail="classroom not found")
    s = SessionRecord(classroom_id=payload.classroom_id, title=payload.title)
    db.add(s)
    db.commit()
    db.refresh(s)
    await ws_hub.broadcast(payload.classroom_id, "session", {"event": "started", "session": s.model_dump()})
    return s


@router.post("/sessions/{session_id}/end")
async def end_session(session_id: int, db: Session = Depends(get_session)) -> dict:
    s = db.get(SessionRecord, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    summary: dict[str, int] = {"roster_count": 0, "present": 0, "absent": 0, "late": 0, "marked_absent": 0}
    if s.ended_at is None:
        summary = await finalize_session_attendance(session_id, s.classroom_id, db)
        s.ended_at = datetime.utcnow()
        db.add(s)
        db.commit()
        db.refresh(s)
        await ws_hub.broadcast(s.classroom_id, "session", {"event": "ended", "session": s.model_dump()})
    return {**s.model_dump(), "attendance_summary": summary}


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(classroom_id: Optional[int] = None, db: Session = Depends(get_session)) -> list[SessionRecord]:
    stmt = select(SessionRecord).order_by(desc(SessionRecord.started_at))
    if classroom_id is not None:
        stmt = stmt.where(SessionRecord.classroom_id == classroom_id)
    return list(db.exec(stmt).all())


@router.post("/attendance", response_model=dict)
async def upsert_attendance(payload: AttendanceUpsert, db: Session = Depends(get_session)) -> dict:
    sess = db.get(SessionRecord, payload.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    student = db.get(Student, payload.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="student not found")
    ts = payload.ts or datetime.utcnow()
    a = Attendance(session_id=payload.session_id, student_id=payload.student_id, status=payload.status, ts=ts)
    db.add(a)
    db.commit()
    await ws_hub.broadcast(
        sess.classroom_id,
        "attendance",
        {"session_id": payload.session_id, "student_id": payload.student_id, "status": payload.status, "ts": ts.isoformat()},
    )
    return {"ok": True}


@router.get("/attendance/latest", response_model=list[dict])
def latest_attendance(session_id: int, db: Session = Depends(get_session)) -> list[dict]:
    # One latest attendance status per student.
    rows = list(
        db.exec(
            select(Attendance)
            .where(Attendance.session_id == session_id)
            .order_by(desc(Attendance.ts))
            .limit(1000)
        ).all()
    )
    latest: dict[int, Attendance] = {}
    for r in rows:
        if r.student_id not in latest:
            latest[r.student_id] = r
    return [
        {
            "id": a.id,
            "session_id": a.session_id,
            "student_id": a.student_id,
            "status": a.status,
            "ts": a.ts.isoformat(),
        }
        for a in latest.values()
    ]


@router.post("/attention", response_model=AttentionOut)
async def ingest_attention(payload: AttentionIn, db: Session = Depends(get_session)) -> AttentionSample:
    sess = db.get(SessionRecord, payload.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    student = db.get(Student, payload.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="student not found")
    ts = payload.ts or datetime.utcnow()
    sample = AttentionSample(
        session_id=payload.session_id,
        student_id=payload.student_id,
        score_attention=payload.score_attention,
        score_expression=payload.score_expression,
        score_headpose=payload.score_headpose,
        score_behavior=payload.score_behavior,
        ear=payload.ear,
        mar=payload.mar,
        yaw=payload.yaw,
        pitch=payload.pitch,
        roll=payload.roll,
        is_live=payload.is_live,
        live_score=payload.live_score,
        ts=ts,
    )
    db.add(sample)
    db.commit()
    db.refresh(sample)
    await ws_hub.broadcast(
        sess.classroom_id,
        "attention",
        {
            "id": sample.id,
            "session_id": sample.session_id,
            "student_id": sample.student_id,
            "score_attention": sample.score_attention,
            "score_expression": sample.score_expression,
            "score_headpose": sample.score_headpose,
            "score_behavior": sample.score_behavior,
            "ear": sample.ear,
            "mar": sample.mar,
            "yaw": sample.yaw,
            "pitch": sample.pitch,
            "roll": sample.roll,
            "is_live": sample.is_live,
            "live_score": sample.live_score,
            "ts": sample.ts.isoformat(),
        },
    )
    return sample


@router.get("/attention/latest", response_model=list[AttentionOut])
def latest_attention(session_id: int, db: Session = Depends(get_session)) -> list[AttentionSample]:
    # One latest sample per student (simple approach for demo).
    samples = list(
        db.exec(
            select(AttentionSample)
            .where(AttentionSample.session_id == session_id)
            .order_by(desc(AttentionSample.ts))
            .limit(500)
        ).all()
    )
    latest: dict[int, AttentionSample] = {}
    for s in samples:
        if s.student_id not in latest:
            latest[s.student_id] = s
    return list(latest.values())


@router.get("/attention/history", response_model=list[AttentionOut])
def attention_history(
    session_id: int,
    student_id: int,
    limit: int = 300,
    db: Session = Depends(get_session),
) -> list[AttentionSample]:
    limit = max(10, min(limit, 2000))
    return list(
        db.exec(
            select(AttentionSample)
            .where(AttentionSample.session_id == session_id, AttentionSample.student_id == student_id)
            .order_by(desc(AttentionSample.ts))
            .limit(limit)
        ).all()
    )[::-1]

