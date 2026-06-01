from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlmodel import Session, desc, select

from app.api.session_context import register_session_roster
from app.api.ws_hub import ws_hub
from app.storage.db import engine, get_session
from app.storage.models import Attendance, AttentionSample, Classroom, SessionRecord, Student


router = APIRouter(prefix="/api/demo", tags=["demo"])


class _DemoRunner:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._classroom_id: Optional[int] = None
        self._session_id: Optional[int] = None

    def status(self) -> dict:
        return {
            "running": self._task is not None and not self._task.done(),
            "classroom_id": self._classroom_id,
            "session_id": self._session_id,
        }

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        self._task = None
        self._classroom_id = None
        self._session_id = None

    def start(self, classroom_id: int, session_id: int, student_ids: list[int]) -> None:
        # Stop previous runner (if any).
        if self._task and not self._task.done():
            self._task.cancel()
        self._classroom_id = classroom_id
        self._session_id = session_id
        self._task = asyncio.create_task(self._loop(classroom_id, session_id, student_ids))

    async def _loop(self, classroom_id: int, session_id: int, student_ids: list[int]) -> None:
        base = {sid: random.uniform(60, 90) for sid in student_ids}
        while True:
            now = datetime.utcnow()
            batch: list[AttentionSample] = []
            for sid in student_ids:
                drift = random.uniform(-3.5, 3.5)
                base[sid] = max(0, min(100, base[sid] + drift))

                expr = max(0, min(100, base[sid] + random.uniform(-6, 6)))
                head = max(0, min(100, base[sid] + random.uniform(-8, 8)))
                beh = max(0, min(100, base[sid] + random.uniform(-10, 10)))

                sample = AttentionSample(
                    session_id=session_id,
                    student_id=sid,
                    score_attention=float(base[sid]),
                    score_expression=float(expr),
                    score_headpose=float(head),
                    score_behavior=float(beh),
                    ear=float(random.uniform(0.18, 0.32)) if base[sid] > 30 else float(random.uniform(0.10, 0.22)),
                    mar=float(random.uniform(0.25, 0.50)),
                    yaw=float(random.uniform(-25, 25)) if base[sid] > 50 else float(random.uniform(-60, 60)),
                    pitch=float(random.uniform(-20, 20)),
                    roll=float(random.uniform(-15, 15)),
                    ts=now,
                )
                batch.append(sample)

            # Persist so REST polling (miniprogram) sees the same data as WebSocket.
            with Session(engine) as db:
                for sample in batch:
                    db.add(sample)
                db.commit()
                for sample in batch:
                    db.refresh(sample)

            for sample in batch:
                payload = {
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
                    "ts": sample.ts.isoformat(),
                }
                await ws_hub.broadcast(classroom_id, "attention", payload)

            await asyncio.sleep(1.0)


demo_runner = _DemoRunner()


@router.get("/status")
def demo_status() -> dict:
    return demo_runner.status()


@router.post("/stop")
async def demo_stop() -> dict:
    await demo_runner.stop()
    return {"ok": True}


@router.post("/start")
async def demo_start(db: Session = Depends(get_session)) -> dict:
    # Ensure at least one classroom exists
    classroom = db.exec(select(Classroom).order_by(desc(Classroom.created_at))).first()
    if not classroom:
        classroom = Classroom(name="示范教室 A")
        db.add(classroom)
        db.commit()
        db.refresh(classroom)

    # Ensure demo students exist (idempotent-ish: create if not enough)
    students = list(db.exec(select(Student).order_by(desc(Student.created_at))).all())
    if len(students) < 4:
        for student_no, name in [("2026001", "张三"), ("2026002", "李四"), ("2026003", "王五"), ("2026004", "赵六")]:
            db.add(Student(student_no=student_no, name=name))
        db.commit()
        students = list(db.exec(select(Student).order_by(desc(Student.created_at))).all())

    # Start a new session
    sess = SessionRecord(classroom_id=classroom.id, title="课堂演示（自动）")
    db.add(sess)
    db.commit()
    db.refresh(sess)

    await ws_hub.broadcast(classroom.id, "session", {"event": "started", "session": sess.model_dump()})

    # Mark everyone present
    for s in students[:10]:
        a = Attendance(session_id=sess.id, student_id=s.id, status="present", ts=datetime.utcnow())
        db.add(a)
        await ws_hub.broadcast(
            classroom.id,
            "attendance",
            {"session_id": sess.id, "student_id": s.id, "status": "present", "ts": a.ts.isoformat()},
        )
    db.commit()

    roster_ids = [int(s.id) for s in students[:10]]
    register_session_roster(int(sess.id), roster_ids)
    demo_runner.start(classroom.id, sess.id, roster_ids)
    return {
        "ok": True,
        "classroom_id": classroom.id,
        "session_id": sess.id,
        "student_ids": roster_ids,
        "student_count": len(roster_ids),
    }

