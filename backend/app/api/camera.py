from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.demo import demo_runner
from app.api.session_context import register_session_roster
from app.api.ws_hub import ws_hub
from app.storage.db import get_session
from app.storage.models import Classroom, SessionRecord, Student


router = APIRouter(prefix="/api/camera", tags=["camera"])


class CameraStartIn(BaseModel):
    classroom_id: Optional[int] = None
    camera_index: int = 0
    api_base: str = "http://127.0.0.1:8001"
    max_students: int = 8


class _CameraRunner:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.session_id: int | None = None
        self.classroom_id: int | None = None
        self.student_ids: list[int] = []

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.process = None
        self.session_id = None
        self.classroom_id = None
        self.student_ids = []


camera_runner = _CameraRunner()


def _backend_root() -> Path:
    # .../backend/app/api/camera.py -> .../backend
    return Path(__file__).resolve().parents[2]


def _python_executable() -> str:
    root = _backend_root()
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _cleanup_stale_camera_processes() -> None:
    """
    Best-effort cleanup for orphan camera processes from previous runs.
    """
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*realtime_camera.py*' } | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        _ = proc.returncode
    except Exception:
        pass


@router.get("/status")
def camera_status() -> dict:
    return {
        "running": camera_runner.is_running(),
        "pid": camera_runner.process.pid if camera_runner.process else None,
        "session_id": camera_runner.session_id,
        "classroom_id": camera_runner.classroom_id,
        "student_ids": camera_runner.student_ids,
    }


@router.post("/stop")
def camera_stop() -> dict:
    camera_runner.stop()
    return {"ok": True}


def _resolve_classroom_id(payload: CameraStartIn, db: Session) -> int:
    classroom_id = payload.classroom_id
    if classroom_id is None:
        first_room = db.exec(select(Classroom).order_by(Classroom.id.asc())).first()
        if not first_room:
            first_room = Classroom(name="示范教室 A")
            db.add(first_room)
            db.commit()
            db.refresh(first_room)
        classroom_id = first_room.id
    return int(classroom_id)


def _load_student_ids(db: Session, max_students: int, *, seed_if_empty: bool) -> list[int]:
    if max_students < 1:
        raise HTTPException(status_code=400, detail="max_students must be >= 1")
    cap = min(max_students, 20)
    students = list(db.exec(select(Student).order_by(Student.id.asc())).all())
    if not students and seed_if_empty:
        for student_no, name in [("2026001", "张三"), ("2026002", "李四"), ("2026003", "王五"), ("2026004", "赵六")]:
            db.add(Student(student_no=student_no, name=name))
        db.commit()
        students = list(db.exec(select(Student).order_by(Student.id.asc())).all())
    if not students:
        raise HTTPException(status_code=400, detail="暂无学生，请先在网页添加学生")
    return [int(s.id) for s in students[:cap]]


def _spawn_camera_process(
    *,
    classroom_id: int,
    session_id: int,
    student_ids: list[int],
    payload: CameraStartIn,
) -> subprocess.Popen:
    root = _backend_root()
    script = root / "tools" / "realtime_camera.py"
    if not script.exists():
        raise HTTPException(status_code=500, detail="camera script not found")
    cmd = [
        _python_executable(),
        str(script),
        "--api-base",
        payload.api_base,
        "--classroom-id",
        str(classroom_id),
        "--student-ids",
        ",".join(str(x) for x in student_ids),
        "--session-id",
        str(session_id),
        "--max-faces",
        str(len(student_ids)),
        "--camera",
        str(payload.camera_index),
        "--stop-demo",
    ]
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return subprocess.Popen(cmd, cwd=str(root), creationflags=creationflags)


def _camera_response(session_id: int, classroom_id: int, student_ids: list[int], message: str, proc: subprocess.Popen) -> dict:
    camera_runner.process = proc
    camera_runner.session_id = session_id
    camera_runner.classroom_id = classroom_id
    camera_runner.student_ids = student_ids
    return {
        "ok": True,
        "message": message,
        "pid": proc.pid,
        "session_id": session_id,
        "classroom_id": classroom_id,
        "student_ids": student_ids,
        "student_count": len(student_ids),
    }


@router.post("/start")
async def camera_start(payload: CameraStartIn, db: Session = Depends(get_session)) -> dict:
    await demo_runner.stop()
    camera_runner.stop()
    _cleanup_stale_camera_processes()

    classroom_id = _resolve_classroom_id(payload, db)
    student_ids = _load_student_ids(db, payload.max_students, seed_if_empty=True)

    session = SessionRecord(classroom_id=classroom_id, title="开始上课（摄像头）")
    db.add(session)
    db.commit()
    db.refresh(session)
    await ws_hub.broadcast(classroom_id, "session", {"event": "started", "session": session.model_dump()})
    register_session_roster(int(session.id), student_ids)

    proc = _spawn_camera_process(
        classroom_id=classroom_id,
        session_id=int(session.id),
        student_ids=student_ids,
        payload=payload,
    )
    return _camera_response(
        int(session.id),
        classroom_id,
        student_ids,
        "camera started (multi-student mode)",
        proc,
    )


@router.post("/restart")
async def camera_restart(payload: CameraStartIn, db: Session = Depends(get_session)) -> dict:
    """Restart camera subprocess with latest student list; keep current session if running."""
    await demo_runner.stop()
    session_id = camera_runner.session_id
    classroom_id = camera_runner.classroom_id or _resolve_classroom_id(payload, db)
    camera_runner.stop()
    _cleanup_stale_camera_processes()

    if session_id is None:
        raise HTTPException(status_code=400, detail="未在上课中，请先点击「开始上课」")

    student_ids = _load_student_ids(db, payload.max_students, seed_if_empty=False)
    register_session_roster(int(session_id), student_ids)
    proc = _spawn_camera_process(
        classroom_id=classroom_id,
        session_id=int(session_id),
        student_ids=student_ids,
        payload=payload,
    )
    return _camera_response(
        int(session_id),
        classroom_id,
        student_ids,
        "camera restarted with updated students",
        proc,
    )

