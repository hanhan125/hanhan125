from __future__ import annotations

from datetime import datetime

from sqlmodel import Session, desc, select

from app.api.session_context import get_session_roster, pop_session_roster
from app.api.ws_hub import ws_hub
from app.storage.models import Attendance, AttentionSample, Student


async def finalize_session_attendance(
    session_id: int,
    classroom_id: int,
    db: Session,
) -> dict[str, int]:
    """
    Mark roster students without present proof as absent when class ends.
    Present = latest attendance is 'present' OR has any attention sample in session.
    """
    roster = pop_session_roster(session_id)
    if not roster:
        roster = get_session_roster(session_id)
    if not roster:
        students = list(db.exec(select(Student).order_by(Student.id.asc())).all())
        roster = [int(s.id) for s in students]

    rows = list(
        db.exec(
            select(Attendance)
            .where(Attendance.session_id == session_id)
            .order_by(desc(Attendance.ts))
            .limit(2000)
        ).all()
    )
    latest_att: dict[int, Attendance] = {}
    for r in rows:
        if r.student_id not in latest_att:
            latest_att[r.student_id] = r

    present_ids: set[int] = set()
    for sid, rec in latest_att.items():
        if rec.status == "present":
            present_ids.add(int(sid))

    sample_rows = list(
        db.exec(
            select(AttentionSample.student_id)
            .where(AttentionSample.session_id == session_id)
            .distinct()
        ).all()
    )
    for sid in sample_rows:
        if sid is not None:
            present_ids.add(int(sid))

    now = datetime.utcnow()
    marked_absent = 0
    for sid in roster:
        if int(sid) in present_ids:
            continue
        a = Attendance(session_id=session_id, student_id=int(sid), status="absent", ts=now)
        db.add(a)
        marked_absent += 1
        await ws_hub.broadcast(
            classroom_id,
            "attendance",
            {
                "session_id": session_id,
                "student_id": int(sid),
                "status": "absent",
                "ts": now.isoformat(),
            },
        )

    if marked_absent:
        db.commit()

    present_count = sum(1 for sid in roster if int(sid) in present_ids)
    absent_count = len(roster) - present_count

    return {
        "roster_count": len(roster),
        "present": present_count,
        "absent": absent_count,
        "late": 0,
        "marked_absent": marked_absent,
    }
