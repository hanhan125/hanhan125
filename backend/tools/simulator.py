from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone

import httpx


API = "http://127.0.0.1:8000"


def utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


async def main() -> None:
    async with httpx.AsyncClient(base_url=API, timeout=10.0) as client:
        # Create classroom
        c = (await client.post("/api/classrooms", json={"name": "示范教室 A"})).json()
        classroom_id = c["id"]

        # Create students
        names = [("2026001", "张三"), ("2026002", "李四"), ("2026003", "王五"), ("2026004", "赵六")]
        students = []
        for student_no, name in names:
            s = (await client.post("/api/students", json={"student_no": student_no, "name": name})).json()
            students.append(s)

        # Start session
        session = (
            await client.post("/api/sessions", json={"classroom_id": classroom_id, "title": "课堂演示"})
        ).json()
        session_id = session["id"]

        # Mark attendance once
        for s in students:
            await client.post(
                "/api/attendance",
                json={"session_id": session_id, "student_id": s["id"], "status": "present", "ts": utcnow_iso()},
            )

        # Stream attention samples
        base = {s["id"]: random.uniform(60, 90) for s in students}
        print(f"Simulating classroom_id={classroom_id}, session_id={session_id} ...")

        for _ in range(10_000):
            for s in students:
                sid = s["id"]
                drift = random.uniform(-3.5, 3.5)
                base[sid] = max(0, min(100, base[sid] + drift))

                # Fake sub-scores and evidence
                expr = max(0, min(100, base[sid] + random.uniform(-6, 6)))
                head = max(0, min(100, base[sid] + random.uniform(-8, 8)))
                beh = max(0, min(100, base[sid] + random.uniform(-10, 10)))
                yaw = random.uniform(-25, 25) if base[sid] > 50 else random.uniform(-60, 60)
                ear = random.uniform(0.18, 0.32) if base[sid] > 30 else random.uniform(0.10, 0.22)
                mar = random.uniform(0.25, 0.5)

                await client.post(
                    "/api/attention",
                    json={
                        "session_id": session_id,
                        "student_id": sid,
                        "score_attention": float(base[sid]),
                        "score_expression": float(expr),
                        "score_headpose": float(head),
                        "score_behavior": float(beh),
                        "ear": float(ear),
                        "mar": float(mar),
                        "yaw": float(yaw),
                        "pitch": float(random.uniform(-20, 20)),
                        "roll": float(random.uniform(-15, 15)),
                        "ts": utcnow_iso(),
                    },
                )
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())

