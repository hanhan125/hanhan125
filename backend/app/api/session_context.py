"""In-memory roster per teaching session (who should be counted for attendance)."""

from __future__ import annotations

_session_rosters: dict[int, list[int]] = {}


def register_session_roster(session_id: int, student_ids: list[int]) -> None:
    _session_rosters[int(session_id)] = [int(x) for x in student_ids]


def get_session_roster(session_id: int) -> list[int]:
    return list(_session_rosters.get(int(session_id), []))


def pop_session_roster(session_id: int) -> list[int]:
    return list(_session_rosters.pop(int(session_id), []))
