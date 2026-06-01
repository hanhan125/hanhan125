"""
Real webcam -> face mesh -> liveness check -> attention metrics -> POST /api/attention

集成 MiniFASNet 静默活体检测：
  - 每张人脸先通过活体检测，假脸（照片/视频回放）会被跳过
  - 假脸不会签到、不会提交注意力数据
  - 显示窗口标注活体/假脸状态

Usage (from backend/, after: pip install -r requirements.txt -r requirements-camera.txt):
  .\\.venv\\Scripts\\python tools\\realtime_camera.py --classroom-id 3 --student-ids 1,2,3,4

Optional: stop fake demo stream first (if running):
  curl -X POST http://127.0.0.1:8001/api/demo/stop
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import httpx
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# ============================================================================
# 导入活体检测模块
# ============================================================================
from tools.liveness_detector import LivenessDetector

# MediaPipe Face Mesh landmark indices (468) — common EAR sets
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth vertical / horizontal (rough MAR proxy)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
NOSE_TIP = 1


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _lm_xy(lm, w: int, h: int) -> tuple[float, float]:
    # FaceLandmarker returns normalized landmarks with x,y in [0,1]
    return float(lm.x * w), float(lm.y * h)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks, idxs: list[int], w: int, h: int) -> float:
    p = [_lm_xy(landmarks[i], w, h) for i in idxs]
    v1 = _dist(p[1], p[5])
    v2 = _dist(p[2], p[4])
    horiz = _dist(p[0], p[3])
    return (v1 + v2) / (2.0 * horiz + 1e-6)


def mouth_aspect_ratio(landmarks, w: int, h: int) -> float:
    top = _lm_xy(landmarks[MOUTH_TOP], w, h)
    bottom = _lm_xy(landmarks[MOUTH_BOTTOM], w, h)
    left = _lm_xy(landmarks[MOUTH_LEFT], w, h)
    right = _lm_xy(landmarks[MOUTH_RIGHT], w, h)
    vert = _dist(top, bottom)
    horiz = _dist(left, right)
    return vert / (horiz + 1e-6)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_from_ear(ear: float) -> float:
    # Typical open eye EAR ~0.25–0.35 (image-dependent); closed lower.
    lo, hi = 0.18, 0.32
    return 100.0 * clamp01((ear - lo) / (hi - lo))


def score_from_mar(mar: float) -> float:
    # Larger MAR => more open mouth / yawning / talking => lower expression focus
    # Typical closed ~0.02–0.06; open wider => 0.1+
    if mar < 0.05:
        return 95.0
    if mar < 0.09:
        return 80.0
    if mar < 0.14:
        return 55.0
    return 30.0


def score_head_center(nose_xy: tuple[float, float], w: int, h: int) -> float:
    cx, cy = w * 0.5, h * 0.5
    nx, ny = nose_xy
    # normalized offset from center
    off = math.hypot((nx - cx) / (w + 1e-6), (ny - cy) / (h + 1e-6))
    # 0 at center -> 1, large offset -> 0
    return 100.0 * clamp01(1.0 - off * 2.2)


def score_behavior_still(speed: float) -> float:
    # speed: nose movement in normalized coords per frame (smoothed)
    # small motion good; large jitter bad
    if speed < 0.002:
        return 92.0
    if speed < 0.006:
        return 75.0
    if speed < 0.012:
        return 55.0
    return 35.0


if TYPE_CHECKING:
    from PIL import ImageFont


@lru_cache(maxsize=8)
def _cn_font(size: int):
    """Load a Chinese-capable TrueType font (OpenCV putText cannot render CJK)."""
    from PIL import ImageFont

    win = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        win / "Fonts" / "msyh.ttc",
        win / "Fonts" / "msyhbd.ttc",
        win / "Fonts" / "simhei.ttf",
        win / "Fonts" / "simsun.ttc",
        win / "Fonts" / "msyh.ttf",
        Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansSC-Regular.otf",
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_cn_text_block(
    frame,
    lines: list[str],
    origin: tuple[int, int],
    *,
    color_bgr: tuple[int, int, int] = (20, 240, 20),
    font_size: int = 18,
    line_gap: int = 22,
) -> None:
    """Draw multiple Chinese/ASCII lines on a BGR frame via Pillow."""
    from PIL import Image as PilImage
    from PIL import ImageDraw

    x, y = origin
    pil = PilImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _cn_font(font_size)
    fill = (color_bgr[2], color_bgr[1], color_bgr[0])
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_gap), line, font=font, fill=fill)
    frame[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def eye_open_label(ear: float) -> str:
    """Align with score_from_ear thresholds (lo=0.18, hi=0.32)."""
    if ear < 0.20:
        return "闭"
    if ear < 0.24:
        return "半"
    return "开"


def _draw_points(frame, landmarks, idxs: list[int], w: int, h: int, color: tuple[int, int, int], radius: int = 2) -> None:
    for i in idxs:
        x, y = _lm_xy(landmarks[i], w, h)
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def _draw_polyline(frame, landmarks, idxs: list[int], w: int, h: int, color: tuple[int, int, int]) -> None:
    pts = [(_lm_xy(landmarks[i], w, h)) for i in idxs]
    arr = [(int(p[0]), int(p[1])) for p in pts]
    if len(arr) >= 2:
        cv2.polylines(frame, [np.array(arr, dtype=np.int32)], True, color, 1, cv2.LINE_AA)


def draw_face_overlay(
    frame,
    lm,
    w: int,
    h: int,
    *,
    student_no: str,
    name: str,
    score_attention: float,
    ear_l: float,
    ear_r: float,
    is_live: bool = True,
    live_score: float = 1.0,
) -> None:
    """Draw landmark points + 学号/姓名/专注度/左右眼状态 + 活体检测结果 on OpenCV window."""
    nose = _lm_xy(lm[NOSE_TIP], w, h)
    nx, ny = int(nose[0]), int(nose[1])

    # 活体检测通过 → 绿色；假脸 → 红色
    if is_live:
        c_left = (255, 200, 0)
        c_right = (0, 220, 255)
        c_mouth = (220, 0, 220)
        c_nose = (0, 220, 0)
        text_color = (20, 240, 20)      # 绿色文字
    else:
        c_left = (0, 0, 255)
        c_right = (0, 0, 255)
        c_mouth = (0, 0, 255)
        c_nose = (0, 0, 255)
        text_color = (0, 0, 220)        # 红色文字

    _draw_polyline(frame, lm, LEFT_EYE, w, h, c_left)
    _draw_polyline(frame, lm, RIGHT_EYE, w, h, c_right)
    _draw_points(frame, lm, LEFT_EYE, w, h, c_left, 3)
    _draw_points(frame, lm, RIGHT_EYE, w, h, c_right, 3)
    for idx in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT):
        x, y = _lm_xy(lm[idx], w, h)
        cv2.circle(frame, (int(x), int(y)), 2, c_mouth, -1)
    cv2.circle(frame, (nx, ny), 5, c_nose, -1)

    # 活体状态标签
    live_label = "真人" if is_live else "假脸"
    lines = [
        f"{student_no} {name} [{live_label}] 活体:{live_score:.2f}",
        f"专注 {score_attention:.0f}",
        f"左眼{eye_open_label(ear_l)} 右眼{eye_open_label(ear_r)}  L={ear_l:.2f} R={ear_r:.2f}",
    ]
    draw_cn_text_block(
        frame,
        lines,
        (max(8, nx - 100), max(24, ny - 72)),
        color_bgr=text_color,
        font_size=18,
        line_gap=22,
    )


async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://127.0.0.1:8001")
    p.add_argument("--classroom-id", type=int, default=None)
    p.add_argument("--student-ids", type=str, required=True, help="comma-separated student ids, e.g. 1,2,3")
    p.add_argument("--max-faces", type=int, default=8)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--session-id", type=int, default=None, help="Use existing session_id instead of creating one")
    p.add_argument("--interval", type=float, default=0.25, help="seconds between POSTs")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--stop-demo", action="store_true", help="POST /api/demo/stop before starting")
    p.add_argument("--no-liveness", action="store_true", help="禁用活体检测（调试用）")
    args = p.parse_args()
    student_ids = [int(x) for x in args.student_ids.split(",") if x.strip()]
    if not student_ids:
        raise SystemExit("student_ids is empty")

    async with httpx.AsyncClient(base_url=args.api_base, timeout=15.0) as client:
        if args.stop_demo:
            try:
                await client.post("/api/demo/stop")
            except Exception:
                pass

        classroom_id = args.classroom_id
        if classroom_id is None:
            rooms = (await client.get("/api/classrooms")).json()
            if not rooms:
                raise SystemExit("No classrooms in DB. Create one in UI or POST /api/classrooms")
            classroom_id = int(rooms[0]["id"])

        if args.session_id is None:
            sess = (
                await client.post(
                    "/api/sessions",
                    json={"classroom_id": classroom_id, "title": "摄像头实测"},
                )
            ).json()
            session_id = int(sess["id"])
        else:
            session_id = int(args.session_id)

        students_resp = (await client.get("/api/students")).json()
        student_by_id: dict[int, dict] = {int(s["id"]): s for s in students_resp}

        print(
            f"API={args.api_base} classroom={classroom_id} session={session_id} "
            f"students={student_ids}"
        )
        print("按 Q 退出")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")

    # MediaPipe Solutions API was removed from recent pip builds; use Tasks FaceLandmarker.
    models_dir = (Path(__file__).resolve().parents[1] / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "face_landmarker.task"
    if not model_path.exists():
        print("Downloading face_landmarker.task (one-time)...")
        async with httpx.AsyncClient(timeout=60.0) as dl:
            r = await dl.get(MODEL_URL)
            r.raise_for_status()
            model_path.write_bytes(r.content)
        print("Model downloaded:", model_path)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=max(1, min(args.max_faces, len(student_ids))),
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # ======================================================================
    # 初始化 MiniFASNet 活体检测器
    # ======================================================================
    liveness_detector: LivenessDetector | None = None
    if not args.no_liveness:
        liveness_detector = LivenessDetector(models_dir=models_dir)
        loaded = liveness_detector.load_model(prefer_onnx=True)
        if loaded:
            print(f"[活体检测] 已启用 (后端: {liveness_detector._backend})")
        else:
            print("[活体检测] 模型加载失败，已禁用活体检测（不影响签到和注意力评分）")
            print(f"[活体检测] 请下载模型到 {models_dir}/")
            print(f"[活体检测] ONNX: https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx")
            print(f"[活体检测] PyTorch: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing")
            liveness_detector = None
    else:
        print("[活体检测] 已通过 --no-liveness 参数禁用")

    # ======================================================================
    # 主循环状态变量
    # ======================================================================
    prev_nose_by_slot: dict[int, tuple[float, float] | None] = {}
    speed_ema_by_slot: dict[int, float] = {}
    last_post_by_student: dict[int, float] = {sid: 0.0 for sid in student_ids}
    marked_present: set[int] = set()
    # 记录每个槽位是否被活体检测拒绝（用于在画面显示假脸状态）
    slot_live_status: dict[int, bool] = {}
    slot_live_score: dict[int, float] = {}

    async with httpx.AsyncClient(base_url=args.api_base, timeout=15.0) as post_client:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)

            if res.face_landmarks:
                faces_sorted = sorted(
                    res.face_landmarks,
                    key=lambda lm: _lm_xy(lm[NOSE_TIP], w, h)[0],
                )
                used = min(len(faces_sorted), len(student_ids))
                for idx in range(used):
                    lm = faces_sorted[idx]
                    sid = student_ids[idx]
                    ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                    ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                    ear = (ear_l + ear_r) * 0.5
                    mar = mouth_aspect_ratio(lm, w, h)
                    nose = _lm_xy(lm[NOSE_TIP], w, h)

                    # ============================================================
                    # MiniFASNet 活体检测（关键新增逻辑）
                    # ============================================================
                    is_live = True
                    live_score = 1.0
                    if liveness_detector is not None:
                        # 使用 MediaPipe 468 关键点精准裁剪并对齐人脸
                        face_crop = LivenessDetector.extract_face_crop_from_landmarks(
                            frame, lm, w, h)
                        is_live, live_score, _ = liveness_detector.predict(face_crop)
                        # 假脸：跳过签到和注意力数据提交
                        if not is_live:
                            slot_live_status[idx] = False
                            slot_live_score[idx] = live_score
                            # 仍然绘制假脸标记，但不提交数据
                            if not args.no_window:
                                meta = student_by_id.get(sid, {})
                                draw_face_overlay(
                                    frame, lm, w, h,
                                    student_no=str(meta.get("student_no", f"id{sid}")),
                                    name=str(meta.get("name", f"学生{sid}")),
                                    score_attention=0.0,
                                    ear_l=ear_l, ear_r=ear_r,
                                    is_live=False,
                                    live_score=live_score,
                                )
                            continue  # 跳过当前人脸，不处理注意力/签到
                    # 真人通过，记录状态
                    slot_live_status[idx] = True
                    slot_live_score[idx] = live_score
                    # ============================================================

                    prev_nose = prev_nose_by_slot.get(idx)
                    if prev_nose is None:
                        prev_nose = nose
                    dx = (nose[0] - prev_nose[0]) / (w + 1e-6)
                    dy = (nose[1] - prev_nose[1]) / (h + 1e-6)
                    speed = math.hypot(dx, dy)
                    speed_ema = speed_ema_by_slot.get(idx, 0.0)
                    speed_ema = 0.85 * speed_ema + 0.15 * speed
                    speed_ema_by_slot[idx] = speed_ema
                    prev_nose_by_slot[idx] = nose

                    s_expr_eye = score_from_ear(ear)
                    s_expr_mouth = score_from_mar(mar)
                    score_expression = 0.65 * s_expr_eye + 0.35 * s_expr_mouth
                    score_headpose = score_head_center(nose, w, h)
                    score_behavior = score_behavior_still(speed_ema)
                    score_attention = 0.42 * score_expression + 0.33 * score_headpose + 0.25 * score_behavior

                    yaw_proxy = (nose[0] - w * 0.5) / (w * 0.5) * 45.0

                    now = time.time()
                    if sid not in marked_present:
                        try:
                            await post_client.post(
                                "/api/attendance",
                                json={
                                    "session_id": session_id,
                                    "student_id": sid,
                                    "status": "present",
                                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                                },
                            )
                            marked_present.add(sid)
                        except Exception:
                            pass
                    if now - last_post_by_student.get(sid, 0.0) >= args.interval:
                        last_post_by_student[sid] = now
                        payload = {
                            "session_id": session_id,
                            "student_id": sid,
                            "score_attention": float(round(score_attention, 2)),
                            "score_expression": float(round(score_expression, 2)),
                            "score_headpose": float(round(score_headpose, 2)),
                            "score_behavior": float(round(score_behavior, 2)),
                            "ear": float(round(ear, 4)),
                            "mar": float(round(mar, 4)),
                            "yaw": float(round(yaw_proxy, 2)),
                            "pitch": None,
                            "roll": None,
                            # MiniFASNet 活体检测结果
                            "is_live": slot_live_status.get(idx, True),
                            "live_score": float(round(slot_live_score.get(idx, 1.0), 4)),
                        }
                        try:
                            await post_client.post("/api/attention", json=payload)
                        except Exception as e:
                            print("POST failed:", e)

                    if not args.no_window:
                        meta = student_by_id.get(sid, {})
                        draw_face_overlay(
                            frame,
                            lm,
                            w,
                            h,
                            student_no=str(meta.get("student_no", f"id{sid}")),
                            name=str(meta.get("name", f"学生{sid}")),
                            score_attention=score_attention,
                            ear_l=ear_l,
                            ear_r=ear_r,
                            is_live=slot_live_status.get(idx, True),
                            live_score=slot_live_score.get(idx, 1.0),
                        )
            else:
                for idx in list(prev_nose_by_slot.keys()):
                    prev_nose_by_slot[idx] = None
                    speed_ema_by_slot[idx] = speed_ema_by_slot.get(idx, 0.0) * 0.9
                # 无人脸时清除活体状态
                slot_live_status.clear()
                slot_live_score.clear()
                if not args.no_window:
                    draw_cn_text_block(
                        frame,
                        ["未检测到人脸"],
                        (12, 12),
                        color_bgr=(0, 0, 220),
                        font_size=20,
                        line_gap=24,
                    )

            if not args.no_window:
                cv2.imshow("realtime_camera (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    landmarker.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
