"""
MiniFASNet 静默活体检测模块
=============================
基于小视科技(Silent-Face-Anti-Spoofing)的 MiniFASNetV2 架构。
输入：80×80 BGR 人脸裁剪图
输出：(is_live: bool, score: float, detail: str)
   - is_live: True=真人, False=假脸(照片/视频回放)
   - score: 活体置信度 0.0~1.0 (越接近1越可能是真人)
   - detail: "live" / "spoof_print" / "spoof_replay"

支持两种推理后端：
  1. ONNX Runtime (推荐，速度快，模型可独立下载)
  2. PyTorch (直接加载 .pth 权重)

Usage:
  from tools.liveness_detector import LivenessDetector
  detector = LivenessDetector()       # 自动下载/加载模型
  is_live, score, detail = detector.predict(face_crop_bgr)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# 尝试导入 PyTorch（用于 PyTorch 后端推理和模型定义）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

# ============================================================================
# 模型下载 URL（HuggingFace 上的 MiniFASNetV2 ONNX 预训练权重）
# ============================================================================
MODEL_ONNX_URL = (
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/"
    "resolve/main/minifasnet_v2.onnx"
)

# 备用：PyTorch 权重（Silent-Face-Anti-Spoofing 官方仓库）
MODEL_PTH_URL = (
    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/"
    "raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
)

# ============================================================================
# MiniFASNetV2 PyTorch 模型定义（用于直接加载 .pth 权重）
# 仅在 torch 可用时定义这些类
# ============================================================================

if HAS_TORCH:

    class L2Norm(torch.nn.Module):
        """L2 归一化层"""
        def __init__(self, n_channels: int, scale: float = 10.0):
            super().__init__()
            self.scale = scale
            self.weight = torch.nn.Parameter(torch.empty(n_channels))
            torch.nn.init.constant_(self.weight, self.scale)

        def forward(self, x):
            norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
            x = torch.div(x, norm)
            return x * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)


    def conv_block(in_c: int, out_c: int, kernel: int = 1,
                   stride: int = 1, padding: int = 0, groups: int = 1):
        """Conv2d + BatchNorm2d + PReLU 基础卷积块"""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, kernel, stride, padding,
                             groups=groups, bias=False),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.PReLU(out_c),
        )


    def linear_block(in_c: int, out_c: int, kernel: int = 1,
                     stride: int = 1, padding: int = 0, groups: int = 1):
        """Conv2d + BatchNorm2d (无激活函数，用于残差路径)"""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, kernel, stride, padding,
                             groups=groups, bias=False),
            torch.nn.BatchNorm2d(out_c),
        )


    class SEModule(torch.nn.Module):
        """Squeeze-and-Excitation 通道注意力模块"""
        def __init__(self, channels: int, reduction: int = 4):
            super().__init__()
            self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels // reduction, 1, bias=False),
                torch.nn.BatchNorm2d(channels // reduction),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channels // reduction, channels, 1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.fc(y)
            return x * y


    class DepthWise(torch.nn.Module):
        """深度可分离卷积残差模块"""
        def __init__(self, in_c: int, out_c: int, residual: bool = False,
                     kernel: int = 3, stride: int = 1, padding: int = 1):
            super().__init__()
            self.residual = residual
            self.conv = torch.nn.Sequential(
                conv_block(in_c, in_c, 1),                          # 1×1 升维
                conv_block(in_c, out_c, kernel, stride, padding,
                            groups=in_c),                           # 3×3 逐通道
                linear_block(out_c, out_c, 1),                      # 1×1 降维
            )

        def forward(self, x):
            shortcut = x
            out = self.conv(x)
            if self.residual:
                out = out + shortcut
            return out


    class DepthWiseSE(DepthWise):
        """带 SE 注意力机制的深度可分离卷积残差模块"""
        def __init__(self, in_c: int, out_c: int, residual: bool = False,
                     kernel: int = 3, stride: int = 1, padding: int = 1):
            super().__init__(in_c, out_c, residual, kernel, stride, padding)
            self.se = SEModule(out_c)

        def forward(self, x):
            shortcut = x
            out = self.conv(x)
            out = self.se(out)
            if self.residual:
                out = out + shortcut
            return out


    class Residual(torch.nn.Module):
        """残差块堆叠"""
        def __init__(self, c: int, num_block: int, groups: int = 1,
                     kernel: int = 3, stride: int = 1, padding: int = 1):
            super().__init__()
            modules = []
            for _ in range(num_block):
                modules.append(DepthWise(c, c, residual=True, kernel=kernel,
                                         stride=stride, padding=padding))
            self.model = torch.nn.Sequential(*modules)

        def forward(self, x):
            return self.model(x)


    class MiniFASNet(torch.nn.Module):
        """
        MiniFASNet 主干网络
        ----------------------
        输入: (B, 3, 80, 80)
        输出: (B, embedding_size)
        """
        def __init__(self, keep_dict: dict, embedding_size: int = 128,
                     drop_ratio: float = 0.2, mode: str = 'ir'):
            super().__init__()
            # 解析各阶段通道数
            k = keep_dict
            stage1 = k.get('1.0_', [16, 16, 16])
            stage2 = k.get('1.8_', [64, 48, 48])

            # 输入层: 3 → stage1[0], stride=2, 80→40
            self.conv1 = conv_block(3, stage1[0], 3, 2, 1)
            # 深度卷积层
            self.conv2_dw = conv_block(stage1[0], stage1[0], 3, 1, 1,
                                        groups=stage1[0])

            # Stage 1→2: 下采样 40→20
            self.conv_23 = DepthWise(stage1[0], stage1[1], residual=False,
                                      kernel=3, stride=2, padding=1)
            # Stage 2 残差块 ×4
            self.conv_3 = Residual(stage1[1], num_block=4, groups=stage1[1],
                                    kernel=3, stride=1, padding=1)

            # Stage 2→3: 下采样 20→10
            self.conv_34 = DepthWise(stage1[1], stage2[0], residual=False,
                                      kernel=3, stride=2, padding=1)
            # Stage 3 残差块 ×6
            self.conv_4 = Residual(stage2[0], num_block=6, groups=stage2[0],
                                    kernel=3, stride=1, padding=1)

            # Stage 3→4: 下采样 10→5
            self.conv_45 = DepthWise(stage2[0], stage2[1], residual=False,
                                      kernel=3, stride=2, padding=1)
            # Stage 4 残差块 ×2
            self.conv_5 = Residual(stage2[1], num_block=2, groups=stage2[1],
                                    kernel=3, stride=1, padding=1)

            # 1×1 卷积调整通道
            self.conv_6_sep = conv_block(stage2[1], stage2[1], 1)
            # 7×7 逐通道卷积替代全连接
            self.conv_6_dw = linear_block(stage2[1], stage2[1], 7, 1, 0,
                                           groups=stage2[1])
            # 最终 1×1 卷积映射到 embedding 维度
            self.conv_6_flatten = torch.nn.Flatten()
            self.linear = torch.nn.Linear(stage2[1], embedding_size, bias=False)
            self.bn = torch.nn.BatchNorm1d(embedding_size)
            self.drop = torch.nn.Dropout(drop_ratio)

            # 如果 embedding_size != 512，需要额外线性层对齐
            if embedding_size != 512:
                self.linear_align = torch.nn.Linear(embedding_size, 512, bias=False)

        def forward(self, x):
            out = self.conv1(x)              # (B, 16, 40, 40)
            out = self.conv2_dw(out)         # (B, 16, 40, 40)
            out = self.conv_23(out)          # (B, 16, 20, 20)
            out = self.conv_3(out)           # (B, 16, 20, 20)
            out = self.conv_34(out)          # (B, 64, 10, 10)
            out = self.conv_4(out)           # (B, 64, 10, 10)
            out = self.conv_45(out)          # (B, 48, 5, 5)
            out = self.conv_5(out)           # (B, 48, 5, 5)
            out = self.conv_6_sep(out)       # (B, 48, 5, 5)
            out = self.conv_6_dw(out)        # (B, 48, 1, 1)
            out = self.conv_6_flatten(out)   # (B, 48)
            out = self.linear(out)           # (B, 128)
            out = self.bn(out)
            # 如果 embedding_size != 512，需要对齐（预训练权重期望 512 维）
            if hasattr(self, 'linear_align'):
                out = self.linear_align(out)  # (B, 512)
            return self.drop(out)


    def MiniFASNetV2(num_classes: int = 3, drop_ratio: float = 0.2):
        """
        创建 MiniFASNetV2 实例
        ------------------------
        参数:
            num_classes: 分类数（3=真脸/打印攻击/重放攻击）
            drop_ratio: dropout 比例
        返回:
            完整的分类网络 (MiniFASNet + 分类头)
        """
        keep_dict = {
            '1.8M_': [64, 48, 48],   # V2 的通道配置（比 V1 更宽）
            '1.0_': [16, 16, 16],
        }
        backbone = MiniFASNet(
            keep_dict=keep_dict,
            embedding_size=128,
            drop_ratio=drop_ratio,
            mode='ir',
        )
        classifier = torch.nn.Linear(512, num_classes, bias=False)
        return torch.nn.Sequential(backbone, classifier)


# ============================================================================
# 活体检测器封装
# ============================================================================

class LivenessDetector:
    """
    MiniFASNet 活体检测器

    使用方式:
        detector = LivenessDetector()
        detector.load_model()           # 首次运行会自动下载模型
        is_live, score, detail = detector.predict(face_crop_bgr)
    """

    # 输入图像尺寸
    INPUT_SIZE = (80, 80)

    # 活体/假脸分类阈值
    LIVE_THRESHOLD = 0.3  # 活体置信度 > 此值判定为真人

    # 三分类标签（需根据实际 ONNX 模型输出调整）
    # 官方 test.py: label 1 = Real Face (真人)
    # 但此 ONNX 是第三方转换，标签顺序可能不同
    # 先用 argmax 判断：若最高概率是某个类且该类别被定义为"live"，则为真人
    CLASS_LABELS = {
        0: ("live", "真人"),
        1: ("spoof_print", "照片打印攻击"),
        2: ("spoof_replay", "视频重放攻击"),
    }

    def __init__(self, models_dir: Optional[Path] = None):
        """
        初始化活体检测器

        参数:
            models_dir: 模型存放目录，默认为 backend/models/
        """
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir

        self._model = None          # PyTorch 模型
        self._ort_session = None    # ONNX Runtime 会话
        self._backend = None        # "onnx" 或 "pytorch"
        self._loaded = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_model(self, prefer_onnx: bool = True) -> bool:
        """
        加载活体检测模型

        参数:
            prefer_onnx: 优先使用 ONNX 后端（True）还是 PyTorch（False）

        返回:
            是否加载成功
        """
        if prefer_onnx:
            if self._try_load_onnx():
                self._backend = "onnx"
                self._loaded = True
                print(f"[活体检测] 已加载 ONNX 模型")
                return True

        # 回退到 PyTorch
        if self._try_load_pytorch():
            self._backend = "pytorch"
            self._loaded = True
            print(f"[活体检测] 已加载 PyTorch 模型")
            return True

        print("[活体检测] 警告: 模型加载失败，活体检测将不可用")
        return False

    def _try_load_onnx(self) -> bool:
        """尝试加载 ONNX 模型"""
        try:
            import onnxruntime as ort
        except ImportError:
            print("[活体检测] onnxruntime 未安装，跳过 ONNX 加载")
            return False

        onnx_path = self.models_dir / "minifasnet_v2.onnx"
        if not onnx_path.exists():
            print(f"[活体检测] ONNX 模型不存在: {onnx_path}")
            print(f"[活体检测] 请手动下载到 {onnx_path}")
            print(f"[活体检测] 下载地址: {MODEL_ONNX_URL}")
            return False

        try:
            self._ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider'],
            )
            return True
        except Exception as e:
            print(f"[活体检测] ONNX 加载失败: {e}")
            return False

    def _try_load_pytorch(self) -> bool:
        """尝试加载 PyTorch 权重"""
        if not HAS_TORCH:
            print("[活体检测] PyTorch 未安装，无法加载模型")
            return False

        pth_path = self.models_dir / "2.7_80x80_MiniFASNetV2.pth"
        if not pth_path.exists():
            print(f"[活体检测] PyTorch 权重不存在: {pth_path}")
            print(f"[活体检测] 请手动下载到 {pth_path}")
            print(f"[活体检测] 下载地址: {MODEL_PTH_URL}")
            return False

        try:
            # 创建模型并加载权重
            model = MiniFASNetV2(num_classes=3, drop_ratio=0.2)
            state_dict = torch.load(str(pth_path), map_location='cpu',
                                     weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._model = model
            return True
        except Exception as e:
            print(f"[活体检测] PyTorch 权重加载失败: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        将 BGR 人脸裁剪图预处理为模型输入

        参数:
            face_crop_bgr: BGR 格式的人脸区域图像 (H×W×3)

        返回:
            预处理后的张量 (1, 3, 80, 80)，值域 [-1, 1]
        """
        # 缩放到 80×80
        img = cv2.resize(face_crop_bgr, (80, 80))
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # uint8 [0,255] → float32 [0,1] → [-1,1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0  # 归一化到 [-1, 1]
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)
        return img

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(self, face_crop_bgr: np.ndarray) -> Tuple[bool, float, str]:
        """
        对单张人脸裁剪图进行活体检测

        参数:
            face_crop_bgr: BGR 格式的人脸区域图像

        返回:
            (is_live, score, detail)
            - is_live: True=真人, False=假脸
            - score: 活体置信度 0.0~1.0
            - detail: "live" / "spoof_print" / "spoof_replay" / "unknown"
        """
        if not self._loaded:
            # 未加载模型时，默认通过（不阻断流程）
            return True, 1.0, "unknown"

        x = self.preprocess(face_crop_bgr)

        if self._backend == "onnx":
            logits = self._predict_onnx(x)
        elif self._backend == "pytorch":
            logits = self._predict_pytorch(x)
        else:
            return True, 1.0, "unknown"

        # softmax 转概率
        probs = self._softmax(logits)
        # squeeze 掉 batch 维度 (1, 3) → (3,)
        probs = np.squeeze(probs)
        pred_class = int(np.argmax(probs))

        # 活体判定：取 argmax 类别，检查是否是 "live"
        label_en, label_cn = self.CLASS_LABELS.get(
            pred_class, ("unknown", "未知"))

        if label_en == "live":
            live_score = float(probs[0])
            is_live = True
        else:
            # 假脸时返回对应类别的概率作为置信度
            live_score = float(probs[pred_class])
            # 假脸攻击，拒绝签到和注意力提交
            is_live = False

        return is_live, live_score, label_en


    def _predict_onnx(self, x: np.ndarray) -> np.ndarray:
        """ONNX 推理"""
        ort_inputs = {self._ort_session.get_inputs()[0].name: x}
        logits = self._ort_session.run(None, ort_inputs)[0]
        return logits

    def _predict_pytorch(self, x: np.ndarray) -> np.ndarray:
        """PyTorch 推理"""
        with torch.no_grad():
            tensor = torch.from_numpy(x)
            logits = self._model(tensor).cpu().numpy()
        return logits

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """稳定版 softmax"""
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # 人脸裁剪辅助
    # ------------------------------------------------------------------

    @staticmethod
    def extract_face_crop(
        frame_bgr: np.ndarray,
        nose_xy: Tuple[float, float],
        frame_w: int,
        frame_h: int,
        expand_ratio: float = 2.2,
    ) -> np.ndarray:
        """
        从原始帧中以鼻尖为中心裁剪人脸区域（回退方法）

        参数:
            frame_bgr: 原始 BGR 帧
            nose_xy: 鼻尖在帧中的像素坐标 (x, y)
            frame_w, frame_h: 帧的宽高
            expand_ratio: 扩展比例（相对标准人脸大小）

        返回:
            80×80 的 BGR 人脸裁剪图
        """
        # 以鼻尖为中心，估算人脸框大小
        face_size = int(min(frame_w, frame_h) * 0.15 * expand_ratio)
        half = face_size // 2

        nx, ny = int(nose_xy[0]), int(nose_xy[1])

        x1 = max(0, nx - half)
        y1 = max(0, ny - half)
        x2 = min(frame_w, nx + half)
        y2 = min(frame_h, ny + half)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return cv2.resize(frame_bgr, (80, 80))

        face_crop = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(face_crop, (80, 80))

    @staticmethod
    def extract_face_crop_from_landmarks(
        frame_bgr: np.ndarray,
        landmarks,  # MediaPipe FaceLandmarker 的 face_landmarks[0]
        frame_w: int,
        frame_h: int,
        padding: float = 0.3,
    ) -> np.ndarray:
        """
        使用 MediaPipe 468 个关键点精准裁剪人脸区域，并进行仿射对齐。

        MiniFASNetV2 期望输入的是对齐后的人脸（双眼水平、居中），
        直接简单裁剪会导致活体检测误判。

        参数:
            frame_bgr: 原始 BGR 帧
            landmarks: MediaPipe 人脸关键点列表（468个点，归一化坐标 0~1）
            frame_w, frame_h: 帧的宽高
            padding: 边界扩展比例（默认30%）

        返回:
            80×80 的 BGR 对齐人脸裁剪图
        """
        # 关键点索引（MediaPipe FaceMesh 468 landmarks）
        LEFT_EYE_IDX = 33    # 左眼外角
        RIGHT_EYE_IDX = 263  # 右眼外角
        LEFT_MOUTH = 61      # 左嘴角
        RIGHT_MOUTH = 291    # 右嘴角
        NOSE = 1             # 鼻尖

        def _xy(idx):
            lm = landmarks[idx]
            return np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float32)

        # 获取双眼中心坐标
        left_eye = _xy(LEFT_EYE_IDX)
        right_eye = _xy(RIGHT_EYE_IDX)
        eye_center = (left_eye + right_eye) / 2.0

        # 计算双眼连线角度
        d = right_eye - left_eye
        angle = np.degrees(np.arctan2(d[1], d[0]))
        eye_dist = np.linalg.norm(d)

        # 获取嘴巴中心
        mouth_left = _xy(LEFT_MOUTH)
        mouth_right = _xy(RIGHT_MOUTH)
        mouth_center = (mouth_left + mouth_right) / 2.0

        # 估算人脸边界：以眼距为基准
        # 标准人脸比例：眼距约占脸宽的 0.4，脸高约为脸宽的 1.25
        face_size = eye_dist * 2.5  # 脸宽约为眼距的2.5倍
        face_h = face_size * 1.35   # 脸高略大于脸宽

        # 以双眼中心为基准，向上偏移约眼距的0.6倍到额头，向下到下巴
        cx, cy = eye_center
        top = cy - face_h * 0.45
        bottom = cy + face_h * 0.55

        half_w = face_size / 2.0
        left = cx - half_w
        right = cx + half_w

        # 加入 padding
        pad_w = face_size * padding
        pad_h = face_h * padding
        x1 = max(0, int(left - pad_w))
        y1 = max(0, int(top - pad_h))
        x2 = min(frame_w, int(right + pad_w))
        y2 = min(frame_h, int(bottom + pad_h))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return cv2.resize(frame_bgr, (80, 80))

        # 裁剪人脸区域
        face_crop = frame_bgr[y1:y2, x1:x2]

        # 缩放到 80×80（不做旋转对齐，因为模型对轻微倾斜有容忍度）
        face_crop = cv2.resize(face_crop, (80, 80))

        return face_crop


# ============================================================================
# 全局单例
# ============================================================================

_liveness_detector: Optional[LivenessDetector] = None


def get_liveness_detector(models_dir: Optional[Path] = None) -> LivenessDetector:
    """
    获取活体检测器全局单例

    参数:
        models_dir: 模型存放目录

    返回:
        LivenessDetector 实例
    """
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector(models_dir=models_dir)
        _liveness_detector.load_model(prefer_onnx=True)
    return _liveness_detector


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MiniFASNet 活体检测模块 - 自测")
    print("=" * 60)

    # 测试模型加载
    detector = LivenessDetector()
    ok = detector.load_model(prefer_onnx=True)
    if ok:
        print(f"✅ 模型加载成功 (后端: {detector._backend})")
    else:
        print("⚠️  模型未加载（缺少权重文件），将使用默认通过策略")
        print("   请下载模型权重到 backend/models/ 目录：")
        print(f"   ONNX: {MODEL_ONNX_URL}")
        print(f"   或 PyTorch: {MODEL_PTH_URL}")

    # 测试预处理
    dummy_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    x = LivenessDetector.preprocess(dummy_face)
    print(f"✅ 预处理输出形状: {x.shape} (期望: (1, 3, 80, 80))")

    # 测试人脸裁剪
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = LivenessDetector.extract_face_crop(
        dummy_frame, (320, 240), 640, 480)
    print(f"✅ 人脸裁剪输出形状: {crop.shape} (期望: (80, 80, 3))")

    print("=" * 60)
    print("自测完成")
