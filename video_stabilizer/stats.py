# Mitchellがどうたらとかそのへんの統計系

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numexpr as ne
import numpy as np

from video_stabilizer.color import worker_yuv_to_float_bgr

if TYPE_CHECKING:
    import av as _av

logger = logging.getLogger(__name__)

_MN_C1 = np.float32(-7 / 18)
_MN_C2 = np.float32(5 / 6)
_MN_C3 = np.float32(-1 / 2)
_MN_C4 = np.float32(1 / 18)
_MN_C5 = np.float32(7 / 6)
_MN_C6 = np.float32(-2)
_MN_C7 = np.float32(8 / 9)
_MN_C8 = np.float32(-7 / 6)
_MN_C9 = np.float32(3 / 2)
_MN_C10 = np.float32(7 / 18)
_MN_C11 = np.float32(-1 / 3)


def get_optimal_sampling_interval(fps: float, target_sec: float) -> int:
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    return max(2, round(fps * target_sec))


def extract_yuv_planes(frame: _av.VideoFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        fmt = str(frame.format.name)
        if not fmt.startswith("yuv420"):
            frame = frame.reformat(format="yuv420p")
        y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
            frame.planes[0].height, frame.planes[0].line_size
        )[:, : frame.planes[0].width].copy()
        u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
            frame.planes[1].height, frame.planes[1].line_size
        )[:, : frame.planes[1].width].copy()
        v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
            frame.planes[2].height, frame.planes[2].line_size
        )[:, : frame.planes[2].width].copy()
        return y, u, v
    except Exception:
        logger.exception("extract_yuv_planes failed; skipping frame")
        return None


def get_stats_and_coeffs(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    stats_max_dim: int,
) -> np.ndarray | None:
    max_dim = max(y.shape)
    scale = min(float(stats_max_dim) / max_dim, 1.0)
    new_w = max(int(y.shape[1] * scale), 1)
    new_h = max(int(y.shape[0] * scale), 1)
    small_y = cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_u = cv2.resize(u, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_v = cv2.resize(v, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr_small = worker_yuv_to_float_bgr(small_y, small_u, small_v)
    bgr_uint8 = np.clip(bgr_small, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV)
    y_mean = np.mean(small_y)
    y_std = np.std(small_y) + 1e-6
    s_mean = np.mean(hsv[:, :, 1]) + 1e-6
    return np.array([y_mean, y_std, s_mean], dtype=np.float32)


def mitchell_netravali(t: float, p: list[np.ndarray]) -> np.ndarray:
    t = np.float32(t)
    expr = (
        "p0 * (t * (t * (_MN_C1 * t + _MN_C2) + _MN_C3) + _MN_C4) + "
        "p1 * ((_MN_C5 * t + _MN_C6) * t**2 + _MN_C7) + "
        "p2 * (t * (t * (_MN_C8 * t + _MN_C9) + _MN_C3) + _MN_C4) + "
        "p3 * ((_MN_C10 * t + _MN_C11) * t**2)"
    )
    local_dict = {
        "t": t,
        "p0": p[0],
        "p1": p[1],
        "p2": p[2],
        "p3": p[3],
        "_MN_C1": _MN_C1,
        "_MN_C2": _MN_C2,
        "_MN_C3": _MN_C3,
        "_MN_C4": _MN_C4,
        "_MN_C5": _MN_C5,
        "_MN_C6": _MN_C6,
        "_MN_C7": _MN_C7,
        "_MN_C8": _MN_C8,
        "_MN_C9": _MN_C9,
        "_MN_C10": _MN_C10,
        "_MN_C11": _MN_C11,
    }
    return ne.evaluate(expr, local_dict=local_dict)


def init_ema_with_warmup(
    warmup_buffer: list,
    ema_alpha: float,
    stats_max_dim: int,
) -> np.ndarray | None:
    ema_stats = None
    for y, u, v in warmup_buffer:
        s = get_stats_and_coeffs(y, u, v, stats_max_dim)
        if ema_stats is None:
            ema_stats = s
        else:
            ema_stats = ema_alpha * s + (1.0 - ema_alpha) * ema_stats
    return ema_stats
