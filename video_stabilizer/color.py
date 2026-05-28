# 色変換と指数圧縮LUT

from __future__ import annotations

import math
import threading

import cv2
import numexpr as ne
import numpy as np

cv2.setNumThreads(0)
ne.set_num_threads(1)

# numexpr は内部スレッドを持つため、パイプラインの ThreadPoolExecutor と
# 競合しないよう 1 スレッド + ロックで直列化する。
_numexpr_lock = threading.Lock()

_DEFAULT_KNEE_HIGH = 226.0
_DEFAULT_KNEE_LOW = 26.0
_DEFAULT_SOFT_HIGH = 9.0
_DEFAULT_SOFT_LOW = 10.0

_LUT_CACHE: dict[tuple[float, float, float, float], np.ndarray] = {}


def build_exp_lut(
    knee_high: float = _DEFAULT_KNEE_HIGH,
    knee_low: float = _DEFAULT_KNEE_LOW,
    soft_high: float = _DEFAULT_SOFT_HIGH,
    soft_low: float = _DEFAULT_SOFT_LOW,
) -> np.ndarray:
    """Build exponential compression LUT for Y channel."""
    key = (knee_high, knee_low, soft_high, soft_low)
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached

    # 入力 Y は 0–255 付近だが、補正後の y_basic ははみ出すため -255..768 をカバー。
    # knee 外側は指数関数でソフトに圧縮し、中間域はリニアのまま通す。
    lut = np.zeros(1024, dtype=np.float32)
    for i in range(-255, 769):
        idx = i + 255
        x = float(i)
        if x > knee_high:
            val = knee_high + soft_high * (1.0 - math.exp(-(x - knee_high) / soft_high))
        elif x < knee_low:
            val = knee_low - soft_low * (1.0 - math.exp((x - knee_low) / soft_low))
        else:
            val = x
        lut[idx] = val
    _LUT_CACHE[key] = lut
    return lut


# Default LUT matches legacy hardcoded behavior
_LUT_EXP = build_exp_lut()


def get_exp_lut(
    knee_high: float = _DEFAULT_KNEE_HIGH,
    knee_low: float = _DEFAULT_KNEE_LOW,
    soft_high: float = _DEFAULT_SOFT_HIGH,
    soft_low: float = _DEFAULT_SOFT_LOW,
) -> np.ndarray:
    return build_exp_lut(knee_high, knee_low, soft_high, soft_low)


def apply_exponential_compression_fast(
    img_float: np.ndarray,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    table = lut if lut is not None else _LUT_EXP
    idx = np.clip(np.round(img_float) + 255, 0, 1023).astype(np.int32)
    return table[idx]


def worker_yuv_to_float_bgr(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    y_offset: float = 16.0,
    y_scale: float = 255.0 / 219.0,
    uv_offset: float = 128.0,
    uv_scale: float = 255.0 / 224.0,
) -> np.ndarray:
    with _numexpr_lock:
        h, w = y.shape
        u_resized = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        v_resized = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
        y_f = y.astype(np.float32)
        u_f = u_resized.astype(np.float32)
        v_f = v_resized.astype(np.float32)
        c_y_off = np.float32(y_offset)
        c_uv_off = np.float32(uv_offset)
        c_y = np.float32(y_scale)
        c_uv = np.float32(uv_scale)
        y_f = ne.evaluate("(y_f - c_y_off) * c_y")
        u_f = ne.evaluate("(u_f - c_uv_off) * c_uv")
        v_f = ne.evaluate("(v_f - c_uv_off) * c_uv")
        c_rv, c_gu, c_gv, c_bu = (
            np.float32(1.5748),
            np.float32(0.1873),
            np.float32(0.4681),
            np.float32(1.8556),
        )
        r = ne.evaluate("y_f + c_rv * v_f")
        g = ne.evaluate("y_f - c_gu * u_f - c_gv * v_f")
        b = ne.evaluate("y_f + c_bu * u_f")
        return np.dstack((b, g, r))
