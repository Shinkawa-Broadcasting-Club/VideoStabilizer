# 色変換と指数圧縮LUT

from __future__ import annotations

import math

import cv2
import numexpr as ne
import numpy as np

cv2.setNumThreads(0)
ne.set_num_threads(1)

_LUT_EXP = np.zeros(1024, dtype=np.float32)
for i in range(-255, 769):
    idx = i + 255
    x = float(i)
    if x > 226.0:
        val = 226.0 + 9.0 * (1.0 - math.exp(-(x - 226.0) / 9.0))
    elif x < 26.0:
        val = 26.0 - 10.0 * (1.0 - math.exp((x - 26.0) / 10.0))
    else:
        val = x
    _LUT_EXP[idx] = val


def apply_exponential_compression_fast(img_float: np.ndarray) -> np.ndarray:
    idx = np.clip(np.round(img_float) + 255, 0, 1023).astype(np.int32)
    return _LUT_EXP[idx]


def worker_yuv_to_float_bgr(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = y.shape
    u_resized = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    y_f = y.astype(np.float32)
    u_f = u_resized.astype(np.float32)
    v_f = v_resized.astype(np.float32)
    c16, c128 = np.float32(16.0), np.float32(128.0)
    c_y, c_uv = np.float32(255.0 / 219.0), np.float32(255.0 / 224.0)
    y_f = ne.evaluate("(y_f - c16) * c_y")
    u_f = ne.evaluate("(u_f - c128) * c_uv")
    v_f = ne.evaluate("(v_f - c128) * c_uv")
    c_rv, c_gu, c_gv, c_bu = np.float32(1.5748), np.float32(0.1873), np.float32(0.4681), np.float32(1.8556)
    r = ne.evaluate("y_f + c_rv * v_f")
    g = ne.evaluate("y_f - c_gu * u_f - c_gv * v_f")
    b = ne.evaluate("y_f + c_bu * u_f")
    return np.dstack((b, g, r))
