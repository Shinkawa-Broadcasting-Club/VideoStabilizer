# 色補正系

from __future__ import annotations

import logging

import cv2
import numexpr as ne
import numpy as np

from video_stabilizer.color import apply_exponential_compression_fast
from video_stabilizer.config import Config

logger = logging.getLogger(__name__)


def process_frame_worker(
    y: np.ndarray | None,
    u: np.ndarray | None,
    v: np.ndarray | None,
    current_stats: np.ndarray,
    target_stats: np.ndarray,
    config: Config,
) -> np.ndarray | None:
    try:
        if y is None:
            return None
        lo, hi = config.ratio_clip_low, config.ratio_clip_high
        c_mean = np.float32(current_stats[0])
        t_mean = np.float32(target_stats[0])
        contrast_ratio = np.float32(np.clip(target_stats[1] / current_stats[1], lo, hi))
        sat_ratio = np.float32(np.clip(target_stats[2] / current_stats[2], lo, hi))
        h, w = y.shape
        h_uv, w_uv = u.shape
        c128, c16, c1, c0 = np.float32(128.0), np.float32(16.0), np.float32(1.0), np.float32(0.0)
        y_f = y.astype(np.float32)
        y_basic = ne.evaluate("(y_f - c_mean) * contrast_ratio + t_mean")
        y_comp = apply_exponential_compression_fast(y_basic)
        y_old_norm = ne.evaluate("y_f - c16")
        y_old_norm = ne.evaluate("where(y_old_norm < c1, c1, y_old_norm)")
        y_new_norm = ne.evaluate("y_comp - c16")
        y_new_norm = ne.evaluate("where(y_new_norm < c0, c0, y_new_norm)")
        y_ratio = ne.evaluate("y_new_norm / y_old_norm")
        y_ratio = np.clip(y_ratio, lo, hi)
        y_ratio_down = cv2.resize(y_ratio, (w_uv, h_uv), interpolation=cv2.INTER_AREA)
        u_f = u.astype(np.float32)
        v_f = v.astype(np.float32)
        uv_scale = ne.evaluate("sat_ratio * y_ratio_down")
        u_c = ne.evaluate("(u_f - c128) * uv_scale")
        v_c = ne.evaluate("(v_f - c128) * uv_scale")
        c_max = ne.evaluate("where(abs(u_c) > abs(v_c), abs(u_c), abs(v_c))")
        c_max_safe = ne.evaluate("where(c_max == 0, c1, c_max)")
        c_thresh = np.float32(config.chroma_soft_clip_threshold)
        c_diff = np.float32(config.chroma_soft_clip_diff)
        c_new = ne.evaluate(
            "where(c_max > c_thresh, c_thresh + c_diff * (c1 - exp(-(c_max - c_thresh) / c_diff)), c_max)"
        )
        chroma_scale = ne.evaluate("c_new / c_max_safe")
        u_comp = ne.evaluate("u_c * chroma_scale + c128")
        v_comp = ne.evaluate("v_c * chroma_scale + c128")
        u_up = cv2.resize(u_comp, (w, h), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize(v_comp, (w, h), interpolation=cv2.INTER_LINEAR)
        c_y = np.float32(255.0 / 219.0)
        c_uv = np.float32(255.0 / 224.0)
        y_norm = ne.evaluate("(y_comp - c16) * c_y")
        u_norm = ne.evaluate("(u_up - c128) * c_uv")
        v_norm = ne.evaluate("(v_up - c128) * c_uv")
        c_rv, c_gu, c_gv, c_bu = np.float32(1.5748), np.float32(0.1873), np.float32(0.4681), np.float32(1.8556)
        r = ne.evaluate("y_norm + c_rv * v_norm")
        g = ne.evaluate("y_norm - c_gu * u_norm - c_gv * v_norm")
        b = ne.evaluate("y_norm + c_bu * u_norm")
        bgr = np.dstack((b, g, r))
        return np.clip(bgr, 16, 235).astype(np.uint8)
    except Exception:
        logger.exception("process_frame_worker failed")
        return None
