# 1 フレーム分の Look 転写: 7D 統計の比率マッチング + Y 指数圧縮 + 彩度ソフトクリップ

from __future__ import annotations

import logging

import cv2
import numexpr as ne
import numpy as np

from video_stabilizer.color import _numexpr_lock, apply_exponential_compression_fast, get_exp_lut
from video_stabilizer.color_metadata import ColorMetadata
from video_stabilizer.config import Config
from video_stabilizer.stats_constants import (
    IDX_S_MEAN,
    IDX_U_MEAN,
    IDX_U_STD,
    IDX_V_MEAN,
    IDX_V_STD,
    IDX_Y_MEAN,
    IDX_Y_STD,
)

logger = logging.getLogger(__name__)


def process_frame_worker(
    y: np.ndarray | None,
    u: np.ndarray | None,
    v: np.ndarray | None,
    current_stats: np.ndarray,
    target_stats: np.ndarray,
    config: Config,
    color_meta: ColorMetadata | None = None,
) -> np.ndarray | None:
    try:
        if y is None:
            return None

        meta = color_meta or ColorMetadata()
        lut = get_exp_lut(
            config.exp_knee_high,
            config.exp_knee_low,
            config.exp_soft_high,
            config.exp_soft_low,
        )

        with _numexpr_lock:
            # --- 輝度 (Y): 平均シフト + コントラスト比 + 指数圧縮 ---
            # current_stats はターゲット側の「今のルック」、target_stats は参照の目標値。
            lo, hi = config.ratio_clip_low, config.ratio_clip_high
            c_mean = np.float32(current_stats[IDX_Y_MEAN])
            t_mean = np.float32(target_stats[IDX_Y_MEAN])
            contrast_ratio = np.float32(
                np.clip(target_stats[IDX_Y_STD] / current_stats[IDX_Y_STD], lo, hi)
            )
            u_mean_c = np.float32(current_stats[IDX_U_MEAN])
            u_mean_t = np.float32(target_stats[IDX_U_MEAN])
            u_std_ratio = np.float32(
                np.clip(target_stats[IDX_U_STD] / current_stats[IDX_U_STD], lo, hi)
            )
            v_mean_c = np.float32(current_stats[IDX_V_MEAN])
            v_mean_t = np.float32(target_stats[IDX_V_MEAN])
            v_std_ratio = np.float32(
                np.clip(target_stats[IDX_V_STD] / current_stats[IDX_V_STD], lo, hi)
            )

            h, w = y.shape
            h_uv, w_uv = u.shape
            c_y_off = np.float32(meta.y_offset)
            c_uv_off = np.float32(meta.uv_offset)
            c_y = np.float32(meta.y_scale)
            c_uv = np.float32(meta.uv_scale)
            c1, c0 = np.float32(1.0), np.float32(0.0)

            y_f = y.astype(np.float32)
            y_basic = ne.evaluate("(y_f - c_mean) * contrast_ratio + t_mean")
            # 指数 LUT でハイライト/シャドウの急激な伸びを抑え、白飛び・黒つぶれを防ぐ。
            y_comp = apply_exponential_compression_fast(y_basic, lut)

            # --- 色差 (U/V): 平均・分散の比率マッチング（Y より穏やか） ---
            u_f = u.astype(np.float32)
            v_f = v.astype(np.float32)
            u_c = ne.evaluate("(u_f - u_mean_c) * u_std_ratio + u_mean_t")
            v_c = ne.evaluate("(v_f - v_mean_c) * v_std_ratio + v_mean_t")

            # --- 彩度ソフトクリップ: U/V の最大偏差が閾値を超えたら指数で圧縮 ---
            # 高彩度領域のオーバーシュート（蛍光色化）を抑える。Y 方向は触らない。
            c_thresh = np.float32(config.chroma_soft_clip_threshold)
            c_diff = np.float32(config.chroma_soft_clip_diff)
            c_max = ne.evaluate("where(abs(u_c - c_uv_off) > abs(v_c - c_uv_off), abs(u_c - c_uv_off), abs(v_c - c_uv_off))")
            c_max_safe = ne.evaluate("where(c_max == 0, c1, c_max)")
            c_new = ne.evaluate(
                "where(c_max > c_thresh, c_thresh + c_diff * (c1 - exp(-(c_max - c_thresh) / c_diff)), c_max)"
            )
            chroma_scale = ne.evaluate("c_new / c_max_safe")
            u_comp = ne.evaluate("(u_c - c_uv_off) * chroma_scale + c_uv_off")
            v_comp = ne.evaluate("(v_c - c_uv_off) * chroma_scale + c_uv_off")

            u_up = cv2.resize(u_comp, (w, h), interpolation=cv2.INTER_LINEAR)
            v_up = cv2.resize(v_comp, (w, h), interpolation=cv2.INTER_LINEAR)

            # limited/full range を考慮して YUV→RGB（BT.601/709 係数）し BGR 出力。
            y_norm = ne.evaluate("(y_comp - c_y_off) * c_y")
            u_norm = ne.evaluate("(u_up - c_uv_off) * c_uv")
            v_norm = ne.evaluate("(v_up - c_uv_off) * c_uv")
            c_rv, c_gu, c_gv, c_bu = (
                np.float32(1.5748),
                np.float32(0.1873),
                np.float32(0.4681),
                np.float32(1.8556),
            )
            r = ne.evaluate("y_norm + c_rv * v_norm")
            g = ne.evaluate("y_norm - c_gu * u_norm - c_gv * v_norm")
            b = ne.evaluate("y_norm + c_bu * u_norm")
            bgr = np.dstack((b, g, r))
            return np.clip(bgr, meta.clip_min, meta.clip_max).astype(np.uint8)
    except Exception:
        logger.exception("process_frame_worker failed")
        return None
