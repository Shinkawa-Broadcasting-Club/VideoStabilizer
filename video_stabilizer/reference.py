# 参照映像の解析

from __future__ import annotations

import logging
import math
import threading
from typing import Callable

import av
import numpy as np

from video_stabilizer.color_metadata import read_color_metadata
from video_stabilizer.config import Config
from video_stabilizer.stats import extract_yuv_planes, get_stats_and_coeffs, robust_aggregate_stats

logger = logging.getLogger(__name__)


def _safe_rate_to_float(rate: object | None) -> float | None:
    if rate is None:
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _estimate_total_ref_frames(ref_stream: av.stream.Stream, config: Config) -> int | None:
    frames = getattr(ref_stream, "frames", 0) or 0
    if frames > 0:
        return int(frames)
    fps = (
        _safe_rate_to_float(getattr(ref_stream, "average_rate", None))
        or _safe_rate_to_float(getattr(ref_stream, "guessed_rate", None))
        or _safe_rate_to_float(getattr(ref_stream, "base_rate", None))
        or config.default_fps
    )
    duration = getattr(ref_stream, "duration", None)
    time_base = getattr(ref_stream, "time_base", None)
    if duration is not None and time_base is not None:
        try:
            seconds = float(duration * time_base)
            if seconds > 0:
                estimated = int(seconds * fps)
                if estimated > 0:
                    return estimated
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def analyze_reference_video(
    ref_path: str,
    config: Config,
    *,
    progress_cb: Callable[[int, int | None], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> np.ndarray | None:
    ref_stats_list: list[np.ndarray] = []
    logger.info("参照映像を解析しています: %s", ref_path)
    try:
        ref_container = av.open(ref_path)
    except Exception:
        logger.exception("Failed to open reference video: %s", ref_path)
        return None

    color_meta = None
    try:
        ref_stream = ref_container.streams.video[0]
        color_meta = read_color_metadata(ref_stream)
        total_ref = _estimate_total_ref_frames(ref_stream, config)

        idx = 0
        for frame in ref_container.decode(ref_stream):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("参照解析がキャンセルされました。")
                return None
            if idx % config.ref_frame_stride == 0:
                yuv = extract_yuv_planes(frame)
                if yuv is not None:
                    y, u, v = yuv
                    ref_stats_list.append(
                        get_stats_and_coeffs(
                            y,
                            u,
                            v,
                            config.stats_max_dim,
                            y_offset=color_meta.y_offset,
                            y_scale=color_meta.y_scale,
                            uv_offset=color_meta.uv_offset,
                            uv_scale=color_meta.uv_scale,
                        )
                    )
            if progress_cb is not None:
                progress_cb(idx + 1, total_ref)
            idx += 1
    finally:
        ref_container.close()

    ref_avg_stats = robust_aggregate_stats(
        ref_stats_list,
        percentile_low=config.ref_trim_percentile_low,
        percentile_high=config.ref_trim_percentile_high,
        min_std=config.ref_stats_min_std,
    )
    # 参照全体の代表 7D 統計。以降、全ターゲットの補正目標 (target_stats) として使う。
    if ref_avg_stats is None:
        logger.error("Could not collect statistics from reference video.")
        return None

    logger.info(
        "参照統計 (7D): Y=%.1f/%.1f U=%.1f/%.1f V=%.1f/%.1f S=%.1f",
        ref_avg_stats[0],
        ref_avg_stats[1],
        ref_avg_stats[2],
        ref_avg_stats[3],
        ref_avg_stats[4],
        ref_avg_stats[5],
        ref_avg_stats[6],
    )
    return ref_avg_stats
