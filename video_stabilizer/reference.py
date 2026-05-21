# リファレンスの解析をこぴぺしただけ

from __future__ import annotations

import logging

import av
import numpy as np
from tqdm import tqdm

from video_stabilizer.config import Config
from video_stabilizer.stats import extract_yuv_planes, get_stats_and_coeffs

logger = logging.getLogger(__name__)


def analyze_reference_video(ref_path: str, config: Config) -> np.ndarray | None:
    ref_stats_list: list[np.ndarray] = []
    logger.info("参照映像を解析しています: %s", ref_path)
    try:
        ref_container = av.open(ref_path)
    except Exception:
        logger.exception("Failed to open reference video: %s", ref_path)
        return None

    try:
        ref_stream = ref_container.streams.video[0]
        total_ref = (
            ref_stream.frames
            if ref_stream.frames > 0
            else int(float(ref_stream.duration * ref_stream.time_base) * ref_stream.average_rate)
        )

        pbar_ref = tqdm(total=total_ref if total_ref > 0 else None, desc="Ref Analysis")
        try:
            for idx, frame in enumerate(ref_container.decode(ref_stream)):
                if idx % config.ref_frame_stride == 0:
                    yuv = extract_yuv_planes(frame)
                    if yuv is None:
                        pbar_ref.update(1)
                        continue
                    y, u, v = yuv
                    ref_stats_list.append(
                        get_stats_and_coeffs(y, u, v, config.stats_max_dim)
                    )
                pbar_ref.update(1)
        finally:
            pbar_ref.close()
    finally:
        ref_container.close()

    if not ref_stats_list:
        logger.error("Could not collect statistics from reference video.")
        return None

    ref_avg_stats = np.mean(ref_stats_list, axis=0)
    ref_avg_stats[1] = max(ref_avg_stats[1], config.ref_stats_min_std)
    ref_avg_stats[2] = max(ref_avg_stats[2], config.ref_stats_min_std)
    return ref_avg_stats
