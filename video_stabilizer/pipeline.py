# パイプライン系そのままこぴぺしただけ

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
from collections import deque

import av
import cv2
import numpy as np
from tqdm import tqdm

from video_stabilizer.config import Config
from video_stabilizer.stats import (
    extract_yuv_planes,
    get_optimal_sampling_interval,
    get_stats_and_coeffs,
    init_ema_with_warmup,
    mitchell_netravali,
)
from video_stabilizer.worker import process_frame_worker

logger = logging.getLogger(__name__)


def _future_result_safe(
    fut: concurrent.futures.Future,
    timeout: float | None = None,
) -> np.ndarray | None:
    try:
        if timeout is None:
            return fut.result()
        return fut.result(timeout=timeout)
    except Exception:
        logger.exception("Worker future raised")
        return None


def _drain_backpressure(
    futures: dict[int, concurrent.futures.Future],
    next_frame_to_write: int,
    out: cv2.VideoWriter,
    pbar: tqdm,
    max_queue_size: int,
    config: Config,
) -> int:
    while len(futures) > max_queue_size:
        if next_frame_to_write in futures:
            res_frame = _future_result_safe(
                futures[next_frame_to_write],
                timeout=config.future_result_timeout_sec,
            )
            if res_frame is not None:
                out.write(res_frame)
            del futures[next_frame_to_write]
            next_frame_to_write += 1
            pbar.update(1)
        else:
            break
    return next_frame_to_write


def process_target_video(
    app_path: str,
    out_path: str,
    ref_avg_stats: np.ndarray,
    config: Config,
) -> None:
    container = None
    out: cv2.VideoWriter | None = None
    executor: concurrent.futures.ThreadPoolExecutor | None = None

    try:
        try:
            container = av.open(app_path)
        except Exception:
            logger.exception("Failed to open target video: %s", app_path)
            return

        stream = container.streams.video[0]
        total_app_frames = (
            stream.frames
            if stream.frames > 0
            else int(float(stream.duration * stream.time_base) * stream.average_rate)
        )
        if total_app_frames <= 0:
            total_app_frames = config.fallback_frame_estimate

        fps = float(stream.average_rate)
        if fps <= 0 or math.isnan(fps):
            fps = config.default_fps

        w, h = stream.width, stream.height
        fourcc = cv2.VideoWriter_fourcc(*config.fourcc)
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not out.isOpened():
            logger.error("VideoWriter failed to open for output: %s", out_path)
            return

        N = get_optimal_sampling_interval(fps, config.target_sampling_sec)
        is_short_video = total_app_frames <= (N * 3)

        logger.info("Processing %s (%sx%s @ %.2ffps)", os.path.basename(app_path), w, h, fps)
        if is_short_video:
            logger.info("Short video (%sF): direct EMA mode", total_app_frames)
        else:
            logger.info("Sampling every %s frames (Mitchell interpolation mode)", N)

        stats_buffer: list[tuple[int, np.ndarray]] = []
        frames_buffer: deque[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = deque()
        warmup_buffer: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=N)
        ema_stats: np.ndarray | None = None
        ema_alpha = config.ema_alpha

        max_workers = os.cpu_count() or 4
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        futures: dict[int, concurrent.futures.Future] = {}

        next_frame_to_write = 0
        max_queue_size = max_workers * config.queue_multiplier

        pbar_total = total_app_frames if total_app_frames > config.tqdm_total_threshold else None
        pbar = tqdm(total=pbar_total, desc=os.path.basename(app_path)[:20])
        f_idx = 0

        def write_completed_frames() -> None:
            nonlocal next_frame_to_write
            while next_frame_to_write in futures and futures[next_frame_to_write].done():
                res_frame = _future_result_safe(futures[next_frame_to_write])
                if res_frame is not None:
                    out.write(res_frame)
                del futures[next_frame_to_write]
                next_frame_to_write += 1
                pbar.update(1)

        try:
            for frame in container.decode(stream):
                target_yuv = extract_yuv_planes(frame)
                if target_yuv is None:
                    futures[f_idx] = executor.submit(
                        process_frame_worker,
                        None,
                        None,
                        None,
                        ref_avg_stats,
                        ref_avg_stats,
                        config,
                    )
                elif is_short_video:
                    s = get_stats_and_coeffs(
                        target_yuv[0], target_yuv[1], target_yuv[2], config.stats_max_dim
                    )
                    if ema_stats is None:
                        ema_stats = s
                    else:
                        ema_stats = ema_alpha * s + (1 - ema_alpha) * ema_stats

                    futures[f_idx] = executor.submit(
                        process_frame_worker,
                        target_yuv[0],
                        target_yuv[1],
                        target_yuv[2],
                        ema_stats,
                        ref_avg_stats,
                        config,
                    )
                else:
                    if f_idx % N == 0:
                        s = get_stats_and_coeffs(
                            target_yuv[0], target_yuv[1], target_yuv[2], config.stats_max_dim
                        )
                        if s is not None:
                            stats_buffer.append((f_idx, s))

                    frames_buffer.append((f_idx, target_yuv))

                    buffer_limit = int(N * config.buffer_frames_multiplier)
                    while len(frames_buffer) > buffer_limit:
                        buf_f_idx, buf_target = frames_buffer[0]
                        interp_s = None
                        target_j = -1

                        for j in range(len(stats_buffer) - 1):
                            if stats_buffer[j][0] <= buf_f_idx < stats_buffer[j + 1][0]:
                                target_j = j
                                break

                        if target_j != -1 and target_j + 2 < len(stats_buffer):
                            p1 = stats_buffer[target_j][1]
                            p2 = stats_buffer[target_j + 1][1]
                            p0 = stats_buffer[target_j - 1][1] if target_j >= 1 else p1
                            p3 = stats_buffer[target_j + 2][1]

                            t = (buf_f_idx - stats_buffer[target_j][0]) / float(N)
                            interp_s = mitchell_netravali(t, [p0, p1, p2, p3])

                            frames_buffer.popleft()
                            warmup_buffer.append(buf_target)
                        else:
                            break

                        if interp_s is not None:
                            futures[buf_f_idx] = executor.submit(
                                process_frame_worker,
                                buf_target[0],
                                buf_target[1],
                                buf_target[2],
                                interp_s,
                                ref_avg_stats,
                                config,
                            )

                write_completed_frames()
                next_frame_to_write = _drain_backpressure(
                    futures, next_frame_to_write, out, pbar, max_queue_size, config
                )
                f_idx += 1

            if not is_short_video:
                while frames_buffer:
                    buf_f_idx, buf_target = frames_buffer.popleft()
                    interp_s = None

                    if len(stats_buffer) >= 2:
                        last_j = len(stats_buffer) - 2
                        p1 = stats_buffer[last_j][1]
                        p2 = stats_buffer[last_j + 1][1]
                        p0 = stats_buffer[last_j - 1][1] if last_j >= 1 else p1
                        p3_extrapolated = p2 + (p2 - p1)
                        t = (buf_f_idx - stats_buffer[last_j][0]) / float(N)
                        interp_s = mitchell_netravali(t, [p0, p1, p2, p3_extrapolated])
                    else:
                        if ema_stats is None:
                            ema_stats = init_ema_with_warmup(
                                list(warmup_buffer), ema_alpha, config.stats_max_dim
                            )
                            if ema_stats is None:
                                ema_stats = get_stats_and_coeffs(
                                    buf_target[0],
                                    buf_target[1],
                                    buf_target[2],
                                    config.stats_max_dim,
                                )
                        s = get_stats_and_coeffs(
                            buf_target[0], buf_target[1], buf_target[2], config.stats_max_dim
                        )
                        ema_stats = ema_alpha * s + (1 - ema_alpha) * ema_stats
                        interp_s = ema_stats

                    futures[buf_f_idx] = executor.submit(
                        process_frame_worker,
                        buf_target[0],
                        buf_target[1],
                        buf_target[2],
                        interp_s,
                        ref_avg_stats,
                        config,
                    )

            while len(futures) > 0:
                if next_frame_to_write in futures:
                    res_frame = _future_result_safe(futures[next_frame_to_write])
                    if res_frame is not None:
                        out.write(res_frame)
                    del futures[next_frame_to_write]
                    next_frame_to_write += 1
                    pbar.update(1)
                else:
                    next_frame_to_write += 1
        finally:
            pbar.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if out is not None:
            out.release()
        if container is not None:
            container.close()
