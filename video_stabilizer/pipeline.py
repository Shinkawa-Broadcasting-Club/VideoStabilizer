# パイプライン系

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

from video_stabilizer.config import Config, FrameFailurePolicy
from video_stabilizer.mux import finalize_output_with_audio
from video_stabilizer.stats import (
    extract_yuv_planes,
    get_optimal_sampling_interval,
    get_stats_and_coeffs,
    init_ema_with_warmup,
    interpolate_stats_for_frame,
    interpolate_stats_tail,
)
from video_stabilizer.worker import process_frame_worker

logger = logging.getLogger(__name__)
PendingFuture = concurrent.futures.Future | None

# OpenCV on Windows is unreliable with mp4/tmp paths; use AVI for the intermediate file.
_INTERMEDIATE_SUFFIX = ".vs-video-only.avi"
_WRITER_CODEC_FALLBACKS = ("mp4v", "XVID", "MJPG", "DIVX")


def _video_only_temp_path(out_path: str) -> str:
    base, _ = os.path.splitext(out_path)
    return base + _INTERMEDIATE_SUFFIX


def _create_video_writer(
    path: str,
    fps: float,
    frame_size: tuple[int, int],
    preferred_fourcc: str,
) -> cv2.VideoWriter | None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    width, height = frame_size
    tried: list[str] = []
    for codec in (preferred_fourcc, *_WRITER_CODEC_FALLBACKS):
        if codec in tried:
            continue
        tried.append(codec)
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            if codec != preferred_fourcc:
                logger.info("VideoWriter using fallback codec %s for %s", codec, path)
            return writer
    return None


class _FrameWriteState:
    """Tracks output frames and applies on_frame_failure policy."""

    def __init__(
        self,
        out: cv2.VideoWriter,
        pbar: tqdm,
        config: Config,
        height: int,
        width: int,
    ) -> None:
        self._out = out
        self._pbar = pbar
        self._config = config
        self._policy: FrameFailurePolicy = config.on_frame_failure
        self._last_good: np.ndarray | None = None
        self._black = np.zeros((height, width, 3), dtype=np.uint8)
        self.aborted = False

    def _placeholder(self) -> np.ndarray | None:
        if self._policy == "black":
            return self._black.copy()
        if self._last_good is not None:
            return self._last_good.copy()
        if self._policy == "hold":
            return self._black.copy()
        return None

    def resolve(self, frame: np.ndarray | None, frame_idx: int) -> bool:
        """Write one output frame. Returns False if processing should abort."""
        if frame is not None:
            self._out.write(frame)
            self._last_good = frame
            self._pbar.update(1)
            return True

        logger.warning("Frame %s processing failed", frame_idx)
        if self._policy == "abort":
            self.aborted = True
            return False

        placeholder = self._placeholder()
        if placeholder is None:
            self.aborted = True
            return False

        self._out.write(placeholder)
        self._pbar.update(1)
        return True


def _future_result_safe(fut: concurrent.futures.Future) -> np.ndarray | None:
    try:
        return fut.result()
    except Exception:
        logger.exception("Worker future raised")
        return None


def _write_next_ready(
    futures: dict[int, PendingFuture],
    next_frame_to_write: int,
    write_state: _FrameWriteState,
) -> int:
    while next_frame_to_write in futures:
        pending = futures[next_frame_to_write]
        if pending is not None and not pending.done():
            break
        res_frame = _future_result_safe(pending) if pending is not None else None
        if not write_state.resolve(res_frame, next_frame_to_write):
            return next_frame_to_write
        del futures[next_frame_to_write]
        next_frame_to_write += 1
    return next_frame_to_write


def _drain_backpressure(
    futures: dict[int, PendingFuture],
    next_frame_to_write: int,
    write_state: _FrameWriteState,
    max_queue_size: int,
) -> int:
    while len(futures) > max_queue_size:
        if next_frame_to_write not in futures:
            break
        pending = futures[next_frame_to_write]
        if pending is not None and not pending.done():
            pending.result()
        res_frame = _future_result_safe(pending) if pending is not None else None
        if not write_state.resolve(res_frame, next_frame_to_write):
            return next_frame_to_write
        del futures[next_frame_to_write]
        next_frame_to_write += 1
    return next_frame_to_write


def _drain_all_futures(
    futures: dict[int, PendingFuture],
    next_frame_to_write: int,
    expected_frames: int,
    write_state: _FrameWriteState,
) -> int:
    while next_frame_to_write < expected_frames:
        if next_frame_to_write in futures:
            pending = futures[next_frame_to_write]
            if pending is not None and not pending.done():
                pending.result()
            res_frame = _future_result_safe(pending) if pending is not None else None
            if not write_state.resolve(res_frame, next_frame_to_write):
                return next_frame_to_write
            del futures[next_frame_to_write]
        else:
            if not write_state.resolve(None, next_frame_to_write):
                return next_frame_to_write
        next_frame_to_write += 1

    while futures:
        key = min(futures.keys())
        if key < next_frame_to_write:
            logger.warning("Orphan future at index %s (expected >= %s)", key, next_frame_to_write)
            del futures[key]
            continue
        pending = futures[key]
        if pending is not None and not pending.done():
            pending.result()
        res_frame = _future_result_safe(pending) if pending is not None else None
        if not write_state.resolve(res_frame, key):
            return next_frame_to_write
        del futures[key]

    return next_frame_to_write


def _remove_partial_output(out_path: str) -> None:
    try:
        if os.path.isfile(out_path):
            os.remove(out_path)
    except OSError:
        logger.exception("Failed to remove partial output: %s", out_path)


def process_target_video(
    app_path: str,
    out_path: str,
    ref_avg_stats: np.ndarray,
    config: Config,
) -> bool:
    container = None
    out: cv2.VideoWriter | None = None
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    success = False

    try:
        try:
            container = av.open(app_path)
        except Exception:
            logger.exception("Failed to open target video: %s", app_path)
            return False

        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if fps <= 0 or math.isnan(fps):
            fps = config.default_fps

        w, h = stream.width, stream.height
        video_only_path = _video_only_temp_path(out_path)
        _remove_partial_output(video_only_path)
        out = _create_video_writer(video_only_path, fps, (w, h), config.fourcc)
        if out is None:
            logger.error("VideoWriter failed to open for output: %s", video_only_path)
            return False

        N = get_optimal_sampling_interval(fps, config.target_sampling_sec)
        total_app_frames = (
            stream.frames
            if stream.frames > 0
            else int(float(stream.duration * stream.time_base) * stream.average_rate)
        )
        if total_app_frames <= 0:
            total_app_frames = config.fallback_frame_estimate
        is_short_video = total_app_frames <= (N * 3)

        stats_buffer: list[tuple[int, np.ndarray]] = []
        frames_buffer: deque[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = deque()
        warmup_buffer: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=N)
        ema_stats: np.ndarray | None = None
        ema_alpha = config.ema_alpha

        max_workers = os.cpu_count() or 4
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        futures: dict[int, PendingFuture] = {}

        next_frame_to_write = 0
        max_queue_size = max_workers * config.queue_multiplier
        f_idx = 0

        pbar_total = total_app_frames if total_app_frames > config.tqdm_total_threshold else None
        pbar = tqdm(total=pbar_total, desc=os.path.basename(app_path)[:20])

        if is_short_video:
            logger.info("Short video (%sF est.): direct EMA mode", total_app_frames)
        else:
            logger.info("Sampling every %s frames (Mitchell interpolation mode)", N)
        write_state = _FrameWriteState(out, pbar, config, h, w)

        def submit_frame(
            frame_idx: int,
            yuv: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            current_stats: np.ndarray,
        ) -> None:
            if yuv is None:
                if config.on_frame_failure == "abort":
                    write_state.aborted = True
                    return
                futures[frame_idx] = None
                return
            y, u, v = yuv
            futures[frame_idx] = executor.submit(
                process_frame_worker,
                y,
                u,
                v,
                current_stats,
                ref_avg_stats,
                config,
            )

        def flush_buffered_frame(
            buf_f_idx: int,
            buf_target: tuple[np.ndarray, np.ndarray, np.ndarray],
            interp_s: np.ndarray,
        ) -> None:
            submit_frame(buf_f_idx, buf_target, interp_s)
            warmup_buffer.append(buf_target)

        try:
            logger.info(
                "Processing %s (%sx%s @ %.2ffps)",
                os.path.basename(app_path),
                w,
                h,
                fps,
            )

            for frame in container.decode(stream):
                target_yuv = extract_yuv_planes(frame)
                if write_state.aborted:
                    break

                if is_short_video:
                    if target_yuv is not None:
                        s = get_stats_and_coeffs(
                            target_yuv[0],
                            target_yuv[1],
                            target_yuv[2],
                            config.stats_max_dim,
                        )
                        if ema_stats is None:
                            ema_stats = s
                        else:
                            ema_stats = ema_alpha * s + (1 - ema_alpha) * ema_stats
                        submit_frame(f_idx, target_yuv, ema_stats)
                    else:
                        submit_frame(f_idx, None, ref_avg_stats)
                else:
                    if target_yuv is not None:
                        if f_idx % N == 0:
                            s = get_stats_and_coeffs(
                                target_yuv[0],
                                target_yuv[1],
                                target_yuv[2],
                                config.stats_max_dim,
                            )
                            stats_buffer.append((f_idx, s))
                        frames_buffer.append((f_idx, target_yuv))

                        buffer_limit = int(N * config.buffer_frames_multiplier)
                        while len(frames_buffer) > buffer_limit:
                            buf_f_idx, buf_target = frames_buffer[0]
                            interp_s = interpolate_stats_for_frame(
                                buf_f_idx,
                                stats_buffer,
                                N,
                                config.mitchell_t_max,
                            )
                            if interp_s is None:
                                break
                            frames_buffer.popleft()
                            flush_buffered_frame(buf_f_idx, buf_target, interp_s)
                    else:
                        submit_frame(f_idx, None, ref_avg_stats)

                if write_state.aborted:
                    break

                next_frame_to_write = _write_next_ready(
                    futures, next_frame_to_write, write_state
                )
                if write_state.aborted:
                    break
                next_frame_to_write = _drain_backpressure(
                    futures, next_frame_to_write, write_state, max_queue_size
                )
                if write_state.aborted:
                    break
                f_idx += 1

            expected_frames = f_idx

            if not write_state.aborted and not is_short_video:
                while frames_buffer:
                    buf_f_idx, buf_target = frames_buffer.popleft()
                    interp_s = interpolate_stats_tail(
                        buf_f_idx,
                        stats_buffer,
                        N,
                        config.mitchell_t_max,
                    )
                    if interp_s is None:
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
                            buf_target[0],
                            buf_target[1],
                            buf_target[2],
                            config.stats_max_dim,
                        )
                        ema_stats = ema_alpha * s + (1 - ema_alpha) * ema_stats
                        interp_s = ema_stats

                    flush_buffered_frame(buf_f_idx, buf_target, interp_s)

            if not write_state.aborted:
                next_frame_to_write = _drain_all_futures(
                    futures,
                    next_frame_to_write,
                    expected_frames,
                    write_state,
                )

            success = (
                not write_state.aborted
                and next_frame_to_write == expected_frames
                and len(futures) == 0
            )
            if not success:
                logger.error(
                    "Processing incomplete for %s: wrote %s/%s frames",
                    app_path,
                    next_frame_to_write,
                    expected_frames,
                )
        finally:
            pbar.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if out is not None:
            out.release()
        if container is not None:
            container.close()
        video_only_path = _video_only_temp_path(out_path)
        if success:
            success = finalize_output_with_audio(
                app_path,
                video_only_path,
                out_path,
                preserve_audio=config.preserve_audio,
            )
        if not success:
            _remove_partial_output(out_path)
        _remove_partial_output(video_only_path)

    return success
