# Config

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

FrameFailurePolicy = Literal["hold", "black", "abort"]


@dataclass(frozen=True)
class Config:
    """Processing and I/O defaults."""

    target_sampling_sec: float = 0.5
    ema_alpha: float = 0.2
    chroma_soft_clip_threshold: float = 80.0
    chroma_soft_clip_diff: float = 32.0
    ratio_clip_low: float = 0.5
    ratio_clip_high: float = 2.0
    stats_max_dim: int = 256
    queue_multiplier: int = 5
    fourcc: str = "mp4v"
    output_subdir: str = "output_corrected"
    output_prefix: str = "corrected_"
    valid_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    # Reference video: use every N-th frame for stats (original: idx % 2 == 0)
    ref_frame_stride: int = 2
    # Buffer length vs N (original: len(frames_buffer) > N * 2.5)
    buffer_frames_multiplier: float = 2.5
    default_fps: float = 30.0
    fallback_frame_estimate: int = 1000
    tqdm_total_threshold: int = 1000
    stats_epsilon: float = 1e-6
    ref_stats_min_std: float = 1e-6
    # hold: repeat last good frame; black: TV black; abort: stop and delete partial output
    on_frame_failure: FrameFailurePolicy = "hold"
    mitchell_t_max: float = 1.0
    # Mux corrected video with audio stream copied from the source file
    preserve_audio: bool = True

    @classmethod
    def from_env(cls) -> Config:
        """Optional overrides via environment (all optional)."""

        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return float(raw)

        def _i(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return int(raw)

        def _b(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _policy(name: str, default: FrameFailurePolicy) -> FrameFailurePolicy:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            value = raw.strip().lower()
            if value in ("hold", "black", "abort"):
                return value  # type: ignore[return-value]
            return default

        return cls(
            target_sampling_sec=_f("VS_TARGET_SAMPLING_SEC", cls.target_sampling_sec),
            ema_alpha=_f("VS_EMA_ALPHA", cls.ema_alpha),
            chroma_soft_clip_threshold=_f(
                "VS_CHROMA_SOFT_CLIP_THRESHOLD", cls.chroma_soft_clip_threshold
            ),
            chroma_soft_clip_diff=_f("VS_CHROMA_SOFT_CLIP_DIFF", cls.chroma_soft_clip_diff),
            ratio_clip_low=_f("VS_RATIO_CLIP_LOW", cls.ratio_clip_low),
            ratio_clip_high=_f("VS_RATIO_CLIP_HIGH", cls.ratio_clip_high),
            stats_max_dim=_i("VS_STATS_MAX_DIM", cls.stats_max_dim),
            queue_multiplier=_i("VS_QUEUE_MULTIPLIER", cls.queue_multiplier),
            on_frame_failure=_policy("VS_ON_FRAME_FAILURE", cls.on_frame_failure),
            mitchell_t_max=_f("VS_MITCHELL_T_MAX", cls.mitchell_t_max),
            preserve_audio=_b("VS_PRESERVE_AUDIO", cls.preserve_audio),
        )
