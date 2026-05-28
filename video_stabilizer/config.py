# Config

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Literal

FrameFailurePolicy = Literal["hold", "black", "abort"]
CollisionPolicy = Literal["overwrite", "skip", "rename"]
ResumeMode = Literal["skip_done", "retry_failed", "run_all"]
PresetName = Literal["standard", "natural", "strong"]
ProgressMode = Literal["frame", "time"]


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
    output_suffix: str = ""
    valid_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    ref_frame_stride: int = 2
    buffer_frames_multiplier: float = 2.5
    default_fps: float = 30.0
    fallback_frame_estimate: int = 1000
    tqdm_total_threshold: int = 1000
    stats_epsilon: float = 1e-6
    ref_stats_min_std: float = 1e-6
    ref_trim_percentile_low: float = 5.0
    ref_trim_percentile_high: float = 95.0
    on_frame_failure: FrameFailurePolicy = "hold"
    mitchell_t_max: float = 1.0
    preserve_audio: bool = True
    collision_policy: CollisionPolicy = "rename"
    resume_mode: ResumeMode = "skip_done"
    preset: PresetName = "standard"
    use_gui_progress: bool = False
    progress_mode: ProgressMode = "frame"
    vfr_threshold: float = 0.05
    exp_knee_high: float = 226.0
    exp_knee_low: float = 26.0
    exp_soft_high: float = 9.0
    exp_soft_low: float = 10.0
    unified_output_dir: str | None = None

    def to_serializable_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

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

        def _s(name: str, default: str | None) -> str | None:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return raw

        def _policy(name: str, default: FrameFailurePolicy) -> FrameFailurePolicy:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            value = raw.strip().lower()
            if value in ("hold", "black", "abort"):
                return value  # type: ignore[return-value]
            return default

        def _collision(name: str, default: CollisionPolicy) -> CollisionPolicy:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            value = raw.strip().lower()
            if value in ("overwrite", "skip", "rename"):
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
            collision_policy=_collision("VS_COLLISION_POLICY", cls.collision_policy),
            exp_knee_high=_f("VS_EXP_KNEE_HIGH", cls.exp_knee_high),
            exp_knee_low=_f("VS_EXP_KNEE_LOW", cls.exp_knee_low),
            exp_soft_high=_f("VS_EXP_SOFT_HIGH", cls.exp_soft_high),
            exp_soft_low=_f("VS_EXP_SOFT_LOW", cls.exp_soft_low),
            unified_output_dir=_s("VS_OUTPUT_DIR", cls.unified_output_dir),
            use_gui_progress=_b("VS_USE_GUI_PROGRESS", cls.use_gui_progress),
        )
