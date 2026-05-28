# GUI プリセット

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from video_stabilizer.config import Config

PresetName = Literal["standard", "natural", "strong"]


def apply_preset(config: Config, preset: PresetName) -> Config:
    if preset == "standard":
        return config
    if preset == "natural":
        return replace(
            config,
            ratio_clip_low=0.7,
            ratio_clip_high=1.4,
            ema_alpha=0.15,
            chroma_soft_clip_threshold=90.0,
        )
    if preset == "strong":
        return replace(
            config,
            ratio_clip_low=0.4,
            ratio_clip_high=2.5,
            ema_alpha=0.28,
            chroma_soft_clip_threshold=70.0,
        )
    return config
