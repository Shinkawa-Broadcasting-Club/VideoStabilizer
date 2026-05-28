# UI 設定の永続化

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from video_stabilizer.config import Config


def settings_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata) / "VideoStabilizer"
    else:
        base = Path.home() / ".video_stabilizer"
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"


def load_ui_settings(default_config: Config) -> dict[str, Any]:
    path = settings_path()
    if not path.is_file():
        return _defaults_from_config(default_config)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merged = _defaults_from_config(default_config)
        merged.update(data)
        return merged
    except (OSError, json.JSONDecodeError, TypeError):
        return _defaults_from_config(default_config)


def save_ui_settings(data: dict[str, Any]) -> None:
    path = settings_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _defaults_from_config(config: Config) -> dict[str, Any]:
    return {
        "ref_path": "",
        "target_paths": [],
        "output_dir": config.unified_output_dir or "",
        "output_prefix": config.output_prefix,
        "output_suffix": config.output_suffix,
        "collision_policy": config.collision_policy,
        "resume_mode": config.resume_mode,
        "preset": config.preset,
        "preserve_audio": config.preserve_audio,
        "ema_alpha": config.ema_alpha,
        "ratio_clip_low": config.ratio_clip_low,
        "ratio_clip_high": config.ratio_clip_high,
        "target_sampling_sec": config.target_sampling_sec,
        "on_frame_failure": config.on_frame_failure,
    }


def config_from_ui_settings(ui: dict[str, Any], base: Config) -> Config:
    from dataclasses import replace

    def _f(key: str, default: float) -> float:
        try:
            value = float(ui.get(key, default))
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return value

    out_dir = (ui.get("output_dir") or "").strip()
    return replace(
        base,
        unified_output_dir=out_dir or None,
        output_prefix=str(ui.get("output_prefix", base.output_prefix)),
        output_suffix=str(ui.get("output_suffix", base.output_suffix)),
        collision_policy=ui.get("collision_policy", base.collision_policy),  # type: ignore[arg-type]
        resume_mode=ui.get("resume_mode", base.resume_mode),  # type: ignore[arg-type]
        preset=ui.get("preset", base.preset),  # type: ignore[arg-type]
        preserve_audio=bool(ui.get("preserve_audio", base.preserve_audio)),
        ema_alpha=_f("ema_alpha", base.ema_alpha),
        ratio_clip_low=_f("ratio_clip_low", base.ratio_clip_low),
        ratio_clip_high=_f("ratio_clip_high", base.ratio_clip_high),
        target_sampling_sec=_f("target_sampling_sec", base.target_sampling_sec),
        on_frame_failure=ui.get("on_frame_failure", base.on_frame_failure),  # type: ignore[arg-type]
        use_gui_progress=True,
    )
