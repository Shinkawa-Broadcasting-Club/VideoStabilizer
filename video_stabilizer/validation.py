# 実行前バリデーション

from __future__ import annotations

import os
from dataclasses import dataclass

from video_stabilizer.config import Config


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def validate_job_inputs(
    ref_path: str,
    target_paths: list[str],
    output_dir: str | None,
    config: Config,
) -> ValidationResult:
    errors: list[str] = []

    if not ref_path or not os.path.isfile(ref_path):
        errors.append("参照映像を選択してください。")
    if not target_paths:
        errors.append("処理対象の動画を1件以上追加してください。")
    for p in target_paths:
        if not os.path.isfile(p):
            errors.append(f"入力ファイルが見つかりません: {p}")

    if output_dir:
        parent = os.path.dirname(os.path.abspath(output_dir)) or "."
        if not os.path.isdir(parent):
            errors.append("出力先の親フォルダが存在しません。")
        elif not os.access(parent, os.W_OK):
            errors.append("出力先の親フォルダに書き込み権限がありません。")

    if not (0.0 < config.ema_alpha <= 1.0):
        errors.append("EMA alpha は 0 より大きく 1 以下である必要があります。")
    if config.ratio_clip_low <= 0 or config.ratio_clip_high <= 0:
        errors.append("ratio clip は正の値である必要があります。")
    if config.ratio_clip_low >= config.ratio_clip_high:
        errors.append("ratio_clip_low は ratio_clip_high より小さくしてください。")
    if config.target_sampling_sec <= 0:
        errors.append("サンプリング間隔は正の値である必要があります。")

    return ValidationResult(ok=len(errors) == 0, errors=errors)
