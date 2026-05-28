# バッチ処理オーケストレーション

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

from video_stabilizer.config import Config
from video_stabilizer.files import build_output_path, collect_targets, resolve_collision
from video_stabilizer.manifest import (
    BatchManifest,
    JobEntry,
    config_hash,
    load_manifest,
    new_manifest,
    save_manifest,
    should_skip_job,
)
from video_stabilizer.pipeline import process_target_video
from video_stabilizer.presets import apply_preset
from video_stabilizer.reference import analyze_reference_video

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int | None], None]
StatusCallback = Callable[[str, str], None]


@dataclass
class BatchResult:
    ok_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    cancelled: bool = False


def _effective_config(config: Config) -> Config:
    cfg = apply_preset(config, config.preset)
    if config.use_gui_progress:
        cfg = replace(cfg, use_gui_progress=True)
    return cfg


def _merge_jobs_with_manifest(
    new_jobs: list[JobEntry],
    existing_jobs: list[JobEntry],
) -> list[JobEntry]:
    """Carry over status/error from previous manifest by input path."""
    existing_by_input = {j.input_path: j for j in existing_jobs}
    merged: list[JobEntry] = []
    for job in new_jobs:
        prev = existing_by_input.get(job.input_path)
        if prev is None:
            merged.append(job)
            continue
        merged.append(
            JobEntry(
                input_path=job.input_path,
                output_path=prev.output_path or job.output_path,
                status=prev.status,
                error=prev.error,
            )
        )
    return merged


def run_batch(
    ref_path: str,
    target_paths: list[str],
    config: Config,
    *,
    manifest_path: str | None = None,
    cancel_event: threading.Event | None = None,
    on_file_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
) -> BatchResult:
    """Run reference analysis and process all target videos."""
    cfg = _effective_config(config)
    result = BatchResult()

    paths = collect_targets(target_paths, cfg.valid_extensions)
    if not paths:
        logger.warning("処理対象の動画がありません。")
        return result

    def _status(msg: str, level: str = "info") -> None:
        if on_status:
            on_status(msg, level)
        if level == "error":
            logger.error(msg)
        else:
            logger.info(msg)

    ref_progress = None
    if on_file_progress:

        def ref_progress(cur: int, tot: int | None) -> None:
            on_file_progress("__ref__", cur, tot)

    ref_stats = analyze_reference_video(
        ref_path,
        cfg,
        progress_cb=ref_progress,
        cancel_event=cancel_event,
    )
    if ref_stats is None:
        result.fail_count = len(paths)
        _status("参照映像の解析に失敗しました。", "error")
        return result

    if cancel_event is not None and cancel_event.is_set():
        result.cancelled = True
        return result

    jobs: list[JobEntry] = []
    for inp in paths:
        out = build_output_path(
            inp,
            output_dir=cfg.unified_output_dir,
            output_subdir=cfg.output_subdir,
            prefix=cfg.output_prefix,
            suffix=cfg.output_suffix,
        )
        resolved = resolve_collision(out, cfg.collision_policy)
        if resolved is None:
            jobs.append(JobEntry(inp, out, status="skipped"))
        else:
            jobs.append(JobEntry(inp, resolved))

    manifest: BatchManifest | None = None
    if manifest_path:
        existing = load_manifest(manifest_path)
        if existing and existing.config_hash == config_hash(cfg.to_serializable_dict()):
            # 設定が同じ manifest があれば、前回の status/error を input_path で引き継ぐ。
            jobs = _merge_jobs_with_manifest(jobs, existing.jobs)
            manifest = existing
            manifest.jobs = jobs
        else:
            manifest = new_manifest(jobs, config_hash(cfg.to_serializable_dict()))
            save_manifest(manifest_path, manifest)
    else:
        manifest = new_manifest(jobs, config_hash(cfg.to_serializable_dict()))

    for i, job in enumerate(jobs):
        if cancel_event is not None and cancel_event.is_set():
            result.cancelled = True
            job.status = "cancelled"
            break

        if manifest and should_skip_job(
            job,
            cfg.resume_mode,
            output_exists=os.path.isfile(job.output_path),
        ):
            job.status = "skipped"
            result.skip_count += 1
            _status(f"スキップ（完了済み）: {os.path.basename(job.input_path)}")
            continue

        if job.status == "skipped":
            result.skip_count += 1
            continue

        _status(f"処理中 ({i + 1}/{len(jobs)}): {os.path.basename(job.input_path)}")

        file_progress = None
        if on_file_progress:

            def file_progress(cur: int, tot: int | None, _path: str = job.input_path) -> None:
                on_file_progress(_path, cur, tot)

            file_progress = file_progress

        job.status = "running"
        if manifest_path and manifest:
            save_manifest(manifest_path, manifest)

        ok = process_target_video(
            job.input_path,
            job.output_path,
            ref_stats,
            cfg,
            progress_cb=file_progress,
            cancel_event=cancel_event,
        )

        if cancel_event is not None and cancel_event.is_set():
            job.status = "cancelled"
            result.cancelled = True
            break

        if ok:
            job.status = "done"
            result.ok_count += 1
            _status(f"完了: {os.path.basename(job.output_path)}")
        else:
            job.status = "failed"
            job.error = "processing failed"
            result.fail_count += 1
            _status(f"失敗: {os.path.basename(job.input_path)}", "error")

        if manifest_path and manifest:
            save_manifest(manifest_path, manifest)

    if manifest_path and manifest:
        save_manifest(manifest_path, manifest)

    _status(
        f"バッチ完了 — 成功: {result.ok_count}, 失敗: {result.fail_count}, "
        f"スキップ: {result.skip_count}"
    )
    return result
