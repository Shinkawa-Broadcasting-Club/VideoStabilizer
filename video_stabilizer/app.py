# アプリケーションエントリ

from __future__ import annotations

import argparse
import logging
import os
import sys

from video_stabilizer.config import Config
from video_stabilizer.files import collect_targets
from video_stabilizer.job_runner import run_batch
from video_stabilizer.logging_setup import configure_logging
from video_stabilizer.ui import select_file, select_folder, shutdown_ui

logger = logging.getLogger(__name__)


def run_cli(config: Config | None = None) -> None:
    """Legacy two-dialog CLI flow."""
    base = config or Config.from_env()
    cfg = Config(**{**base.to_serializable_dict(), "use_gui_progress": False})

    try:
        ref_path = select_file("参照映像(Lookの元)を選択してください")
        if not ref_path:
            return

        target_folder = select_folder("適用先の映像が入っているフォルダを選択してください")
        if not target_folder:
            logger.info("フォルダ選択がキャンセルされました。")
            return

        targets = collect_targets([target_folder], cfg.valid_extensions)
        if not targets:
            logger.warning("対象の動画ファイルが見つかりませんでした。")
            return

        manifest_path = os.path.join(target_folder, "manifest.json")
        result = run_batch(ref_path, targets, cfg, manifest_path=manifest_path)
        logger.info(
            "--- 処理完了 (成功: %s, 失敗: %s, スキップ: %s) ---",
            result.ok_count,
            result.fail_count,
            result.skip_count,
        )
    finally:
        shutdown_ui()


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Video Stabilizer — Look 転写")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="GUI ではなく従来のファイルダイアログ CLI を起動",
    )
    args = parser.parse_args(argv)

    if args.cli:
        run_cli()
        return

    from video_stabilizer.gui import launch_gui

    launch_gui()


if __name__ == "__main__":
    main()
