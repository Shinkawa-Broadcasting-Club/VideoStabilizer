# まあなんか処理系

from __future__ import annotations

import logging
import os

from video_stabilizer.config import Config
from video_stabilizer.logging_setup import configure_logging
from video_stabilizer.pipeline import process_target_video
from video_stabilizer.reference import analyze_reference_video
from video_stabilizer.ui import select_file, select_folder

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = Config.from_env()

    ref_path = select_file("参照映像(Lookの元)を選択してください")
    if not ref_path:
        return

    logger.info("--- 参照映像を解析中 ---")
    ref_avg_stats = analyze_reference_video(ref_path, config)
    if ref_avg_stats is None:
        logger.error("参照映像から統計情報を取得できませんでした。")
        return

    target_folder = select_folder("適用先の映像が入っているフォルダを選択してください")
    if not target_folder:
        return

    target_files = [
        f for f in os.listdir(target_folder) if f.lower().endswith(config.valid_extensions)
    ]
    if not target_files:
        logger.warning("対象の動画ファイルが見つかりませんでした。")
        return

    out_dir = os.path.join(target_folder, config.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("合計 %s 個の動画ファイルを処理します。", len(target_files))

    for file_name in target_files:
        app_path = os.path.join(target_folder, file_name)
        out_path = os.path.join(out_dir, f"{config.output_prefix}{file_name}")
        try:
            process_target_video(app_path, out_path, ref_avg_stats, config)
        except Exception:
            logger.exception("ファイル処理中に予期せぬエラー: %s", file_name)
            continue

    logger.info("--- すべての処理が完了しました ---")


if __name__ == "__main__":
    main()
