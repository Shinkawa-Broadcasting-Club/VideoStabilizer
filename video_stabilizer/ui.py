# Qt ファイルダイアログ（CLI 用）

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QFileDialog

logger = logging.getLogger(__name__)

_owns_app: bool = False

VIDEO_FILTER = "動画ファイル (*.mp4 *.avi *.mov *.mkv);;すべて (*.*)"


def _ensure_app() -> QApplication:
    global _owns_app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        _owns_app = True
    return app


def shutdown_ui() -> None:
    global _owns_app
    if _owns_app:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
            app.quit()
        _owns_app = False


def select_file(title: str) -> str:
    _ensure_app()
    try:
        path, _ = QFileDialog.getOpenFileName(None, title, "", VIDEO_FILTER)
        return path or ""
    except Exception:
        logger.exception("File dialog failed")
        return ""


def select_folder(title: str) -> str:
    _ensure_app()
    try:
        path = QFileDialog.getExistingDirectory(
            None,
            title,
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        return path or ""
    except Exception:
        logger.exception("Folder dialog failed")
        return ""
