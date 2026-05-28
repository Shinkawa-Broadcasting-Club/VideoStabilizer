# バックグラウンド処理と UI 通信

from __future__ import annotations

import logging
import os
import threading

from PySide6.QtCore import QObject, QThread, Signal

from video_stabilizer.config import Config
from video_stabilizer.job_runner import run_batch
from video_stabilizer.validation import validate_job_inputs

logger = logging.getLogger(__name__)


class BatchWorker(QObject):
    status_changed = Signal(str, str)
    ref_progress = Signal(int, object)
    file_progress = Signal(str, int, object, str)
    batch_done = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        ref_path: str,
        target_paths: list[str],
        config: Config,
        manifest_path: str | None,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._ref_path = ref_path
        self._target_paths = target_paths
        self._config = config
        self._manifest_path = manifest_path
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:

            def on_file_progress(path: str, cur: int, tot: int | None) -> None:
                if path == "__ref__":
                    self.ref_progress.emit(cur, tot)
                else:
                    self.file_progress.emit(path, cur, tot, os.path.basename(path))

            def on_status(msg: str, level: str) -> None:
                self.status_changed.emit(msg, level)

            result = run_batch(
                self._ref_path,
                self._target_paths,
                self._config,
                manifest_path=self._manifest_path,
                cancel_event=self._cancel_event,
                on_file_progress=on_file_progress,
                on_status=on_status,
            )
            self.batch_done.emit(result)
        except Exception as exc:
            logger.exception("Batch worker failed")
            self.error_occurred.emit(str(exc))


class ProcessingController(QObject):
    status_changed = Signal(str, str)
    ref_progress = Signal(int, object)
    file_progress = Signal(str, int, object, str)
    batch_done = Signal(object)
    error_occurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancel_event = threading.Event()
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(
        self,
        ref_path: str,
        target_paths: list[str],
        config: Config,
        *,
        manifest_path: str | None = None,
    ) -> bool:
        validation = validate_job_inputs(
            ref_path,
            target_paths,
            config.unified_output_dir,
            config,
        )
        if not validation.ok:
            for err in validation.errors:
                self.error_occurred.emit(err)
            return False

        if self.is_running():
            self.error_occurred.emit("処理が既に実行中です。")
            return False

        self._cancel_event.clear()

        self._thread = QThread()
        self._worker = BatchWorker(
            ref_path,
            target_paths,
            config,
            manifest_path,
            self._cancel_event,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(self.status_changed)
        self._worker.ref_progress.connect(self.ref_progress)
        self._worker.file_progress.connect(self.file_progress)
        self._worker.batch_done.connect(self.batch_done)
        self._worker.error_occurred.connect(self.error_occurred)
        self._worker.batch_done.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()
        return True

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def wait(self, timeout_ms: int = 5000) -> bool:
        if self._thread is None:
            return True
        return self._thread.wait(timeout_ms)
