# GUI ログ Handler

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal


class QtLogBridge(QObject, logging.Handler):
    """Emit log records to the UI thread via Qt signals."""

    log_record = Signal(str)

    def __init__(self, level: int = logging.WARNING) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self, level=level)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_record.emit(self.format(record))
        except Exception:
            self.handleError(record)


def attach_gui_logging(level: int = logging.INFO) -> QtLogBridge:
    handler = QtLogBridge(level=level)
    logging.getLogger().addHandler(handler)
    return handler


def detach_gui_logging(handler: QtLogBridge) -> None:
    logging.getLogger().removeHandler(handler)
