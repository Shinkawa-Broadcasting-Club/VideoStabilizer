# Tkinter系

from __future__ import annotations

import logging
import sys
import tkinter as tk
from tkinter import filedialog

logger = logging.getLogger(__name__)

_root: tk.Tk | None = None


def _ensure_root() -> tk.Tk:
    """Return a single hidden root window for all file dialogs."""
    global _root
    if _root is None:
        _root = tk.Tk()
        _root.withdraw()
        if sys.platform == "win32":
            try:
                _root.attributes("-topmost", True)
            except tk.TclError:
                pass
    _root.update_idletasks()
    _root.update()
    return _root


def shutdown_ui() -> None:
    global _root
    if _root is not None:
        try:
            _root.destroy()
        except tk.TclError:
            pass
        _root = None


def select_file(title: str) -> str:
    root = _ensure_root()
    try:
        file_path = filedialog.askopenfilename(
            parent=root,
            title=title,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
        )
        return file_path or ""
    except tk.TclError:
        logger.exception("File dialog failed")
        return ""


def select_folder(title: str) -> str:
    root = _ensure_root()
    try:
        folder_path = filedialog.askdirectory(
            parent=root,
            title=title,
            mustexist=True,
        )
        return folder_path or ""
    except tk.TclError:
        logger.exception("Folder dialog failed")
        return ""
