# Tkinter系

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog


def select_file(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    try:
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
        )
        return file_path or ""
    finally:
        root.destroy()


def select_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    try:
        folder_path = filedialog.askdirectory(title=title)
        return folder_path or ""
    finally:
        root.destroy()
