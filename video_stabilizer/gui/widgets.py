# 再利用可能な GUI 部品

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from video_stabilizer.gui.design_system import metrics, typography


class FileListPanel(QWidget):
    """スクロール可能なファイル一覧（ウィンドウリサイズに追従）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._paths: list[str] = []
        m = metrics()
        typo = typography()

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.space_xs)

        header = QHBoxLayout()
        header.setSpacing(m.space_sm)
        self._count = QLabel("0 件")
        self._count.setObjectName("countLabel")
        self._count.setFont(typo.caption)
        header.addWidget(self._count)
        hint = QLabel("チェックした項目を「選択削除」")
        hint.setObjectName("hintLabel")
        hint.setFont(typo.caption)
        header.addWidget(hint)
        header.addStretch()
        layout.addLayout(header)

        self._frame = QFrame()
        self._frame.setObjectName("fileListPanel")
        self._frame.setMinimumHeight(m.file_list_min_height)
        self._frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        frame_layout.addWidget(self._stack)

        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(m.space_md, m.space_md, m.space_md, m.space_md)
        self._placeholder = QLabel("ファイルまたはフォルダを追加")
        self._placeholder.setObjectName("placeholderLabel")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setFont(typo.caption)
        self._placeholder.setWordWrap(True)
        empty_layout.addStretch()
        empty_layout.addWidget(self._placeholder)
        empty_layout.addStretch()
        self._stack.addWidget(empty)

        self._list = QListWidget()
        self._list.setObjectName("fileList")
        self._list.setFont(typo.body)
        self._list.setAlternatingRowColors(True)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._list.setSpacing(2)
        self._list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._stack.addWidget(self._list)

        layout.addWidget(self._frame, stretch=1)

    def set_paths(self, paths: list[str]) -> None:
        self._paths = list(paths)
        self._rebuild()

    def get_paths(self) -> list[str]:
        return list(self._paths)

    def get_selected_indices(self) -> list[int]:
        indices: list[int] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                indices.append(i)
        return indices

    def _rebuild(self) -> None:
        m = metrics()
        self._list.clear()
        n = len(self._paths)
        self._count.setText(f"{n} 件")
        self._stack.setCurrentIndex(1 if n > 0 else 0)

        for path in self._paths:
            name = os.path.basename(path)
            folder = os.path.dirname(path)
            item = QListWidgetItem(f"{name}\n{folder}")
            item.setToolTip(path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setSizeHint(QSize(0, m.file_list_item_height))
            self._list.addItem(item)

        if n > 0:
            self._list.scrollToTop()
