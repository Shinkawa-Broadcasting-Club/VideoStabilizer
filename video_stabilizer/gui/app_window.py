# PySide6 メインウィンドウ

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from video_stabilizer.config import Config
from video_stabilizer.gui.controller import ProcessingController
from video_stabilizer.gui.design_system import (
    apply_global_font,
    apply_initial_window_geometry,
    app_header_block,
    build_stylesheet,
    hint_label,
    init_ui_environment,
    metrics,
    section_label,
    status_label,
    typography,
)
from video_stabilizer.gui.log_handler import QtLogBridge, attach_gui_logging, detach_gui_logging
from video_stabilizer.gui.settings_store import (
    config_from_ui_settings,
    load_ui_settings,
    save_ui_settings,
)
from video_stabilizer.gui.widgets import FileListPanel
from video_stabilizer.logging_setup import configure_logging
from video_stabilizer.ui import VIDEO_FILTER

logger = logging.getLogger(__name__)

COLLISION_LABELS = {
    "rename": "同名は連番で保存",
    "skip": "スキップ",
    "overwrite": "上書き",
}
COLLISION_VALUES = {v: k for k, v in COLLISION_LABELS.items()}

RESUME_LABELS = {
    "skip_done": "完了分を飛ばす",
    "retry_failed": "失敗分だけ再実行",
    "run_all": "最初から全部",
}
RESUME_VALUES = {v: k for k, v in RESUME_LABELS.items()}

PRESET_LABELS = {
    "standard": "標準",
    "natural": "自然",
    "strong": "強め",
}
PRESET_VALUES = {v: k for k, v in PRESET_LABELS.items()}

FAILURE_LABELS = {
    "hold": "前の画を繰り返す",
    "black": "黒画面",
    "abort": "処理を止める",
}
FAILURE_VALUES = {v: k for k, v in FAILURE_LABELS.items()}

PRESET_NUMERIC = {
    "standard": (0.2, 0.5, 2.0),
    "natural": (0.15, 0.7, 1.4),
    "strong": (0.28, 0.4, 2.5),
}


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("sectionDivider")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


def _ghost_button(text: str, ref_width: int, handler) -> QPushButton:
    m = metrics()
    btn = QPushButton(text)
    btn.setFixedWidth(m.sx(ref_width))
    btn.setMinimumHeight(m.button_height)
    btn.setProperty("secondary", True)
    btn.setFont(typography().button)
    btn.clicked.connect(handler)
    return btn


def _field_button(text: str, ref_width: int, handler) -> QPushButton:
    m = metrics()
    btn = QPushButton(text)
    btn.setFixedWidth(m.sx(ref_width))
    btn.setMinimumHeight(m.input_height)
    btn.setFont(typography().body)
    btn.clicked.connect(handler)
    return btn


class VideoStabilizerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Stabilizer")

        self._base_config = Config.from_env()
        self._ui_settings = load_ui_settings(self._base_config)
        self._controller = ProcessingController()
        self._log_bridge: QtLogBridge = attach_gui_logging(level=logging.INFO)
        self._log_bridge.log_record.connect(self._append_log)

        self._target_paths: list[str] = list(self._ui_settings.get("target_paths", []))

        self._build_ui()
        self._load_settings_into_widgets()
        self._connect_controller()

    def _build_ui(self) -> None:
        m = metrics()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(m.space_xl, m.space_lg, m.space_xl, m.space_lg)
        layout.setSpacing(m.space_lg)

        layout.addWidget(
            app_header_block(
                "市立札幌新川高等学校放送局 N73 研究発表",
                "Video Stabilizer",
            )
        )

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        self._tab_home = QWidget()
        self._tab_detail = QWidget()
        self._tabs.addTab(self._tab_home, "ホーム")
        self._tabs.addTab(self._tab_detail, "詳細設定")

        self._build_home_tab()
        self._build_detail_tab()

    def _build_home_tab(self) -> None:
        m = metrics()
        layout = QVBoxLayout(self._tab_home)
        layout.setContentsMargins(m.space_xs, m.space_lg, m.space_xs, m.space_xs)
        layout.setSpacing(m.space_md)

        # --- 参照動画 ---
        layout.addWidget(section_label("参照動画"))
        layout.addWidget(hint_label("合わせたい色味の動画を1本選びます"))

        ref_row = QHBoxLayout()
        ref_row.setSpacing(m.space_sm)
        self._ref_path = QLineEdit()
        self._ref_path.setPlaceholderText("参照動画のパス")
        self._ref_path.setMinimumHeight(m.input_height)
        ref_row.addWidget(self._ref_path, stretch=1)
        ref_row.addWidget(_field_button("参照…", 80, self._pick_ref))
        layout.addLayout(ref_row)

        layout.addWidget(_divider())

        # --- 処理対象 ---
        layout.addWidget(section_label("補正する動画"))

        tools = QHBoxLayout()
        tools.setSpacing(m.space_sm)
        for label, width, slot in (
            ("ファイル追加", 100, self._add_files),
            ("フォルダ追加", 100, self._add_folder),
            ("選択削除", 88, self._remove_selected),
            ("すべてクリア", 100, self._clear_targets),
        ):
            tools.addWidget(_ghost_button(label, width, slot))
        tools.addStretch()
        layout.addLayout(tools)

        self._file_panel = FileListPanel()
        layout.addWidget(self._file_panel, stretch=2)

        layout.addWidget(_divider())

        # --- 出力 ---
        layout.addWidget(section_label("出力"))
        layout.addWidget(hint_label("空欄の場合、各動画と同じ場所の output_corrected フォルダに保存します"))

        out_row = QHBoxLayout()
        out_row.setSpacing(m.space_sm)
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("出力フォルダ（任意）")
        self._output_dir.setMinimumHeight(m.input_height)
        out_row.addWidget(self._output_dir, stretch=1)
        out_row.addWidget(_field_button("参照…", 80, self._pick_output_dir))
        layout.addLayout(out_row)

        layout.addWidget(_divider())

        # --- 実行 ---
        run_row = QHBoxLayout()
        run_row.setSpacing(m.space_sm)
        self._run_btn = QPushButton("実行")
        self._run_btn.setObjectName("runButton")
        self._run_btn.setFixedWidth(m.sx(112))
        self._run_btn.setMinimumHeight(m.button_height_lg)
        self._run_btn.setFont(typography().emphasis)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self._run_btn)

        self._cancel_btn = _ghost_button("停止", 80, self._on_cancel)
        self._cancel_btn.setMinimumHeight(m.button_height_lg)
        self._cancel_btn.setEnabled(False)
        run_row.addStretch()
        layout.addLayout(run_row)

        # --- 進捗 ---
        progress_box = QVBoxLayout()
        progress_box.setSpacing(m.space_sm)

        self._status_label = status_label("参照と動画を選んで「実行」")
        progress_box.addWidget(self._status_label)

        self._ref_progress = QProgressBar()
        self._ref_progress.setFixedHeight(m.progress_height)
        self._ref_progress.setTextVisible(False)
        progress_box.addWidget(self._ref_progress)

        self._current_file = QLabel("")
        self._current_file.setObjectName("fileNameLabel")
        self._current_file.setFont(typography().caption)
        progress_box.addWidget(self._current_file)

        self._file_progress = QProgressBar()
        self._file_progress.setFixedHeight(m.progress_height)
        self._file_progress.setTextVisible(False)
        progress_box.addWidget(self._file_progress)

        layout.addLayout(progress_box)

        # --- ログ ---
        log_header = hint_label("ログ")
        layout.addWidget(log_header)

        self._log_box = QPlainTextEdit()
        self._log_box.setObjectName("logBox")
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(m.log_min_height)
        self._log_box.setFont(typography().caption)
        layout.addWidget(self._log_box, stretch=1)

    def _build_detail_tab(self) -> None:
        m = metrics()
        layout = QVBoxLayout(self._tab_detail)
        layout.setContentsMargins(m.space_xs, m.space_lg, m.space_xs, m.space_xs)
        layout.setSpacing(m.space_lg)

        layout.addWidget(hint_label("普段は変更不要です。うまくいかないときだけ調整してください。"))
        layout.addWidget(_divider())

        form = QFormLayout()
        form.setSpacing(m.space_md)
        form.setVerticalSpacing(m.space_md)
        form.setHorizontalSpacing(m.space_lg)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(m.space_md)
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(m.sx(160))
        self._preset_combo.addItems(list(PRESET_LABELS.values()))
        self._preset_combo.currentTextChanged.connect(self._on_preset_display_change)
        preset_layout.addWidget(self._preset_combo)
        self._preserve_audio = QCheckBox("音声を残す")
        preset_layout.addWidget(self._preserve_audio)
        preset_layout.addStretch()
        form.addRow("プリセット", preset_row)

        names_row = QWidget()
        names_layout = QHBoxLayout(names_row)
        names_layout.setContentsMargins(0, 0, 0, 0)
        names_layout.setSpacing(m.space_sm)
        prefix_lbl = QLabel("先頭")
        prefix_lbl.setFont(typography().caption)
        names_layout.addWidget(prefix_lbl)
        self._output_prefix = QLineEdit()
        self._output_prefix.setFixedWidth(m.sx(100))
        self._output_prefix.setMinimumHeight(m.input_height)
        names_layout.addWidget(self._output_prefix)
        suffix_lbl = QLabel("末尾")
        suffix_lbl.setFont(typography().caption)
        names_layout.addWidget(suffix_lbl)
        self._output_suffix = QLineEdit()
        self._output_suffix.setFixedWidth(m.sx(100))
        self._output_suffix.setMinimumHeight(m.input_height)
        names_layout.addWidget(self._output_suffix)
        names_layout.addStretch()
        form.addRow("出力ファイル名", names_row)

        self._collision_combo = QComboBox()
        self._collision_combo.setMinimumWidth(m.sx(220))
        self._collision_combo.addItems(list(COLLISION_LABELS.values()))
        form.addRow("既にあるファイル", self._collision_combo)

        self._resume_combo = QComboBox()
        self._resume_combo.setMinimumWidth(m.sx(220))
        self._resume_combo.addItems(list(RESUME_LABELS.values()))
        form.addRow("再実行", self._resume_combo)

        ema_row = QWidget()
        ema_layout = QHBoxLayout(ema_row)
        ema_layout.setContentsMargins(0, 0, 0, 0)
        ema_layout.setSpacing(m.space_sm)
        self._ema = QLineEdit()
        self._ema.setFixedWidth(m.sx(64))
        self._ema.setMinimumHeight(m.input_height)
        ema_layout.addWidget(self._ema)
        ema_layout.addWidget(hint_label("0〜1・大きいほど反応が速い"))
        ema_layout.addStretch()
        form.addRow("EMA", ema_row)

        clip_row = QWidget()
        clip_layout = QHBoxLayout(clip_row)
        clip_layout.setContentsMargins(0, 0, 0, 0)
        clip_layout.setSpacing(m.space_sm)
        self._clip_lo = QLineEdit()
        self._clip_lo.setFixedWidth(m.sx(56))
        self._clip_lo.setMinimumHeight(m.input_height)
        clip_layout.addWidget(self._clip_lo)
        clip_layout.addWidget(QLabel("〜"))
        self._clip_hi = QLineEdit()
        self._clip_hi.setFixedWidth(m.sx(56))
        self._clip_hi.setMinimumHeight(m.input_height)
        clip_layout.addWidget(self._clip_hi)
        clip_layout.addStretch()
        form.addRow("補正の強さ", clip_row)

        sampling_row = QWidget()
        sampling_layout = QHBoxLayout(sampling_row)
        sampling_layout.setContentsMargins(0, 0, 0, 0)
        sampling_layout.setSpacing(m.space_sm)
        self._sampling = QLineEdit()
        self._sampling.setFixedWidth(m.sx(64))
        self._sampling.setMinimumHeight(m.input_height)
        sampling_layout.addWidget(self._sampling)
        unit = QLabel("秒")
        unit.setFont(typography().caption)
        sampling_layout.addWidget(unit)
        sampling_layout.addStretch()
        form.addRow("解析間隔", sampling_row)

        self._frame_fail_combo = QComboBox()
        self._frame_fail_combo.setMinimumWidth(m.sx(200))
        self._frame_fail_combo.addItems(list(FAILURE_LABELS.values()))
        form.addRow("フレームエラー時", self._frame_fail_combo)

        layout.addLayout(form)
        layout.addStretch()

    def _load_settings_into_widgets(self) -> None:
        s = self._ui_settings
        self._ref_path.setText(s.get("ref_path", ""))
        self._output_dir.setText(s.get("output_dir", ""))
        self._output_prefix.setText(s.get("output_prefix", "corrected_"))
        self._output_suffix.setText(s.get("output_suffix", ""))

        preset_key = s.get("preset", "standard")
        self._preset_combo.setCurrentText(PRESET_LABELS.get(preset_key, "標準"))

        coll_key = s.get("collision_policy", "rename")
        self._collision_combo.setCurrentText(COLLISION_LABELS.get(coll_key, "同名は連番で保存"))

        resume_key = s.get("resume_mode", "skip_done")
        self._resume_combo.setCurrentText(RESUME_LABELS.get(resume_key, "完了分を飛ばす"))

        self._preserve_audio.setChecked(bool(s.get("preserve_audio", True)))
        self._ema.setText(str(s.get("ema_alpha", 0.2)))
        self._clip_lo.setText(str(s.get("ratio_clip_low", 0.5)))
        self._clip_hi.setText(str(s.get("ratio_clip_high", 2.0)))
        self._sampling.setText(str(s.get("target_sampling_sec", 0.5)))

        fail_key = s.get("on_frame_failure", "hold")
        self._frame_fail_combo.setCurrentText(FAILURE_LABELS.get(fail_key, "前の画を繰り返す"))

        self._file_panel.set_paths(self._target_paths)

    def _connect_controller(self) -> None:
        self._controller.status_changed.connect(self._on_status)
        self._controller.ref_progress.connect(self._on_ref_progress)
        self._controller.file_progress.connect(self._on_file_progress)
        self._controller.error_occurred.connect(self._on_error)
        self._controller.batch_done.connect(self._on_batch_done)

    def _on_preset_display_change(self, _choice: str) -> None:
        key = PRESET_VALUES.get(self._preset_combo.currentText(), "standard")
        if key in PRESET_NUMERIC:
            ema, lo, hi = PRESET_NUMERIC[key]
            self._ema.setText(str(ema))
            self._clip_lo.setText(str(lo))
            self._clip_hi.setText(str(hi))

    def _pick_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "参照動画", "", VIDEO_FILTER)
        if path:
            self._ref_path.setText(path)

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "出力フォルダ",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if path:
            self._output_dir.setText(path)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "動画を追加", "", VIDEO_FILTER)
        for p in paths:
            if p and p not in self._target_paths:
                self._target_paths.append(p)
        self._file_panel.set_paths(self._target_paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "フォルダ",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not folder:
            return
        from video_stabilizer.files import collect_targets

        for p in collect_targets([folder], self._base_config.valid_extensions):
            if p not in self._target_paths:
                self._target_paths.append(p)
        self._file_panel.set_paths(self._target_paths)

    def _remove_selected(self) -> None:
        for idx in reversed(self._file_panel.get_selected_indices()):
            if 0 <= idx < len(self._target_paths):
                del self._target_paths[idx]
        self._file_panel.set_paths(self._target_paths)

    def _clear_targets(self) -> None:
        self._target_paths.clear()
        self._file_panel.set_paths(self._target_paths)

    def _gather_ui_settings(self) -> dict[str, Any]:
        return {
            "ref_path": self._ref_path.text(),
            "target_paths": list(self._target_paths),
            "output_dir": self._output_dir.text(),
            "output_prefix": self._output_prefix.text(),
            "output_suffix": self._output_suffix.text(),
            "collision_policy": COLLISION_VALUES.get(
                self._collision_combo.currentText(), "rename"
            ),
            "resume_mode": RESUME_VALUES.get(self._resume_combo.currentText(), "skip_done"),
            "preset": PRESET_VALUES.get(self._preset_combo.currentText(), "standard"),
            "preserve_audio": self._preserve_audio.isChecked(),
            "ema_alpha": self._ema.text(),
            "ratio_clip_low": self._clip_lo.text(),
            "ratio_clip_high": self._clip_hi.text(),
            "target_sampling_sec": self._sampling.text(),
            "on_frame_failure": FAILURE_VALUES.get(
                self._frame_fail_combo.currentText(), "hold"
            ),
        }

    def _build_config(self) -> Config:
        return config_from_ui_settings(self._gather_ui_settings(), self._base_config)

    def _manifest_path(self) -> str | None:
        out = self._output_dir.text().strip()
        if out:
            return os.path.join(out, "manifest.json")
        if self._target_paths:
            return os.path.join(os.path.dirname(self._target_paths[0]), "manifest.json")
        return None

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)

    def _on_run(self) -> None:
        save_ui_settings(self._gather_ui_settings())
        if not self._controller.start(
            self._ref_path.text().strip(),
            self._target_paths,
            self._build_config(),
            manifest_path=self._manifest_path(),
        ):
            return
        self._set_running(True)
        self._status_label.setText("処理中…")
        self._ref_progress.setValue(0)
        self._file_progress.setValue(0)
        self._current_file.setText("")

    def _on_cancel(self) -> None:
        self._controller.cancel()
        self._status_label.setText("停止しています…")

    def _append_log(self, text: str) -> None:
        self._log_box.appendPlainText(text)
        scrollbar = self._log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_progress(self, bar: QProgressBar, cur: int, tot: int | None) -> None:
        if tot and tot > 0:
            bar.setRange(0, tot)
            bar.setValue(min(cur, tot))
        else:
            bar.setRange(0, 0)

    def _on_status(self, msg: str, _level: str) -> None:
        self._status_label.setText(msg)
        self._append_log(msg)

    def _on_ref_progress(self, cur: int, tot: object) -> None:
        total = tot if isinstance(tot, int) else None
        if total and total > 0:
            self._set_progress(self._ref_progress, cur, total)
        else:
            self._ref_progress.setRange(0, 500)
            self._ref_progress.setValue(min(cur, 500))

    def _on_file_progress(self, _path: str, cur: int, tot: object, name: str) -> None:
        self._current_file.setText(name)
        total = tot if isinstance(tot, int) else None
        if total and total > 0:
            self._set_progress(self._file_progress, cur, total)
        else:
            self._file_progress.setRange(0, 500)
            self._file_progress.setValue(min(cur, 500))

    def _on_error(self, message: str) -> None:
        self._status_label.setText(str(message))
        self._append_log(f"エラー: {message}")
        if not self._controller.is_running():
            self._set_running(False)

    def _on_batch_done(self, result: object) -> None:
        from video_stabilizer.job_runner import BatchResult

        if not isinstance(result, BatchResult):
            return
        self._set_running(False)
        self._status_label.setText(
            f"完了 — 成功 {result.ok_count} / 失敗 {result.fail_count} / "
            f"スキップ {result.skip_count}"
        )
        if result.ok_count:
            self._file_progress.setRange(0, 100)
            self._file_progress.setValue(100)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._controller.cancel()
        self._controller.wait(5000)
        detach_gui_logging(self._log_bridge)
        save_ui_settings(self._gather_ui_settings())
        event.accept()


def launch_gui() -> None:
    configure_logging()
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv)
    init_ui_environment(app.primaryScreen())
    apply_global_font(app)
    app.setStyleSheet(build_stylesheet())

    window = VideoStabilizerWindow()
    window.show()
    apply_initial_window_geometry(window)
    app.exec()
