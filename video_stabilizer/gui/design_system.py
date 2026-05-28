# デザインシステム — タイポグラフィ・色・間隔・QSS

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QScreen
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# --- カラー（ニュートラルのみ） ---
COLOR_BG = "#FEFEFE"
COLOR_SURFACE = "#FFFFFF"
COLOR_SURFACE_SUBTLE = "#F8F8F8"
COLOR_BORDER = "#E5E5E5"
COLOR_BORDER_STRONG = "#D4D4D4"
COLOR_BORDER_FOCUS = "#A3A3A3"

COLOR_TEXT = "#1C1C1C"
COLOR_TEXT_SECONDARY = "#6B6B6B"
COLOR_TEXT_TERTIARY = "#9CA3AF"
COLOR_TEXT_INVERSE = "#FEFEFE"

COLOR_INTERACTIVE = "#2A2A2A"
COLOR_INTERACTIVE_HOVER = "#404040"
COLOR_INTERACTIVE_PRESSED = "#1A1A1A"
COLOR_INTERACTIVE_SUBTLE = "#F3F3F3"
COLOR_INTERACTIVE_SUBTLE_HOVER = "#EBEBEB"

COLOR_PROGRESS_TRACK = "#EBEBEB"
COLOR_PROGRESS_FILL = "#525252"

# --- 基準デザイン（4K で 760×1000 のとき scale=1.0） ---
DESIGN_REF_WIDTH = 760
DESIGN_REF_HEIGHT = 1000

# 基準 px（scale=1.0 時の値）
_REF_FONT_CAPTION = 11
_REF_FONT_BODY = 13
_REF_FONT_SECTION = 13
_REF_FONT_EMPHASIS = 14
_REF_FONT_TITLE = 20
_REF_SPACE_XS = 4
_REF_SPACE_SM = 8
_REF_SPACE_MD = 12
_REF_SPACE_LG = 16
_REF_SPACE_XL = 24
_REF_INPUT_HEIGHT = 36
_REF_BUTTON_HEIGHT = 34
_REF_BUTTON_HEIGHT_LG = 40
_REF_BORDER_RADIUS = 6
_REF_BORDER_RADIUS_LG = 8
_REF_PROGRESS_HEIGHT = 4
_REF_FILE_LIST_MIN_HEIGHT = 120
_REF_FILE_LIST_ITEM_HEIGHT = 44
_REF_LOG_MIN_HEIGHT = 72
_REF_FORM_LABEL_WIDTH = 120
_REF_SCROLLBAR_WIDTH = 8

_metrics: "UiMetrics | None" = None
_FONT_FAMILY_CACHE: str | None = None


def _scaled(base: float, scale: float, *, minimum: int = 1) -> int:
    return max(minimum, round(base * scale))


@dataclass(frozen=True)
class UiMetrics:
    """基準 760×1000 デザインを scale 倍した実寸。"""

    scale: float
    font_caption: int
    font_body: int
    font_section: int
    font_emphasis: int
    font_title: int
    space_xs: int
    space_sm: int
    space_md: int
    space_lg: int
    space_xl: int
    input_height: int
    button_height: int
    button_height_lg: int
    border_radius: int
    border_radius_lg: int
    progress_height: int
    file_list_min_height: int
    file_list_item_height: int
    log_min_height: int
    form_label_width: int
    scrollbar_width: int
    window_width: int
    window_height: int

    def sx(self, value: int) -> int:
        """任意の基準 px を現在スケールに換算。"""
        return _scaled(value, self.scale)


def compute_ui_metrics(scale: float) -> UiMetrics:
    s = scale
    return UiMetrics(
        scale=s,
        font_caption=_scaled(_REF_FONT_CAPTION, s, minimum=9),
        font_body=_scaled(_REF_FONT_BODY, s, minimum=10),
        font_section=_scaled(_REF_FONT_SECTION, s, minimum=10),
        font_emphasis=_scaled(_REF_FONT_EMPHASIS, s, minimum=11),
        font_title=_scaled(_REF_FONT_TITLE, s, minimum=14),
        space_xs=_scaled(_REF_SPACE_XS, s, minimum=2),
        space_sm=_scaled(_REF_SPACE_SM, s, minimum=4),
        space_md=_scaled(_REF_SPACE_MD, s, minimum=6),
        space_lg=_scaled(_REF_SPACE_LG, s, minimum=8),
        space_xl=_scaled(_REF_SPACE_XL, s, minimum=12),
        input_height=_scaled(_REF_INPUT_HEIGHT, s, minimum=28),
        button_height=_scaled(_REF_BUTTON_HEIGHT, s, minimum=26),
        button_height_lg=_scaled(_REF_BUTTON_HEIGHT_LG, s, minimum=30),
        border_radius=_scaled(_REF_BORDER_RADIUS, s, minimum=4),
        border_radius_lg=_scaled(_REF_BORDER_RADIUS_LG, s, minimum=4),
        progress_height=max(2, _scaled(_REF_PROGRESS_HEIGHT, s, minimum=2)),
        file_list_min_height=_scaled(_REF_FILE_LIST_MIN_HEIGHT, s, minimum=80),
        file_list_item_height=_scaled(_REF_FILE_LIST_ITEM_HEIGHT, s, minimum=32),
        log_min_height=_scaled(_REF_LOG_MIN_HEIGHT, s, minimum=48),
        form_label_width=_scaled(_REF_FORM_LABEL_WIDTH, s, minimum=80),
        scrollbar_width=_scaled(_REF_SCROLLBAR_WIDTH, s, minimum=6),
        window_width=_scaled(DESIGN_REF_WIDTH, s, minimum=480),
        window_height=_scaled(DESIGN_REF_HEIGHT, s, minimum=520),
    )


def resolve_ui_scale(screen: QScreen | None = None) -> float:
    """基準 760×1000 が収まらない画面では 1 未満を返し UI 全体を縮小する。"""
    if screen is None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app else None
    if screen is None:
        return 1.0
    avail = screen.availableGeometry()
    return min(
        avail.width() / DESIGN_REF_WIDTH,
        avail.height() / DESIGN_REF_HEIGHT,
        1.0,
    )


def init_ui_environment(screen: QScreen | None = None) -> UiMetrics:
    global _metrics
    _metrics = compute_ui_metrics(resolve_ui_scale(screen))
    return _metrics


def metrics() -> UiMetrics:
    if _metrics is None:
        return compute_ui_metrics(1.0)
    return _metrics


@dataclass(frozen=True)
class Typography:
    caption: QFont
    body: QFont
    section: QFont
    emphasis: QFont
    button: QFont
    title: QFont


def resolve_ui_font_family() -> str:
    global _FONT_FAMILY_CACHE
    if _FONT_FAMILY_CACHE is not None:
        return _FONT_FAMILY_CACHE

    if sys.platform == "win32":
        candidates = ("Segoe UI", "Yu Gothic UI", "Meiryo UI", "MS UI Gothic", "Tahoma")
    elif sys.platform == "darwin":
        candidates = ("SF Pro Text", "Helvetica Neue", "Hiragino Sans")
    else:
        candidates = ("DejaVu Sans", "Ubuntu", "Noto Sans")

    families = {f.lower(): f for f in QFontDatabase.families()}
    for name in candidates:
        hit = families.get(name.lower())
        if hit:
            _FONT_FAMILY_CACHE = hit
            return hit

    _FONT_FAMILY_CACHE = QFont().defaultFamily()
    return _FONT_FAMILY_CACHE


def _make_font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(resolve_ui_font_family(), size)
    font.setWeight(weight)
    return font


def typography() -> Typography:
    m = metrics()
    return Typography(
        caption=_make_font(m.font_caption),
        body=_make_font(m.font_body),
        section=_make_font(m.font_section, QFont.Weight.DemiBold),
        emphasis=_make_font(m.font_emphasis, QFont.Weight.DemiBold),
        button=_make_font(m.font_body, QFont.Weight.Medium),
        title=_make_font(m.font_title, QFont.Weight.DemiBold),
    )


def apply_global_font(app) -> None:
    app.setFont(_make_font(metrics().font_body))


def section_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("sectionLabel")
    label.setFont(typography().section)
    return label


def hint_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("hintLabel")
    label.setFont(typography().caption)
    label.setWordWrap(True)
    return label


def status_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("statusLabel")
    label.setFont(typography().body)
    return label


def app_header_block(
    subtitle: str,
    title: str,
    parent: QWidget | None = None,
) -> QWidget:
    block = QWidget(parent)
    m = metrics()
    layout = QVBoxLayout(block)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(m.space_xs)
    layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

    sub = QLabel(subtitle, block)
    sub.setObjectName("appSubtitle")
    sub.setFont(typography().caption)
    sub.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(sub)

    main = QLabel(title, block)
    main.setObjectName("appTitle")
    main.setFont(typography().title)
    main.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(main)

    return block


@dataclass(frozen=True)
class WindowGeometry:
    width: int
    height: int
    min_width: int
    min_height: int


def resolve_window_geometry(screen: QScreen | None = None) -> WindowGeometry:
    """基準 760×1000 を scale 適用したサイズを初期値・最小値とする。"""
    if screen is not None:
        init_ui_environment(screen)
    m = metrics()
    return WindowGeometry(
        width=m.window_width,
        height=m.window_height,
        min_width=m.window_width,
        min_height=m.window_height,
    )


def center_window_on_screen(window: QMainWindow, screen: QScreen | None = None) -> None:
    if screen is None:
        screen = window.screen()
    if screen is None:
        return
    frame = window.frameGeometry()
    frame.moveCenter(screen.availableGeometry().center())
    window.move(frame.topLeft())


def apply_initial_window_geometry(window: QMainWindow) -> None:
    screen = window.screen()
    if screen is None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app else None
    geo = resolve_window_geometry(screen)
    window.resize(geo.width, geo.height)
    window.setMinimumSize(geo.min_width, geo.min_height)
    center_window_on_screen(window, screen)


def build_stylesheet() -> str:
    m = metrics()
    line_edit_inner = max(8, m.input_height - m.sx(18))
    button_inner = max(8, m.button_height - m.sx(18))
    run_inner = max(8, m.button_height_lg - m.sx(20))
    item_radius = max(2, m.border_radius - m.sx(2))
    progress_radius = max(1, m.progress_height // 2)

    return f"""
    QMainWindow, QWidget {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
        font-size: {m.font_body}px;
    }}

    QLabel#sectionLabel {{
        color: {COLOR_TEXT};
        font-weight: 600;
        padding-top: {m.space_sm}px;
        padding-bottom: {m.space_xs}px;
    }}

    QLabel#hintLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {m.font_caption}px;
    }}

    QLabel#statusLabel {{
        color: {COLOR_TEXT};
        font-size: {m.font_body}px;
    }}

    QLabel#fileNameLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {m.font_caption}px;
    }}

    QLabel#countLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {m.font_caption}px;
    }}

    QLabel#placeholderLabel {{
        color: {COLOR_TEXT_TERTIARY};
        font-size: {m.font_caption}px;
    }}

    QLabel#appSubtitle {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {m.font_caption}px;
        padding: 0;
        margin: 0;
    }}

    QLabel#appTitle {{
        color: {COLOR_TEXT};
        font-size: {m.font_title}px;
        font-weight: 600;
        padding: 0;
        margin: 0;
    }}

    QTabWidget::pane {{
        border: none;
        background: {COLOR_BG};
        top: -1px;
    }}

    QTabBar {{
        background: transparent;
    }}

    QTabBar::tab {{
        background: transparent;
        color: {COLOR_TEXT_SECONDARY};
        padding: {m.space_sm}px {m.space_lg}px;
        margin-right: {m.space_xs}px;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: {m.font_body}px;
        font-weight: 500;
    }}

    QTabBar::tab:selected {{
        color: {COLOR_TEXT};
        border-bottom: 2px solid {COLOR_INTERACTIVE};
        font-weight: 600;
    }}

    QTabBar::tab:hover:!selected {{
        color: {COLOR_TEXT};
    }}

    QLineEdit {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: {m.border_radius}px;
        padding: {m.space_sm}px {m.space_md}px;
        min-height: {line_edit_inner}px;
        font-size: {m.font_body}px;
        selection-background-color: {COLOR_INTERACTIVE_SUBTLE_HOVER};
        selection-color: {COLOR_TEXT};
    }}

    QLineEdit:focus {{
        border-color: {COLOR_BORDER_FOCUS};
    }}

    QLineEdit:disabled {{
        background: {COLOR_SURFACE_SUBTLE};
        color: {COLOR_TEXT_TERTIARY};
    }}

    QPlainTextEdit#logBox {{
        background: {COLOR_SURFACE_SUBTLE};
        border: 1px solid {COLOR_BORDER};
        border-radius: {m.border_radius}px;
        padding: {m.space_sm}px {m.space_md}px;
        font-size: {m.font_caption}px;
        color: {COLOR_TEXT_SECONDARY};
        font-family: "Consolas", "Cascadia Mono", monospace;
    }}

    QPushButton {{
        background: {COLOR_SURFACE};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: {m.border_radius}px;
        padding: {m.space_sm}px {m.space_md}px;
        min-height: {button_inner}px;
        font-size: {m.font_body}px;
        font-weight: 500;
    }}

    QPushButton:hover {{
        background: {COLOR_INTERACTIVE_SUBTLE};
        border-color: {COLOR_BORDER_FOCUS};
    }}

    QPushButton:pressed {{
        background: {COLOR_INTERACTIVE_SUBTLE_HOVER};
    }}

    QPushButton:disabled {{
        color: {COLOR_TEXT_TERTIARY};
        background: {COLOR_SURFACE_SUBTLE};
        border-color: {COLOR_BORDER};
    }}

    QPushButton[secondary="true"] {{
        background: transparent;
        border-color: {COLOR_BORDER_STRONG};
        color: {COLOR_TEXT_SECONDARY};
        font-weight: 400;
    }}

    QPushButton[secondary="true"]:hover {{
        background: {COLOR_INTERACTIVE_SUBTLE};
        color: {COLOR_TEXT};
    }}

    QPushButton#runButton {{
        background: {COLOR_INTERACTIVE};
        color: {COLOR_TEXT_INVERSE};
        border: 1px solid {COLOR_INTERACTIVE};
        font-weight: 600;
        font-size: {m.font_emphasis}px;
        min-height: {run_inner}px;
    }}

    QPushButton#runButton:hover {{
        background: {COLOR_INTERACTIVE_HOVER};
        border-color: {COLOR_INTERACTIVE_HOVER};
    }}

    QPushButton#runButton:pressed {{
        background: {COLOR_INTERACTIVE_PRESSED};
        border-color: {COLOR_INTERACTIVE_PRESSED};
    }}

    QPushButton#runButton:disabled {{
        background: {COLOR_BORDER_STRONG};
        border-color: {COLOR_BORDER_STRONG};
        color: {COLOR_TEXT_INVERSE};
    }}

    QComboBox {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: {m.border_radius}px;
        padding: {m.space_sm}px {m.space_md}px;
        min-height: {button_inner}px;
        font-size: {m.font_body}px;
    }}

    QComboBox:focus, QComboBox:on {{
        border-color: {COLOR_BORDER_FOCUS};
    }}

    QComboBox::drop-down {{
        border: none;
        width: {m.sx(24)}px;
    }}

    QComboBox QAbstractItemView {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        selection-background-color: {COLOR_INTERACTIVE_SUBTLE};
        selection-color: {COLOR_TEXT};
        outline: none;
    }}

    QCheckBox {{
        font-size: {m.font_body}px;
        spacing: {m.space_sm}px;
        color: {COLOR_TEXT};
    }}

    QCheckBox::indicator {{
        width: {m.sx(16)}px;
        height: {m.sx(16)}px;
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 3px;
        background: {COLOR_SURFACE};
    }}

    QCheckBox::indicator:checked {{
        background: {COLOR_INTERACTIVE};
        border-color: {COLOR_INTERACTIVE};
    }}

    QCheckBox::indicator:hover {{
        border-color: {COLOR_BORDER_FOCUS};
    }}

    QFrame#fileListPanel {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: {m.border_radius_lg}px;
    }}

    QListWidget#fileList {{
        background: transparent;
        border: none;
        padding: {m.space_xs}px;
        font-size: {m.font_body}px;
        outline: none;
    }}

    QListWidget#fileList::item {{
        padding: {m.space_sm}px {m.space_md}px;
        border-radius: {item_radius}px;
        color: {COLOR_TEXT};
    }}

    QListWidget#fileList::item:alternate {{
        background: {COLOR_SURFACE_SUBTLE};
    }}

    QListWidget#fileList::item:hover {{
        background: {COLOR_INTERACTIVE_SUBTLE};
    }}

    QListWidget#fileList::item:selected {{
        background: {COLOR_INTERACTIVE_SUBTLE_HOVER};
        color: {COLOR_TEXT};
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: {m.scrollbar_width}px;
        margin: {m.space_xs}px 2px;
    }}

    QScrollBar::handle:vertical {{
        background: {COLOR_BORDER_STRONG};
        border-radius: {max(2, m.scrollbar_width // 2)}px;
        min-height: {m.sx(32)}px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {COLOR_BORDER_FOCUS};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    QProgressBar {{
        border: none;
        background: {COLOR_PROGRESS_TRACK};
        border-radius: {progress_radius}px;
        min-height: {m.progress_height}px;
        max-height: {m.progress_height}px;
    }}

    QProgressBar::chunk {{
        background: {COLOR_PROGRESS_FILL};
        border-radius: {progress_radius}px;
    }}

    QFormLayout QLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {m.font_body}px;
        min-width: {m.form_label_width}px;
    }}

    QFrame#sectionDivider {{
        background: {COLOR_BORDER};
        max-height: 1px;
        min-height: 1px;
        border: none;
    }}
    """
