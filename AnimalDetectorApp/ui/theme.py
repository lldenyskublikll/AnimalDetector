from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget


EVERGREEN = "#132A13"
HUNTER_GREEN = "#31572C"
FERN = "#4F772D"
PALM_LEAF = "#90A955"
LIME_CREAM = "#ECF39E"
DANGER = "#B23A3A"
DANGER_HOVER = "#D14A4A"
WARNING = "#D6A63A"
PREVIEW_DARK = "#0B160B"
DISABLED_BG = "#4A5A3A"
DISABLED_TEXT = "#AAB39A"
CHECKMARK_ICON = (Path(__file__).resolve().parent / "assets" / "checkmark.svg").as_posix()


STYLE_SHEET = f"""
QMainWindow {{
    background: {EVERGREEN};
}}

QWidget {{
    background: {EVERGREEN};
    color: {LIME_CREAM};
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 10pt;
    selection-background-color: {FERN};
    selection-color: #FFFFFF;
}}

#ContentRoot, #ContentStack, #ContentPage {{
    background: {EVERGREEN};
}}

#Sidebar {{
    background: {EVERGREEN};
    border-right: 1px solid {HUNTER_GREEN};
}}

#Sidebar QPushButton {{
    background: transparent;
    color: {LIME_CREAM};
    border: 0;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: left;
}}

#Sidebar QPushButton:hover {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
}}

#Sidebar QPushButton:checked {{
    background: {FERN};
    color: #FFFFFF;
}}

#Sidebar QPushButton:pressed {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
}}

#ToggleButton, #SecondaryButton {{
    background: transparent;
    border: 1px solid {PALM_LEAF};
    color: {LIME_CREAM};
}}

#ToggleButton:hover, #SecondaryButton:hover {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
}}

#PageTitle {{
    background: transparent;
    color: {LIME_CREAM};
    font-size: 24pt;
    font-weight: 700;
}}

#PageSubtitle {{
    background: transparent;
    color: {PALM_LEAF};
    font-size: 13pt;
}}

#HomeLead {{
    background: transparent;
    color: {LIME_CREAM};
    font-size: 12pt;
    line-height: 145%;
}}

#HomeSection {{
    background: transparent;
}}

#HomeSectionTitle {{
    background: transparent;
    color: {LIME_CREAM};
    font-size: 13pt;
    font-weight: 700;
}}

#HomeBody {{
    background: transparent;
    color: {PALM_LEAF};
    font-size: 10.5pt;
    line-height: 145%;
}}

#MutedText {{
    background: transparent;
    color: {PALM_LEAF};
}}

QPushButton {{
    background: {FERN};
    color: #FFFFFF;
    border: 1px solid {FERN};
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 20px;
}}

QPushButton:hover {{
    background: {PALM_LEAF};
    color: {EVERGREEN};
    border-color: {PALM_LEAF};
}}

QPushButton:pressed {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
    border-color: {HUNTER_GREEN};
}}

QPushButton:disabled {{
    background: {DISABLED_BG};
    color: {DISABLED_TEXT};
    border-color: {DISABLED_BG};
}}

#DangerButton {{
    background: {DANGER};
    color: #FFFFFF;
    border-color: {DANGER};
}}

#Sidebar #DangerButton {{
    background: {DANGER};
    color: #FFFFFF;
    border-color: {DANGER};
}}

#DangerButton:hover {{
    background: {DANGER_HOVER};
    border-color: {DANGER_HOVER};
    color: #FFFFFF;
}}

#Sidebar #DangerButton:hover {{
    background: {DANGER_HOVER};
    color: #FFFFFF;
}}

#DangerButton:pressed {{
    background: #8F2F2F;
    border-color: #8F2F2F;
    color: #FFFFFF;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {LIME_CREAM};
    color: {EVERGREEN};
    border: 1px solid {HUNTER_GREEN};
    border-radius: 6px;
    padding: 6px;
    selection-background-color: {FERN};
    selection-color: #FFFFFF;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {PALM_LEAF};
}}

QComboBox:hover {{
    border: 1px solid {PALM_LEAF};
}}

QComboBox::drop-down {{
    border: 0;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
    border: 1px solid {PALM_LEAF};
    selection-background-color: {FERN};
    selection-color: #FFFFFF;
    outline: 0;
}}

QListWidget, QTableWidget, QTableView {{
    background: {PREVIEW_DARK};
    color: {LIME_CREAM};
    border: 1px solid {HUNTER_GREEN};
    border-radius: 8px;
    gridline-color: {HUNTER_GREEN};
    alternate-background-color: {HUNTER_GREEN};
    selection-background-color: {FERN};
    selection-color: #FFFFFF;
}}

QListWidget::item {{
    border-radius: 5px;
    padding: 6px;
}}

QListWidget::item:hover {{
    background: {HUNTER_GREEN};
}}

QListWidget::item:selected {{
    background: {FERN};
    color: #FFFFFF;
}}

QHeaderView::section {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
    border: 1px solid {FERN};
    padding: 6px;
}}

QTabWidget::pane {{
    background: {EVERGREEN};
    border: 1px solid {HUNTER_GREEN};
    border-radius: 8px;
}}

QTabBar::tab {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
    border: 1px solid {HUNTER_GREEN};
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 12px;
}}

QTabBar::tab:hover {{
    background: {PALM_LEAF};
    color: {EVERGREEN};
}}

QTabBar::tab:selected {{
    background: {FERN};
    color: #FFFFFF;
}}

QProgressBar {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
    border: 1px solid {FERN};
    border-radius: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: {PALM_LEAF};
    border-radius: 5px;
}}

QCheckBox, QRadioButton, QGroupBox, QLabel {{
    background: transparent;
    color: {LIME_CREAM};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
}}

QCheckBox::indicator:unchecked {{
    background: {LIME_CREAM};
    border: 1px solid {HUNTER_GREEN};
    border-radius: 3px;
}}

QCheckBox::indicator:checked {{
    background: {FERN};
    border: 1px solid {PALM_LEAF};
    border-radius: 3px;
    image: url("{CHECKMARK_ICON}");
}}

QFormLayout QLabel {{
    color: {LIME_CREAM};
}}

QScrollBar:vertical, QScrollBar:horizontal {{
    background: {EVERGREEN};
    border: 0;
    margin: 0;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {FERN};
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {PALM_LEAF};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}

QStatusBar {{
    background: {EVERGREEN};
    color: {LIME_CREAM};
    border-top: 1px solid {HUNTER_GREEN};
}}

QToolBar {{
    background: {HUNTER_GREEN};
    color: {LIME_CREAM};
    border: 0;
}}

#PreviewArea {{
    background: {PREVIEW_DARK};
    color: {PALM_LEAF};
    border: 1px solid {HUNTER_GREEN};
    border-radius: 8px;
}}

#SuccessText {{
    color: {PALM_LEAF};
}}

#WarningText {{
    color: {WARNING};
}}

#ErrorText {{
    color: {DANGER_HOVER};
}}
"""


def apply_theme(target: QApplication | QWidget) -> None:
    target.setStyleSheet(STYLE_SHEET)
