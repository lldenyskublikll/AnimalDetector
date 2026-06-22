from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.model_manager import ModelManager
from core.result_manager import ResultManager
from ui.home_page import HomePage
from ui.photo_detection_page import PhotoDetectionPage
from ui.realtime_detection_page import RealtimeDetectionPage
from ui.results_page import ResultsPage
from ui.settings_page import SettingsPage
from ui.theme import apply_theme
from ui.video_detection_page import VideoDetectionPage


ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"


class NavButton(QPushButton):
    def __init__(self, text: str, icon: QIcon) -> None:
        super().__init__(text)
        self.full_text = text
        self.setIcon(icon)
        self.setIconSize(QSize(20, 20))
        self.setCheckable(True)
        self.setToolTip(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setObjectName("NavButton")

    def set_collapsed(self, collapsed: bool) -> None:
        self.setText("" if collapsed else self.full_text)


class MainWindow(QMainWindow):
    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        model_manager: ModelManager | None = None,
        result_manager: ResultManager | None = None,
    ) -> None:
        super().__init__()
        self.config_manager = config_manager or ConfigManager()
        self.model_manager = model_manager or ModelManager()
        self.result_manager = result_manager or ResultManager()
        self.collapsed = False
        self.nav_buttons: list[NavButton] = []

        self.setWindowTitle("AnimalDetector")
        self.resize(1120, 720)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(232)

        self.stack = QStackedWidget()
        self.stack.setObjectName("ContentStack")
        self._build_pages()
        self._build_layout()
        self._apply_styles()
        self._select_page(0)

    def _build_pages(self) -> None:
        self.results_page = ResultsPage(self.result_manager)
        self.photo_page = PhotoDetectionPage(self.config_manager, self.model_manager)
        self.video_page = VideoDetectionPage(self.config_manager, self.model_manager)
        self.realtime_page = RealtimeDetectionPage(self.config_manager, self.model_manager)
        self.settings_page = SettingsPage(self.config_manager, self.model_manager)
        self.settings_page.settings_saved.connect(self.photo_page.refresh_settings)
        self.settings_page.settings_saved.connect(self.video_page.refresh_settings)
        self.settings_page.settings_saved.connect(self.realtime_page.refresh_settings)
        self.photo_page.detection_finished.connect(self.results_page.refresh)
        self.video_page.detection_finished.connect(self.results_page.refresh)
        self.realtime_page.detection_finished.connect(self.results_page.refresh)

        self.pages = [
            HomePage(),
            self.results_page,
            self.photo_page,
            self.video_page,
            self.realtime_page,
            self.settings_page,
        ]
        for page in self.pages:
            self.stack.addWidget(page)

    def _build_layout(self) -> None:
        root = QWidget()
        root.setObjectName("ContentRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(12, 14, 12, 14)
        sidebar_layout.setSpacing(8)

        self.toggle_button = QPushButton("Collapse")
        self.toggle_button.setObjectName("ToggleButton")
        self.toggle_button.setIcon(self._asset_icon("arrow_back.svg"))
        self.toggle_button.clicked.connect(self._toggle_sidebar)
        sidebar_layout.addWidget(self.toggle_button)
        sidebar_layout.addSpacing(8)

        top_items: list[tuple[str, str, Callable[[], None]]] = [
            ("Home", "home.svg", lambda: self._select_page(0)),
            ("Detection Results", "results.svg", lambda: self._select_page(1)),
            ("Photo Detection", "photo.svg", lambda: self._select_page(2)),
            ("Video Detection", "video.svg", lambda: self._select_page(3)),
            ("Real-Time Detection", "stream.svg", lambda: self._select_page(4)),
        ]

        for text, icon_name, callback in top_items:
            sidebar_layout.addWidget(self._nav_button(text, self._asset_icon(icon_name), callback))

        sidebar_layout.addStretch(1)

        settings = self._nav_button(
            "Settings",
            self._asset_icon("settings.svg"),
            lambda: self._select_page(5),
        )
        exit_button = self._nav_button(
            "Exit",
            self._asset_icon("exit.svg"),
            QApplication.instance().quit,
        )
        exit_button.setObjectName("DangerButton")
        exit_button.setCheckable(False)
        sidebar_layout.addWidget(settings)
        sidebar_layout.addWidget(exit_button)

    def _nav_button(self, text: str, icon: QIcon, callback: Callable[[], None]) -> NavButton:
        button = NavButton(text, icon)
        button.clicked.connect(callback)
        self.nav_buttons.append(button)
        return button

    def _select_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if index == 4:
            self.realtime_page.ensure_sources_loaded()
        for button_index, button in enumerate(self.nav_buttons):
            if button.full_text == "Exit":
                continue
            button.setChecked(button_index == index)

    def _toggle_sidebar(self) -> None:
        self.collapsed = not self.collapsed
        self.sidebar.setFixedWidth(72 if self.collapsed else 232)
        self.toggle_button.setText("" if self.collapsed else "Collapse")
        self.toggle_button.setIcon(self._asset_icon("arrow_forward.svg" if self.collapsed else "arrow_back.svg"))
        for button in self.nav_buttons:
            button.set_collapsed(self.collapsed)

    def _icon(self, standard_icon: QStyle.StandardPixmap) -> QIcon:
        return self.style().standardIcon(standard_icon)

    @staticmethod
    def _asset_icon(file_name: str) -> QIcon:
        return QIcon(str(ICON_DIR / file_name))

    def _apply_styles(self) -> None:
        apply_theme(self)
