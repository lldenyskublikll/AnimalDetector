from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from core.app_paths import ensure_project_structure
from ui.main_window import MainWindow
from ui.theme import apply_theme


def main() -> int:
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.*=false")
    ensure_project_structure()
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
