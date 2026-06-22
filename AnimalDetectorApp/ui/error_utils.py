from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from core.app_paths import project_path
from core.model_manager import ModelManager


def exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def show_warning(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.warning(parent, title, message)


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


def validate_model_selection(parent: QWidget, model_manager: ModelManager, model_path: str) -> bool:
    if not model_path:
        show_warning(parent, "Model is not selected", "Select a default model in Settings first.")
        return False
    if not model_manager.model_exists(model_path):
        show_warning(parent, "Model file is missing", f"Model file not found:\n{model_path}")
        return False
    return True


def validate_existing_source(parent: QWidget, source_path: str, source_description: str) -> bool:
    if not source_path:
        show_warning(parent, "Source is not selected", f"Select {source_description}.")
        return False
    if not project_path(source_path).exists():
        show_warning(parent, "Source was not found", f"Source path does not exist:\n{source_path}")
        return False
    return True
