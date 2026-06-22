from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.app_paths import MODEL_FAMILIES, MODEL_SIZES, project_path, relative_to_project
from core.config_manager import ConfigManager, DEFAULT_SETTINGS
from core.model_manager import ModelInfo, ModelManager
from downloader.model_downloader import download_base_model, expand_model_selection
from ui.error_utils import exception_message, show_error, show_warning


class DirectoryField(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.editor = QLineEdit()
        button = QPushButton("Browse")
        button.setObjectName("SecondaryButton")
        button.clicked.connect(self._browse_directory)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.editor, 1)
        layout.addWidget(button)

    def text(self) -> str:
        return self.editor.text()

    def setText(self, value: str) -> None:
        self.editor.setText(value)

    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select directory", self.editor.text())
        if selected:
            self.editor.setText(relative_to_project(selected))


class BaseModelDownloadWorker(QObject):
    finished = Signal(object)

    def __init__(self, selections: list[tuple[str, str]]) -> None:
        super().__init__()
        self.selections = selections

    @Slot()
    def run(self) -> None:
        result: dict[str, list[str]] = {
            "downloaded": [],
            "skipped": [],
            "failed": [],
        }
        for family, size in self.selections:
            model_name = f"{family}{size}.pt"
            try:
                path = download_base_model(family, size)
                result["downloaded"].append(relative_to_project(path))
            except FileExistsError:
                result["skipped"].append(model_name)
            except Exception as exc:
                result["failed"].append(f"{model_name}: {exc}")
        self.finished.emit(result)


class SettingsPage(QWidget):
    settings_saved = Signal()

    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager) -> None:
        super().__init__()
        self.setObjectName("ContentPage")
        self.config_manager = config_manager
        self.model_manager = model_manager
        self._model_items: list[ModelInfo] = []
        self._download_thread: QThread | None = None
        self._download_worker: BaseModelDownloadWorker | None = None

        self.model_type = QComboBox()
        self.model_type.addItem("Custom model", "custom")
        self.model_type.addItem("Base YOLO model", "base")
        self.model_type.currentIndexChanged.connect(self._reload_models)

        self.model_path = QComboBox()
        self.model_path.setMinimumWidth(360)

        self.base_family = QComboBox()
        self.base_family.addItem("YOLO11", "yolo11")
        self.base_family.addItem("YOLO26", "yolo26")
        self.base_family.addItem("Both families", "all")

        self.base_size = QComboBox()
        for size in MODEL_SIZES:
            self.base_size.addItem(size, size)
        self.base_size.addItem("All sizes", "all")

        self.download_base_button = QPushButton("Download")
        self.download_base_button.setObjectName("PrimaryButton")
        self.download_base_button.clicked.connect(self.download_base_models)

        self.delete_base_button = QPushButton("Delete")
        self.delete_base_button.setObjectName("DangerButton")
        self.delete_base_button.clicked.connect(self.delete_base_models)

        self.custom_family = QComboBox()
        for family in MODEL_FAMILIES:
            self.custom_family.addItem(family.upper(), family)

        self.add_custom_button = QPushButton("Add")
        self.add_custom_button.setObjectName("SecondaryButton")
        self.add_custom_button.clicked.connect(self.add_custom_model)

        self.custom_delete_model = QComboBox()
        self.custom_delete_model.setMinimumWidth(360)

        self.delete_custom_button = QPushButton("Delete")
        self.delete_custom_button.setObjectName("DangerButton")
        self.delete_custom_button.clicked.connect(self.delete_selected_custom_model)

        self.photo_input_dir = self._path_row("photo_input_dir")
        self.video_input_dir = self._path_row("video_input_dir")
        self.photo_results_dir = self._path_row("photo_results_dir")
        self.video_results_dir = self._path_row("video_results_dir")
        self.stream_results_dir = self._path_row("stream_results_dir")

        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 1.0)
        self.confidence.setSingleStep(0.01)
        self.confidence.setDecimals(2)

        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.01, 1.0)
        self.iou.setSingleStep(0.01)
        self.iou.setDecimals(2)

        self.imgsz = QSpinBox()
        self.imgsz.setRange(128, 2048)
        self.imgsz.setSingleStep(32)

        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])

        self.save_txt = QCheckBox("Save TXT labels")
        self.save_conf = QCheckBox("Save confidence in TXT")
        self.show_labels = QCheckBox("Show labels")
        self.show_conf = QCheckBox("Show confidence")

        self.status = QLabel("")
        self.status.setObjectName("MutedText")

        save_button = QPushButton("Save Settings")
        save_button.setObjectName("PrimaryButton")
        save_button.clicked.connect(self.save_settings)

        reset_button = QPushButton("Reset Defaults")
        reset_button.setObjectName("DangerButton")
        reset_button.clicked.connect(self.reset_defaults)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")

        model_form = self._form_layout()
        model_form.addRow("Default model type", self.model_type)
        model_form.addRow("Default model", self.model_path)

        folders_form = self._form_layout()
        folders_form.addRow("Photo input", self.photo_input_dir)
        folders_form.addRow("Video input", self.video_input_dir)
        folders_form.addRow("Photo results", self.photo_results_dir)
        folders_form.addRow("Video results", self.video_results_dir)
        folders_form.addRow("Stream results", self.stream_results_dir)

        parameters_form = self._form_layout()
        parameters_form.addRow("Confidence", self.confidence)
        parameters_form.addRow("IoU", self.iou)
        parameters_form.addRow("Image size", self.imgsz)
        parameters_form.addRow("Device", self.device)
        parameters_form.addRow("", self.save_txt)
        parameters_form.addRow("", self.save_conf)
        parameters_form.addRow("", self.show_labels)
        parameters_form.addRow("", self.show_conf)

        base_form = self._form_layout()
        base_form.addRow("Model family", self.base_family)
        base_form.addRow("Model size", self.base_size)
        base_form.addRow("", self._button_row(self.download_base_button, self.delete_base_button))

        custom_form = self._form_layout()
        custom_form.addRow("Model family", self._field_button_row(self.custom_family, self.add_custom_button))
        custom_form.addRow("Delete model", self._field_button_row(self.custom_delete_model, self.delete_custom_button))

        actions = QHBoxLayout()
        actions.addWidget(save_button)
        actions.addWidget(reset_button)
        actions.addStretch(1)

        content = QWidget()
        content.setObjectName("ContentPage")
        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_layout.setContentsMargins(36, 34, 36, 34)
        content_layout.setSpacing(18)
        content_layout.addWidget(title)
        content_layout.addWidget(self._group_box("Choose default model:", model_form))
        content_layout.addWidget(self._group_box("Folders setup:", folders_form))
        content_layout.addWidget(self._group_box("Model parameters setup:", parameters_form))
        content_layout.addWidget(self._group_box("Download/delete base YOLO models:", base_form))
        content_layout.addWidget(self._group_box("Download/delete custom YOLO model:", custom_form))
        content_layout.addLayout(actions)
        content_layout.addWidget(self.status)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)

        self.load_settings()

    def load_settings(self) -> None:
        settings = self.config_manager.settings
        model_type = settings.get("default_model_type", DEFAULT_SETTINGS["default_model_type"])
        self.model_type.setCurrentIndex(max(0, self.model_type.findData(model_type)))
        self._reload_models()
        self._set_selected_model(settings.get("default_model_path", ""))

        for key in (
            "photo_input_dir",
            "video_input_dir",
            "photo_results_dir",
            "video_results_dir",
            "stream_results_dir",
        ):
            editor = getattr(self, key)
            editor.setText(str(settings.get(key, DEFAULT_SETTINGS[key])))

        self.confidence.setValue(float(settings.get("confidence", DEFAULT_SETTINGS["confidence"])))
        self.iou.setValue(float(settings.get("iou", DEFAULT_SETTINGS["iou"])))
        self.imgsz.setValue(int(settings.get("imgsz", DEFAULT_SETTINGS["imgsz"])))
        self.device.setCurrentIndex(max(0, self.device.findText(str(settings.get("device", "auto")))))
        self.save_txt.setChecked(bool(settings.get("save_txt", True)))
        self.save_conf.setChecked(bool(settings.get("save_conf", True)))
        self.show_labels.setChecked(bool(settings.get("show_labels", True)))
        self.show_conf.setChecked(bool(settings.get("show_conf", True)))

    def save_settings(self) -> None:
        selected_model = self.model_path.currentData()
        settings: dict[str, Any] = {
            "default_model_type": self.model_type.currentData(),
            "default_model_path": selected_model or self.model_path.currentText(),
            "photo_input_dir": self.photo_input_dir.text().strip(),
            "video_input_dir": self.video_input_dir.text().strip(),
            "photo_results_dir": self.photo_results_dir.text().strip(),
            "video_results_dir": self.video_results_dir.text().strip(),
            "stream_results_dir": self.stream_results_dir.text().strip(),
            "confidence": self.confidence.value(),
            "iou": self.iou.value(),
            "imgsz": self.imgsz.value(),
            "device": self.device.currentText(),
            "save_txt": self.save_txt.isChecked(),
            "save_conf": self.save_conf.isChecked(),
            "show_labels": self.show_labels.isChecked(),
            "show_conf": self.show_conf.isChecked(),
        }
        try:
            self.config_manager.update(settings)
        except Exception as exc:
            message = exception_message(exc)
            self.status.setText("Could not save settings.")
            show_error(self, "Could not save settings", message)
            return
        self.status.setText("Settings saved.")
        QMessageBox.information(self, "Settings saved", "Settings have been saved.")
        self.settings_saved.emit()

    def reset_defaults(self) -> None:
        try:
            self.config_manager.reset()
        except Exception as exc:
            message = exception_message(exc)
            self.status.setText("Could not reset settings.")
            show_error(self, "Could not reset settings", message)
            return
        self.load_settings()
        self.status.setText("Default settings restored.")
        QMessageBox.information(self, "Defaults restored", "Settings have been reset to defaults.")
        self.settings_saved.emit()

    def download_base_models(self) -> None:
        family = str(self.base_family.currentData())
        size = str(self.base_size.currentData())
        selections = expand_model_selection(family, size)
        if not selections:
            return

        self.download_base_button.setEnabled(False)
        self.delete_base_button.setEnabled(False)
        self.status.setText("Downloading base model weights...")

        self._download_thread = QThread(self)
        self._download_worker = BaseModelDownloadWorker(selections)
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.finished.connect(self._on_base_download_finished)
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_thread.finished.connect(self._cleanup_download_worker)
        self._download_thread.start()

    def delete_base_models(self) -> None:
        family = str(self.base_family.currentData())
        size = str(self.base_size.currentData())
        selections = expand_model_selection(family, size)
        if not selections:
            return

        answer = QMessageBox.question(
            self,
            "Delete base models",
            "Delete selected base model files from the project?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        deleted: list[str] = []
        missing: list[str] = []
        failed: list[str] = []
        for selected_family, selected_size in selections:
            model_name = f"{selected_family}{selected_size}.pt"
            try:
                path = self.model_manager.delete_base_model(selected_family, selected_size)
                deleted.append(relative_to_project(path))
            except FileNotFoundError:
                missing.append(model_name)
            except Exception as exc:
                failed.append(f"{model_name}: {exc}")

        self._reload_models()
        self._show_operation_summary(
            title="Base model deletion",
            completed_label="Deleted",
            completed=deleted,
            skipped_label="Already missing",
            skipped=missing,
            failed=failed,
        )
        self.settings_saved.emit()

    def add_custom_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select custom YOLO model",
            str(Path.home()),
            "PyTorch weights (*.pt)",
        )
        if not path:
            return

        try:
            model = self.model_manager.add_custom_model(path, str(self.custom_family.currentData()))
        except Exception as exc:
            show_error(self, "Could not add custom model", exception_message(exc))
            return

        self.model_type.setCurrentIndex(max(0, self.model_type.findData("custom")))
        self._reload_models()
        self._set_selected_model(model.relative_path)
        self.status.setText(f"Custom model added: {model.relative_path}")
        self.settings_saved.emit()

    def delete_selected_custom_model(self) -> None:
        selected_model = self.custom_delete_model.currentData()
        if not selected_model:
            show_warning(self, "No custom model selected", "Select a custom model first.")
            return

        answer = QMessageBox.question(
            self,
            "Delete custom model",
            f"Delete selected custom model?\n{selected_model}",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            deleted = self.model_manager.delete_custom_model(selected_model)
        except Exception as exc:
            show_error(self, "Could not delete custom model", exception_message(exc))
            return

        self._reload_models()
        self.status.setText(f"Custom model deleted: {relative_to_project(deleted)}")
        self.settings_saved.emit()

    def _on_base_download_finished(self, result: object) -> None:
        summary = result if isinstance(result, dict) else {}
        downloaded = list(summary.get("downloaded", []))
        skipped = list(summary.get("skipped", []))
        failed = list(summary.get("failed", []))
        self._reload_models()
        self._show_operation_summary(
            title="Base model download",
            completed_label="Downloaded",
            completed=downloaded,
            skipped_label="Already exists",
            skipped=skipped,
            failed=failed,
        )
        self.download_base_button.setEnabled(True)
        self.delete_base_button.setEnabled(True)
        self.settings_saved.emit()

    def _cleanup_download_worker(self) -> None:
        if self._download_worker is not None:
            self._download_worker.deleteLater()
        if self._download_thread is not None:
            self._download_thread.deleteLater()
        self._download_worker = None
        self._download_thread = None

    def _reload_models(self) -> None:
        selected = self.model_path.currentData()
        model_type = self.model_type.currentData()
        self._model_items = self.model_manager.list_models(model_type)
        self.model_path.clear()

        if not self._model_items:
            self.model_path.addItem("No .pt models found", "")
            self._reload_custom_delete_models()
            return

        for model in self._model_items:
            self.model_path.addItem(model.display_name, model.relative_path)

        if selected:
            self._set_selected_model(selected)
        self._reload_custom_delete_models()

    def _reload_custom_delete_models(self) -> None:
        selected = self.custom_delete_model.currentData()
        self.custom_delete_model.clear()
        custom_models = self.model_manager.list_custom_models()
        if not custom_models:
            self.custom_delete_model.addItem("No custom models found", "")
            return
        for model in custom_models:
            self.custom_delete_model.addItem(model.display_name, model.relative_path)
        if selected:
            index = self.custom_delete_model.findData(selected)
            if index >= 0:
                self.custom_delete_model.setCurrentIndex(index)

    def _set_selected_model(self, model_path: str) -> None:
        if not model_path:
            return
        normalized = relative_to_project(project_path(model_path))
        index = self.model_path.findData(normalized)
        if index >= 0:
            self.model_path.setCurrentIndex(index)

    def _show_operation_summary(
        self,
        title: str,
        completed_label: str,
        completed: list[str],
        skipped_label: str,
        skipped: list[str],
        failed: list[str],
    ) -> None:
        lines = [title]
        lines.append(f"{completed_label}: {len(completed)}")
        lines.extend(f"- {item}" for item in completed[:10])
        if len(completed) > 10:
            lines.append(f"- ... and {len(completed) - 10} more")
        lines.append(f"{skipped_label}: {len(skipped)}")
        lines.extend(f"- {item}" for item in skipped[:10])
        if len(skipped) > 10:
            lines.append(f"- ... and {len(skipped) - 10} more")
        lines.append(f"Failed: {len(failed)}")
        lines.extend(f"- {item}" for item in failed[:10])
        if len(failed) > 10:
            lines.append(f"- ... and {len(failed) - 10} more")

        self.status.setText(
            f"{completed_label}: {len(completed)}; {skipped_label}: {len(skipped)}; Failed: {len(failed)}"
        )
        if failed:
            QMessageBox.warning(self, title, "\n".join(lines))
        else:
            QMessageBox.information(self, title, "\n".join(lines))

    @staticmethod
    def _button_row(*buttons: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for button in buttons:
            layout.addWidget(button)
        layout.addStretch(1)
        return row

    @staticmethod
    def _field_button_row(field: QWidget, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(field, 1)
        layout.addWidget(button)
        return row

    @staticmethod
    def _form_layout() -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)
        return form

    @staticmethod
    def _group_box(title: str, form: QFormLayout) -> QGroupBox:
        group = QGroupBox(title)
        group.setLayout(form)
        return group

    def _path_row(self, key: str) -> DirectoryField:
        row = DirectoryField(self)
        row.setObjectName(key)
        return row
