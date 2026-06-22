from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_paths import project_path, relative_to_project
from core.config_manager import ConfigManager
from core.model_manager import ModelManager
from detectors.image_detector import ImageDetectionOptions, ImageDetectionRun, ImageFileResult, run_image_detection
from ui.error_utils import exception_message, show_error, validate_existing_source, validate_model_selection


class ImageDetectionWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        model_path: str,
        source_path: str,
        result_dir: str,
        options: ImageDetectionOptions,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.source_path = source_path
        self.result_dir = result_dir
        self.options = options

    @Slot()
    def run(self) -> None:
        try:
            result = run_image_detection(
                model_path=self.model_path,
                source_path=self.source_path,
                result_dir=self.result_dir,
                options=self.options,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(exception_message(exc))
            return
        self.finished.emit(result)


class PhotoDetectionPage(QWidget):
    detection_finished = Signal()

    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager) -> None:
        super().__init__()
        self.setObjectName("ContentPage")
        self.config_manager = config_manager
        self.model_manager = model_manager
        self._thread: QThread | None = None
        self._worker: ImageDetectionWorker | None = None
        self._preview_files: list[Path] = []
        self._preview_details: dict[Path, ImageFileResult] = {}
        self._preview_index = -1
        self._original_pixmap = QPixmap()

        title = QLabel("Photo Detection")
        title.setObjectName("PageTitle")

        self.model_label = QLabel()
        self.model_label.setObjectName("MutedText")
        self._refresh_model_label()

        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Select one image or a directory with images")
        self.source_input.setText(str(self.config_manager.get("photo_input_dir", "test_samples/photo")))

        browse_file = QPushButton("Image")
        browse_file.setObjectName("SecondaryButton")
        browse_file.clicked.connect(self._choose_image)

        browse_dir = QPushButton("Folder")
        browse_dir.setObjectName("SecondaryButton")
        browse_dir.clicked.connect(self._choose_directory)

        self.start_button = QPushButton("Start Photo Detection")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_detection)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.preview = QLabel("Preview will appear after detection")
        self.preview.setObjectName("PreviewArea")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setWordWrap(True)

        self.result_list = QListWidget()
        self.result_list.setMinimumWidth(240)
        self.result_list.currentRowChanged.connect(self._on_result_selected)

        self.previous_button = QPushButton("Previous")
        self.previous_button.setObjectName("SecondaryButton")
        self.previous_button.clicked.connect(self._select_previous_image)
        self.previous_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("SecondaryButton")
        self.next_button.clicked.connect(self._select_next_image)
        self.next_button.setEnabled(False)

        self.image_counter = QLabel("0 / 0")
        self.image_counter.setObjectName("MutedText")
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.image_details = QTextEdit()
        self.image_details.setReadOnly(True)
        self.image_details.setMinimumHeight(150)
        self.image_details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_details.setPlaceholderText("Select an annotated image to view detection details.")

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        source_row.addWidget(self.source_input, 1)
        source_row.addWidget(browse_file)
        source_row.addWidget(browse_dir)

        navigation_row = QHBoxLayout()
        navigation_row.setSpacing(8)
        navigation_row.addWidget(self.previous_button)
        navigation_row.addWidget(self.image_counter, 1)
        navigation_row.addWidget(self.next_button)

        result_panel = QVBoxLayout()
        result_panel.setSpacing(8)
        result_label = QLabel("Annotated Images")
        result_label.setObjectName("MutedText")
        result_panel.addWidget(result_label)
        result_panel.addWidget(self.result_list, 1)
        result_panel.addLayout(navigation_row)
        details_label = QLabel("Selected Image Details")
        details_label.setObjectName("MutedText")
        result_panel.addWidget(details_label)
        result_panel.addWidget(self.image_details, 1)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(14)
        preview_row.addWidget(self.preview, 2)
        preview_row.addLayout(result_panel, 1)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(36, 34, 36, 34)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(self.model_label)
        layout.addLayout(source_row)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress)
        layout.addLayout(preview_row, 1)
        layout.addWidget(self.log)

    def refresh_settings(self) -> None:
        self._refresh_model_label()

    def _choose_image(self) -> None:
        start_dir = str(project_path(self.source_input.text() or self.config_manager.get("photo_input_dir", "")))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            start_dir,
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.avif)",
        )
        if path:
            self.source_input.setText(relative_to_project(path))

    def _choose_directory(self) -> None:
        start_dir = str(project_path(self.source_input.text() or self.config_manager.get("photo_input_dir", "")))
        path = QFileDialog.getExistingDirectory(self, "Select image folder", start_dir)
        if path:
            self.source_input.setText(relative_to_project(path))

    def _start_detection(self) -> None:
        settings = self.config_manager.load()
        self.config_manager.settings = settings
        self._refresh_model_label()

        model_path = str(settings.get("default_model_path", "")).strip()
        source_path = self.source_input.text().strip()
        result_dir = str(settings.get("photo_results_dir", "detect_results/photo")).strip()

        if not validate_model_selection(self, self.model_manager, model_path):
            return
        if not validate_existing_source(self, source_path, "one image or an image directory"):
            return

        options = ImageDetectionOptions(
            confidence=float(settings.get("confidence", 0.25)),
            iou=float(settings.get("iou", 0.7)),
            imgsz=int(settings.get("imgsz", 640)),
            device=str(settings.get("device", "auto")),
            save_txt=bool(settings.get("save_txt", True)),
            save_conf=bool(settings.get("save_conf", True)),
            show_labels=bool(settings.get("show_labels", True)),
            show_conf=bool(settings.get("show_conf", True)),
        )

        self.log.clear()
        self._clear_preview()
        self._append_log("Starting photo detection...")
        self._append_log(f"Model: {model_path}")
        self._append_log(f"Source: {source_path}")
        self.progress.setValue(0)
        self.start_button.setEnabled(False)

        self._thread = QThread(self)
        self._worker = ImageDetectionWorker(model_path, source_path, result_dir, options)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setValue(int(current / total * 100))
        self._append_log(message)

    def _on_finished(self, result: Any) -> None:
        detection_run = result if isinstance(result, ImageDetectionRun) else None
        if detection_run is None:
            self._append_log("Detection finished.")
            self.start_button.setEnabled(True)
            return

        self.progress.setValue(100)
        self._append_log("")
        self._append_log("Detection finished successfully.")
        self._append_log(f"Processed files: {detection_run.processed_files}/{detection_run.total_files}")
        self._append_log(f"Total detections: {detection_run.total_detections}")
        self._append_log(f"Results: {relative_to_project(detection_run.run_dir)}")
        self._show_preview(detection_run.annotated_files, detection_run.image_results)
        self.start_button.setEnabled(True)
        self.detection_finished.emit()

    def _on_failed(self, message: str) -> None:
        self._append_log("")
        self._append_log(f"Error: {message}")
        self.progress.setValue(0)
        self.start_button.setEnabled(True)
        show_error(self, "Photo detection failed", message)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _show_preview(self, files: list[Path], details: list[ImageFileResult] | None = None) -> None:
        self._preview_files = list(files)
        self._preview_details = {
            image_result.annotated_path: image_result
            for image_result in details or []
        }
        self._preview_index = -1
        self._original_pixmap = QPixmap()
        self.result_list.clear()

        if not files:
            self.preview.setText("No annotated image was created.")
            self.preview.setPixmap(QPixmap())
            self.image_details.clear()
            self._update_preview_controls()
            return

        for index, file_path in enumerate(files, start=1):
            item = QListWidgetItem(f"{index}. {file_path.name}")
            item.setToolTip(str(file_path))
            self.result_list.addItem(item)

        self.result_list.setCurrentRow(0)
        self._update_preview_controls()

    def _on_result_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._preview_files):
            return
        self._preview_index = row
        self._load_current_preview()
        self._show_current_detection_details()

    def _select_previous_image(self) -> None:
        if self._preview_index > 0:
            self.result_list.setCurrentRow(self._preview_index - 1)

    def _select_next_image(self) -> None:
        if self._preview_index < len(self._preview_files) - 1:
            self.result_list.setCurrentRow(self._preview_index + 1)

    def _load_current_preview(self) -> None:
        if self._preview_index < 0 or self._preview_index >= len(self._preview_files):
            return

        file_path = self._preview_files[self._preview_index]
        pixmap = QPixmap(str(file_path))
        self._original_pixmap = pixmap
        if pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"Annotated image saved:\n{relative_to_project(file_path)}")
        else:
            self.preview.setText("")
            self.preview.setToolTip(str(file_path))
            self._render_current_preview()
        self._update_preview_controls()

    def _show_current_detection_details(self) -> None:
        if self._preview_index < 0 or self._preview_index >= len(self._preview_files):
            return

        file_path = self._preview_files[self._preview_index]
        details = self._preview_details.get(file_path)
        if details is None:
            self.image_details.setPlainText(
                "\n".join(
                    [
                        f"Image: {file_path.name}",
                        "Detection details are not available for this image.",
                        f"Annotated image: {relative_to_project(file_path)}",
                    ]
                )
            )
            return

        lines = [
            f"Image: {details.image_name}",
            "",
            f"Summary: {details.verbose_summary or 'No detections'}",
            "",
            "Boxes:",
        ]
        if details.box_descriptions:
            lines.extend(f"- {description}" for description in details.box_descriptions)
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                f"Source image: {relative_to_project(details.source_path)}",
                f"Annotated image: {relative_to_project(details.annotated_path)}",
            ]
        )
        if details.label_path is not None:
            lines.append(f"Labels: {relative_to_project(details.label_path)}")

        self.image_details.setPlainText("\n".join(lines))

    def _render_current_preview(self) -> None:
        if self._original_pixmap.isNull():
            return

        target_size = self.preview.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self._original_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)

    def _update_preview_controls(self) -> None:
        total = len(self._preview_files)
        current = self._preview_index + 1 if self._preview_index >= 0 else 0
        self.image_counter.setText(f"{current} / {total}")
        self.previous_button.setEnabled(total > 1 and self._preview_index > 0)
        self.next_button.setEnabled(total > 1 and self._preview_index < total - 1)

    def _clear_preview(self) -> None:
        self._preview_files = []
        self._preview_details = {}
        self._preview_index = -1
        self._original_pixmap = QPixmap()
        self.result_list.clear()
        self.image_details.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Preview will appear after detection")
        self._update_preview_controls()

    def _refresh_model_label(self) -> None:
        model_path = str(self.config_manager.get("default_model_path", "")).strip()
        status = "ready" if model_path and self.model_manager.model_exists(model_path) else "missing"
        self.model_label.setText(f"Default model: {model_path or 'not selected'} ({status})")

    def _append_log(self, message: str) -> None:
        self.log.append(message)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render_current_preview()
