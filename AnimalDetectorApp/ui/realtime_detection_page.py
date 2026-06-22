from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_paths import relative_to_project
from core.config_manager import ConfigManager
from core.model_manager import ModelManager
from detectors.stream_detector import (
    StreamDetectionError,
    StreamDetectionOptions,
    StreamDetectionSession,
    find_camera_sources,
)
from ui.error_utils import exception_message, show_error, show_warning, validate_model_selection


class RealtimeDetectionPage(QWidget):
    detection_finished = Signal()

    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager) -> None:
        super().__init__()
        self.setObjectName("ContentPage")
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.session: StreamDetectionSession | None = None
        self._last_pixmap = QPixmap()
        self._sources_loaded = False

        title = QLabel("Real-Time Detection")
        title.setObjectName("PageTitle")

        self.model_label = QLabel()
        self.model_label.setObjectName("MutedText")
        self._refresh_model_label()

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(260)
        self.source_combo.addItem("Press Refresh to scan cameras", None)

        self.start_button = QPushButton("Start Real-Time Detection")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_detection)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("SecondaryButton")
        self.refresh_button.clicked.connect(self._refresh_sources)

        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.setObjectName("DangerButton")
        self.stop_button.clicked.connect(self._stop_detection)
        self.stop_button.setEnabled(False)

        self.preview = QLabel("Stream preview will appear after detection starts")
        self.preview.setObjectName("PreviewArea")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setWordWrap(True)

        self.current_details = QTextEdit()
        self.current_details.setReadOnly(True)
        self.current_details.setMinimumHeight(160)
        self.current_details.setPlaceholderText("Current frame detections will appear here.")

        self.window_summary = QTextEdit()
        self.window_summary.setReadOnly(True)
        self.window_summary.setMinimumHeight(160)
        self.window_summary.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.window_summary.setPlaceholderText("Last 5 minutes summary will appear here.")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.timer = QTimer(self)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self._process_frame)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        source_row.addWidget(self.source_combo, 1)
        source_row.addWidget(self.refresh_button)
        source_row.addWidget(self.start_button)
        source_row.addWidget(self.stop_button)

        side_panel = QVBoxLayout()
        side_panel.setSpacing(8)
        current_label = QLabel("Current Frame")
        current_label.setObjectName("MutedText")
        summary_label = QLabel("Last 5 Minutes Summary")
        summary_label.setObjectName("MutedText")
        side_panel.addWidget(current_label)
        side_panel.addWidget(self.current_details, 1)
        side_panel.addWidget(summary_label)
        side_panel.addWidget(self.window_summary, 1)

        stream_row = QHBoxLayout()
        stream_row.setSpacing(14)
        stream_row.addWidget(self.preview, 2)
        stream_row.addLayout(side_panel, 1)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(36, 34, 36, 34)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(self.model_label)
        layout.addLayout(source_row)
        layout.addLayout(stream_row, 1)
        layout.addWidget(self.log)

        self.start_button.setEnabled(False)

    def refresh_settings(self) -> None:
        self._refresh_model_label()

    def ensure_sources_loaded(self) -> None:
        if not self._sources_loaded:
            self._refresh_sources()

    def _start_detection(self) -> None:
        settings = self.config_manager.load()
        self.config_manager.settings = settings
        self._refresh_model_label()

        model_path = str(settings.get("default_model_path", "")).strip()
        if not validate_model_selection(self, self.model_manager, model_path):
            return

        source = self.source_combo.currentData()
        if source is None:
            show_warning(self, "Source is not selected", "Select an available camera source.")
            return

        options = StreamDetectionOptions(
            confidence=float(settings.get("confidence", 0.25)),
            iou=float(settings.get("iou", 0.7)),
            imgsz=int(settings.get("imgsz", 640)),
            device=str(settings.get("device", "auto")),
            show_labels=bool(settings.get("show_labels", True)),
            show_conf=bool(settings.get("show_conf", True)),
        )

        self.log.clear()
        self.current_details.clear()
        self.window_summary.clear()
        self._last_pixmap = QPixmap()
        self._append_log("Starting real-time detection...")
        self._append_log(f"Model: {model_path}")
        self._append_log(f"Source: {self.source_combo.currentText()}")

        try:
            self.session = StreamDetectionSession(
                model_path=model_path,
                source=source,
                result_dir=str(settings.get("stream_results_dir", "detect_results/stream")),
                options=options,
            )
            self.session.start()
        except Exception as exc:
            message = exception_message(exc)
            self.session = None
            self._append_log(f"Error: {message}")
            show_error(self, "Real-time detection failed", message)
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.timer.start()
        self._append_log("Real-time detection started.")

    def _process_frame(self) -> None:
        if self.session is None:
            return

        try:
            frame_result = self.session.process_next_frame()
        except StreamDetectionError as exc:
            self._append_log(f"Stream stopped: {exc}")
            self._stop_detection(show_message=False)
            return
        except Exception as exc:
            message = exception_message(exc)
            self._append_log(f"Error: {message}")
            self._stop_detection(show_message=False)
            show_error(self, "Real-time detection failed", message)
            return

        self._show_frame(frame_result.annotated_frame)
        self._set_text_preserving_scroll(self.current_details, self._format_current_frame(frame_result))
        self._set_text_preserving_scroll(self.window_summary, self._format_window_summary(frame_result.window_summary))

    def _stop_detection(self, show_message: bool = True) -> None:
        self.timer.stop()
        run = None
        if self.session is not None:
            try:
                run = self.session.stop()
            except Exception as exc:
                message = exception_message(exc)
                self._append_log(f"Error while saving stream results: {message}")
                show_error(self, "Could not save stream results", message)
            finally:
                self.session = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)

        if run is not None:
            self._append_log("Real-time detection stopped.")
            self._append_log(f"Processed frames: {run.processed_frames}")
            self._append_log(f"Total detections: {run.total_detections}")
            self._append_log(f"Average processing FPS: {run.average_processing_fps:.2f}")
            self._append_log(f"Results: {relative_to_project(run.run_dir)}")
            self.detection_finished.emit()

        if show_message and run is not None:
            QMessageBox.information(self, "Real-time detection stopped", "Stream results were saved.")

    def _show_frame(self, frame: Any) -> None:
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_BGR888,
        ).copy()
        self._last_pixmap = QPixmap.fromImage(image)
        self._render_preview()

    def _render_preview(self) -> None:
        if self._last_pixmap.isNull():
            return
        target_size = self.preview.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.preview.setText("")
        self.preview.setPixmap(
            self._last_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _refresh_model_label(self) -> None:
        model_path = str(self.config_manager.get("default_model_path", "")).strip()
        status = "ready" if model_path and self.model_manager.model_exists(model_path) else "missing"
        self.model_label.setText(f"Default model: {model_path or 'not selected'} ({status})")

    def _append_log(self, message: str) -> None:
        self.log.append(message)

    @staticmethod
    def _set_text_preserving_scroll(widget: QTextEdit, text: str) -> None:
        scroll_bar = widget.verticalScrollBar()
        old_value = scroll_bar.value()
        was_at_bottom = old_value >= scroll_bar.maximum() - 2
        widget.setPlainText(text)
        if was_at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())
        else:
            scroll_bar.setValue(min(old_value, scroll_bar.maximum()))

    def _refresh_sources(self) -> None:
        self._sources_loaded = True
        previous_source = self.source_combo.currentData()
        self.source_combo.clear()
        for index in find_camera_sources():
            self.source_combo.addItem(f"Camera {index}", index)

        if self.source_combo.count() == 0:
            self.source_combo.addItem("No cameras found", None)
            self.start_button.setEnabled(False)
            return

        self.start_button.setEnabled(self.session is None)
        if previous_source is not None:
            for row in range(self.source_combo.count()):
                if self.source_combo.itemData(row) == previous_source:
                    self.source_combo.setCurrentIndex(row)
                    break

    @staticmethod
    def _format_current_frame(frame_result: Any) -> str:
        lines = [
            f"Frame: {frame_result.frame_index}",
            f"FPS: {frame_result.current_fps:.2f}",
            f"Objects in frame: {frame_result.current_detections}",
            "",
            "Classes in current frame:",
        ]
        if frame_result.current_class_counts:
            for class_name, count in sorted(frame_result.current_class_counts.items()):
                confidence = frame_result.current_confidences.get(class_name, 0.0)
                lines.append(f"- {class_name}: {count} (avg conf {confidence:.3f})")
        else:
            lines.append("- none")
        return "\n".join(lines)

    @staticmethod
    def _format_window_summary(summary: dict[str, Any]) -> str:
        lines = [
            f"Window seconds: {summary.get('window_seconds', 0)}",
            f"Max objects in frame: {summary.get('max_objects_in_frame', 0)}",
            "",
            "Classes detected:",
        ]
        classes = summary.get("classes_detected")
        if isinstance(classes, dict) and classes:
            lines.extend(f"- {class_name}: {count}" for class_name, count in sorted(classes.items()))
        else:
            lines.append("- none")

        lines.extend(["", "Average confidence by class:"])
        confidences = summary.get("average_confidence_by_class")
        if isinstance(confidences, dict) and confidences:
            lines.extend(f"- {class_name}: {value}" for class_name, value in sorted(confidences.items()))
        else:
            lines.append("- none")

        unstable = summary.get("unstable_detections")
        lines.extend(["", "Unstable / suspected detections:"])
        if isinstance(unstable, list) and unstable:
            for item in unstable:
                if not isinstance(item, dict):
                    continue
                lines.append(RealtimeDetectionPage._format_track_summary_line(item))
        else:
            lines.append("- none")
        return "\n".join(lines)

    @staticmethod
    def _format_track_summary_line(item: dict[str, Any]) -> str:
        reasons = ", ".join(str(reason) for reason in item.get("reasons", [])) or "none"
        class_counts = RealtimeDetectionPage._format_class_counts(item.get("class_counts"))
        return (
            "- "
            f"track_id={item.get('track_id')}, "
            f"status={item.get('status')}, "
            f"dominant_class={item.get('dominant_class')}, "
            f"tracked_frames={item.get('frames')}, "
            f"dominant_class_detections={item.get('dominant_class_detections', 'unknown')}, "
            f"avg_conf={item.get('average_confidence')}, "
            f"reasons={reasons}, "
            f"class_counts={class_counts}"
        )

    @staticmethod
    def _format_class_counts(value: Any) -> str:
        if not isinstance(value, dict) or not value:
            return "none"
        return ", ".join(f"{class_name}:{count}" for class_name, count in sorted(value.items()))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render_preview()
