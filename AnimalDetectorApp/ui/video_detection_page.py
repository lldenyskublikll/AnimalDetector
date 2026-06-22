from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
from detectors.video_detector import VideoDetectionOptions, VideoDetectionRun, run_video_detection
from ui.error_utils import exception_message, show_error, validate_existing_source, validate_model_selection
from ui.video_result_viewer import VideoResultViewer


class VideoDetectionWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        model_path: str,
        source_path: str,
        result_dir: str,
        options: VideoDetectionOptions,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.source_path = source_path
        self.result_dir = result_dir
        self.options = options

    @Slot()
    def run(self) -> None:
        try:
            result = run_video_detection(
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


class VideoDetectionPage(QWidget):
    detection_finished = Signal()

    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager) -> None:
        super().__init__()
        self.setObjectName("ContentPage")
        self.config_manager = config_manager
        self.model_manager = model_manager
        self._thread: QThread | None = None
        self._worker: VideoDetectionWorker | None = None

        title = QLabel("Video Detection")
        title.setObjectName("PageTitle")

        self.model_label = QLabel()
        self.model_label.setObjectName("MutedText")
        self._refresh_model_label()

        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Select one video or a directory with videos")
        self.source_input.setText(str(self.config_manager.get("video_input_dir", "test_samples/video")))

        browse_file = QPushButton("Video")
        browse_file.setObjectName("SecondaryButton")
        browse_file.clicked.connect(self._choose_video)

        browse_dir = QPushButton("Folder")
        browse_dir.setObjectName("SecondaryButton")
        browse_dir.clicked.connect(self._choose_directory)

        self.start_button = QPushButton("Start Video Detection")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_detection)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.video_viewer = VideoResultViewer()
        self.video_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        source_row.addWidget(self.source_input, 1)
        source_row.addWidget(browse_file)
        source_row.addWidget(browse_dir)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(36, 34, 36, 34)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(self.model_label)
        layout.addLayout(source_row)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.video_viewer, 1)
        layout.addWidget(self.log)

    def refresh_settings(self) -> None:
        self._refresh_model_label()

    def _choose_video(self) -> None:
        start_dir = str(project_path(self.source_input.text() or self.config_manager.get("video_input_dir", "")))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            start_dir,
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm)",
        )
        if path:
            self.source_input.setText(relative_to_project(path))

    def _choose_directory(self) -> None:
        start_dir = str(project_path(self.source_input.text() or self.config_manager.get("video_input_dir", "")))
        path = QFileDialog.getExistingDirectory(self, "Select video folder", start_dir)
        if path:
            self.source_input.setText(relative_to_project(path))

    def _start_detection(self) -> None:
        settings = self.config_manager.load()
        self.config_manager.settings = settings
        self._refresh_model_label()

        model_path = str(settings.get("default_model_path", "")).strip()
        source_path = self.source_input.text().strip()
        result_dir = str(settings.get("video_results_dir", "detect_results/video")).strip()

        if not validate_model_selection(self, self.model_manager, model_path):
            return
        if not validate_existing_source(self, source_path, "one video or a video directory"):
            return

        options = VideoDetectionOptions(
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
        self.video_viewer.clear()
        self._append_log("Starting video detection...")
        self._append_log(f"Model: {model_path}")
        self._append_log(f"Source: {source_path}")
        self.progress.setValue(0)
        self.start_button.setEnabled(False)

        self._thread = QThread(self)
        self._worker = VideoDetectionWorker(model_path, source_path, result_dir, options)
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
        detection_run = result if isinstance(result, VideoDetectionRun) else None
        if detection_run is None:
            self._append_log("Video detection finished.")
            self.start_button.setEnabled(True)
            return

        self.progress.setValue(100)
        self._append_log("")
        self._append_log("Video detection finished successfully.")
        self._append_log(f"Processed files: {detection_run.processed_files}/{detection_run.total_files}")
        self._append_log(f"Processed frames: {detection_run.total_frames}")
        self._append_log(f"Total detections: {detection_run.total_detections}")
        self._append_log(f"Average processing FPS: {detection_run.average_processing_fps:.2f}")
        self._append_log(f"Unique tracked objects: {sum(detection_run.unique_tracked_objects.values())}")
        self._append_log(f"Unstable or suspected detections: {len(detection_run.unstable_detections)}")
        self._append_log(f"Results: {relative_to_project(detection_run.run_dir)}")

        self.video_viewer.set_videos([self._video_result_payload(video_result) for video_result in detection_run.video_results])

        self.start_button.setEnabled(True)
        self.detection_finished.emit()

    def _on_failed(self, message: str) -> None:
        self._append_log("")
        self._append_log(f"Error: {message}")
        self.progress.setValue(0)
        self.start_button.setEnabled(True)
        show_error(self, "Video detection failed", message)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _refresh_model_label(self) -> None:
        model_path = str(self.config_manager.get("default_model_path", "")).strip()
        status = "ready" if model_path and self.model_manager.model_exists(model_path) else "missing"
        self.model_label.setText(f"Default model: {model_path or 'not selected'} ({status})")

    def _append_log(self, message: str) -> None:
        self.log.append(message)

    @staticmethod
    def _video_result_payload(video_result: Any) -> dict[str, Any]:
        return {
            "video_name": video_result.video_name,
            "source_video_path": str(video_result.source_path),
            "annotated_video_path": str(video_result.annotated_video_path),
            "frames": [str(path) for path in video_result.frame_paths],
            "labels": [str(path) for path in video_result.label_paths],
            "statistics": {
                "processed_frames": video_result.processed_frames,
                "total_detections": video_result.total_detections,
                "average_processing_fps": round(video_result.average_processing_fps, 4),
                "max_objects_in_frame": video_result.max_objects_in_frame,
                "class_frame_detections": video_result.class_frame_detections,
                "class_object_detections": video_result.class_object_detections,
                "tracking_enabled": video_result.tracking_enabled,
                "unique_tracked_objects": video_result.unique_tracked_objects,
                "track_statistics": video_result.track_statistics,
                "unstable_detections": video_result.unstable_detections,
            },
        }
