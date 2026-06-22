from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_paths import project_path
from core.result_manager import ResultManager, ResultRun
from ui.video_result_viewer import VideoResultViewer


class ResultsPage(QWidget):
    def __init__(self, result_manager: ResultManager) -> None:
        super().__init__()
        self.setObjectName("ContentPage")
        self.result_manager = result_manager
        self.runs: list[ResultRun] = []
        self._current_images: list[dict[str, Any]] = []
        self._current_image_paths: list[Path] = []
        self._original_pixmap = QPixmap()

        title = QLabel("Detection Results")
        title.setObjectName("PageTitle")

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("SecondaryButton")
        self.refresh_button.clicked.connect(self.refresh)

        self.run_list = QListWidget()
        self.run_list.currentRowChanged.connect(self._show_run)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("No detection results found yet.")
        self.details.setFixedHeight(210)
        self.details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.preview = QLabel("Select a photo result")
        self.preview.setObjectName("PreviewArea")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setWordWrap(True)

        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(220)
        self.image_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_list.currentRowChanged.connect(self._show_image)

        self.previous_image_button = QPushButton("Previous")
        self.previous_image_button.setObjectName("SecondaryButton")
        self.previous_image_button.clicked.connect(self._select_previous_image)
        self.previous_image_button.setEnabled(False)

        self.next_image_button = QPushButton("Next")
        self.next_image_button.setObjectName("SecondaryButton")
        self.next_image_button.clicked.connect(self._select_next_image)
        self.next_image_button.setEnabled(False)

        self.image_counter = QLabel("0 / 0")
        self.image_counter.setObjectName("MutedText")
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_details = QTextEdit()
        self.image_details.setReadOnly(True)
        self.image_details.setMinimumHeight(150)
        self.image_details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_details.setPlaceholderText("Select an image to view detection details.")
        self.video_viewer = VideoResultViewer()
        self.video_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.refresh_button)

        image_browser = QHBoxLayout()
        image_browser.setContentsMargins(0, 0, 0, 0)
        image_browser.setSpacing(14)
        image_browser.addWidget(self.preview, 2)

        image_panel = QVBoxLayout()
        image_panel.setSpacing(8)
        image_label = QLabel("Annotated Images")
        image_label.setObjectName("MutedText")
        image_panel.addWidget(image_label)
        image_panel.addWidget(self.image_list, 1)
        navigation_row = QHBoxLayout()
        navigation_row.setSpacing(8)
        navigation_row.addWidget(self.previous_image_button)
        navigation_row.addWidget(self.image_counter, 1)
        navigation_row.addWidget(self.next_image_button)
        image_panel.addLayout(navigation_row)
        image_details_label = QLabel("Selected Image Details")
        image_details_label.setObjectName("MutedText")
        image_panel.addWidget(image_details_label)
        image_panel.addWidget(self.image_details, 1)
        image_browser.addLayout(image_panel, 1)
        self.photo_browser = QWidget()
        self.photo_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.photo_browser.setLayout(image_browser)

        self.media_stack = QStackedWidget()
        self.media_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.media_stack.addWidget(self.photo_browser)
        self.media_stack.addWidget(self.video_viewer)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        right_panel.addWidget(self.details, 1)
        right_panel.addWidget(self.media_stack, 2)

        content = QHBoxLayout()
        content.setSpacing(18)
        content.addWidget(self.run_list, 3)
        content.addLayout(right_panel, 13)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 34, 36, 34)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addLayout(content, 1)

        self.refresh()

    def refresh(self) -> None:
        self.runs = self.result_manager.list_runs()
        self.run_list.clear()
        for run in self.runs:
            created = run.manifest.get("created_at", Path(run.path).name)
            item = QListWidgetItem(f"{run.display_type} - {created}")
            item.setToolTip(str(run.path))
            self.run_list.addItem(item)

        if self.runs:
            self.run_list.setCurrentRow(0)
        else:
            self.details.clear()
            self._clear_images()

    def _show_run(self, row: int) -> None:
        if row < 0 or row >= len(self.runs):
            return

        run = self.runs[row]
        manifest = run.manifest
        self.details.setPlainText(self._format_run_details(run, manifest))
        if run.run_type in {"video", "stream"}:
            self._clear_images()
            self._load_run_videos(manifest)
        else:
            self.video_viewer.clear()
            self._load_run_images(run, manifest)

    def _format_run_details(self, run: ResultRun, manifest: dict[str, Any]) -> str:
        if not manifest:
            return "\n".join(
                [
                    f"Type: {run.display_type}",
                    f"Result folder: {run.path.name}",
                    "",
                    "manifest.json was not found or could not be read.",
                ]
            )

        lines = [
            f"Type: {run.display_type}",
            f"Created at: {manifest.get('created_at', 'Unknown')}",
            f"Result folder: {run.path.name}",
            f"Model: {manifest.get('model_name', 'Unknown')}",
            f"Source: {self._display_path_name(manifest.get('source_path', 'Unknown'))}",
            "",
            "Parameters:",
        ]
        lines.extend(self._format_key_values(manifest.get("parameters")))
        lines.extend(["", "Summary:"])
        lines.extend(self._format_summary(manifest.get("summary")))
        if manifest.get("run_type") in {"video", "stream"}:
            lines.extend(["", "Videos:"])
            lines.extend(self._format_videos(manifest))
        else:
            lines.extend(["", "Images:"])
            lines.extend(self._format_images(manifest))
        return "\n".join(lines)

    def _load_run_images(self, run: ResultRun, manifest: dict[str, Any]) -> None:
        self.media_stack.setCurrentWidget(self.photo_browser)
        self._clear_images()
        if run.run_type != "photo" or not manifest:
            self.preview.setText("Image preview is available for photo results.")
            return

        images = manifest.get("images")
        if isinstance(images, list) and images:
            for image in images:
                if not isinstance(image, dict):
                    continue
                annotated_path = image.get("annotated_image_path")
                if not annotated_path:
                    continue
                resolved = project_path(annotated_path)
                self._current_images.append(image)
                self._current_image_paths.append(resolved)
                item = QListWidgetItem(image.get("image_name") or resolved.name)
                item.setToolTip(str(resolved))
                self.image_list.addItem(item)
        else:
            files = manifest.get("files", {})
            annotated_files = files.get("annotated") if isinstance(files, dict) else None
            if isinstance(annotated_files, list):
                for annotated_path in annotated_files:
                    resolved = project_path(annotated_path)
                    self._current_images.append(
                        {
                            "image_name": resolved.name,
                            "annotated_image_path": str(annotated_path),
                            "summary": "Details are not available in this older manifest.",
                            "boxes": [],
                        }
                    )
                    self._current_image_paths.append(resolved)
                    item = QListWidgetItem(resolved.name)
                    item.setToolTip(str(resolved))
                    self.image_list.addItem(item)

        if self._current_images:
            self.image_list.setCurrentRow(0)
        else:
            self.preview.setText("No annotated images found for this result.")

    def _load_run_videos(self, manifest: dict[str, Any]) -> None:
        self.media_stack.setCurrentWidget(self.video_viewer)
        videos = manifest.get("videos")
        if isinstance(videos, list):
            self.video_viewer.set_videos([video for video in videos if isinstance(video, dict)])
        else:
            self.video_viewer.clear()

    def _clear_images(self) -> None:
        self._current_images = []
        self._current_image_paths = []
        self._original_pixmap = QPixmap()
        self.image_list.clear()
        self.image_details.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Select a photo result")
        self._update_image_navigation(-1)

    def _show_image(self, row: int) -> None:
        if row < 0 or row >= len(self._current_images):
            self._update_image_navigation(row)
            return

        image = self._current_images[row]
        image_path = self._current_image_paths[row]
        self._original_pixmap = QPixmap(str(image_path))
        if self._original_pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"Annotated image not found:\n{image_path.name}")
        else:
            self.preview.setText("")
            self.preview.setToolTip(str(image_path))
            self._render_preview()

        self.image_details.setPlainText(self._format_selected_image(image, image_path))
        self._update_image_navigation(row)

    def _select_previous_image(self) -> None:
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def _select_next_image(self) -> None:
        row = self.image_list.currentRow()
        if row < len(self._current_images) - 1:
            self.image_list.setCurrentRow(row + 1)

    def _update_image_navigation(self, row: int) -> None:
        total = len(self._current_images)
        current = row + 1 if 0 <= row < total else 0
        self.image_counter.setText(f"{current} / {total}")
        self.previous_image_button.setEnabled(total > 1 and row > 0)
        self.next_image_button.setEnabled(total > 1 and 0 <= row < total - 1)

    def _render_preview(self) -> None:
        if self._original_pixmap.isNull():
            return
        target_size = self.preview.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.preview.setPixmap(
            self._original_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _format_selected_image(self, image: dict[str, Any], image_path: Path) -> str:
        lines = [
            f"Image: {image.get('image_name') or image_path.name}",
            "",
            f"Summary: {image.get('summary') or 'No detections'}",
            "",
            "Boxes:",
        ]
        boxes = image.get("boxes")
        if isinstance(boxes, list) and boxes:
            lines.extend(f"- {box}" for box in boxes)
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                f"Source image: {self._display_path_name(image.get('source_image_path', 'Unknown'))}",
                f"Annotated image: {self._display_path_name(image.get('annotated_image_path', image_path.name))}",
            ]
        )
        if image.get("labels_path"):
            lines.append(f"Labels: {self._display_path_name(image.get('labels_path'))}")
        return "\n".join(lines)

    def _format_key_values(self, value: Any, indent: str = "- ") -> list[str]:
        if not isinstance(value, dict) or not value:
            return [f"{indent}none"]

        lines: list[str] = []
        for key, item in value.items():
            label = str(key).replace("_", " ")
            if isinstance(item, dict):
                lines.append(f"{indent}{label}:")
                if item:
                    for nested_key, nested_value in item.items():
                        lines.append(f"  - {nested_key}: {nested_value}")
                else:
                    lines.append("  - none")
            else:
                lines.append(f"{indent}{label}: {item}")
        return lines

    def _format_images(self, manifest: dict[str, Any]) -> list[str]:
        images = manifest.get("images")
        if isinstance(images, list) and images:
            lines: list[str] = []
            for index, image in enumerate(images, start=1):
                if not isinstance(image, dict):
                    continue
                image_name = image.get("image_name") or self._display_path_name(
                    image.get("source_image_path", f"image_{index}")
                )
                lines.extend(
                    [
                        f"{index}. {image_name}",
                        f"   Summary: {image.get('summary') or 'No detections'}",
                        "   Boxes:",
                    ]
                )
                boxes = image.get("boxes")
                if isinstance(boxes, list) and boxes:
                    lines.extend(f"   - {box}" for box in boxes)
                else:
                    lines.append("   - none")

                lines.append(
                    f"   Source image: {self._display_path_name(image.get('source_image_path', 'Unknown'))}"
                )
                lines.append(
                    f"   Annotated image: {self._display_path_name(image.get('annotated_image_path', 'Unknown'))}"
                )
                if image.get("labels_path"):
                    lines.append(f"   Labels: {self._display_path_name(image.get('labels_path'))}")
                lines.append("")
            return lines or ["- none"]

        files = manifest.get("files")
        if isinstance(files, dict) and files:
            lines = ["Detailed image records are not available in this older manifest.", "Files:"]
            for group_name, group_files in files.items():
                lines.append(f"- {group_name}:")
                if isinstance(group_files, list) and group_files:
                    lines.extend(f"  - {self._display_path_name(path)}" for path in group_files)
                else:
                    lines.append("  - none")
            return lines

        return ["- none"]

    def _format_videos(self, manifest: dict[str, Any]) -> list[str]:
        videos = manifest.get("videos")
        if not isinstance(videos, list) or not videos:
            return ["- none"]

        lines: list[str] = []
        for index, video in enumerate(videos, start=1):
            if not isinstance(video, dict):
                continue
            statistics = video.get("statistics", {})
            lines.extend(
                [
                    f"{index}. {video.get('video_name') or self._display_path_name(video.get('source_video_path'))}",
                    f"   Source video: {self._display_path_name(video.get('source_video_path', 'Unknown'))}",
                    f"   Annotated video: {self._display_path_name(video.get('annotated_video_path', 'Unknown'))}",
                    f"   Processed frames: {statistics.get('processed_frames', 0) if isinstance(statistics, dict) else 0}",
                    f"   Total detections: {statistics.get('total_detections', 0) if isinstance(statistics, dict) else 0}",
                    f"   Average processing FPS: {statistics.get('average_processing_fps', 0) if isinstance(statistics, dict) else 0}",
                    f"   Max objects in frame: {statistics.get('max_objects_in_frame', 0) if isinstance(statistics, dict) else 0}",
                    f"   Tracking enabled: {statistics.get('tracking_enabled', False) if isinstance(statistics, dict) else False}",
                    f"   Unstable or suspected detections: {len(statistics.get('unstable_detections', [])) if isinstance(statistics, dict) and isinstance(statistics.get('unstable_detections'), list) else 0}",
                ]
            )
            if isinstance(statistics, dict):
                classes = statistics.get("class_object_detections")
                lines.append("   Classes detected:")
                if isinstance(classes, dict) and classes:
                    lines.extend(f"   - {class_name}: {count}" for class_name, count in classes.items())
                else:
                    lines.append("   - none")
                tracked_objects = statistics.get("unique_tracked_objects")
                lines.append("   Unique tracked objects:")
                if isinstance(tracked_objects, dict) and tracked_objects:
                    lines.extend(f"   - {class_name}: {count}" for class_name, count in tracked_objects.items())
                else:
                    lines.append("   - none")
                unstable = statistics.get("unstable_detections")
                lines.append("   Unstable and suspected detections:")
                if isinstance(unstable, list) and unstable:
                    for item in unstable:
                        if not isinstance(item, dict):
                            continue
                        lines.append(self._format_track_summary_line(item, "   - "))
                else:
                    lines.append("   - none")
            lines.append("")
        return lines or ["- none"]

    def _format_track_summary_line(self, item: dict[str, Any], prefix: str) -> str:
        reasons = ", ".join(str(reason) for reason in item.get("reasons", [])) or "none"
        class_counts = self._format_class_counts(item.get("class_counts"))
        return (
            prefix
            + f"track_id={item.get('track_id')}, "
            + f"status={item.get('status')}, "
            + f"dominant_class={item.get('dominant_class')}, "
            + f"tracked_frames={item.get('frames')}, "
            + f"dominant_class_detections={item.get('dominant_class_detections', 'unknown')}, "
            + f"avg_conf={item.get('average_confidence')}, "
            + f"reasons={reasons}, "
            + f"class_counts={class_counts}"
        )

    def _format_class_counts(self, value: Any) -> str:
        if not isinstance(value, dict) or not value:
            return "none"
        return ", ".join(f"{class_name}:{count}" for class_name, count in sorted(value.items()))

    def _format_summary(self, value: Any) -> list[str]:
        if not isinstance(value, dict) or not value:
            return ["- none", "", "Classes detected:", "  - none"]

        lines = [
            f"- total files: {value.get('total_files', 0)}",
            f"- processed files: {value.get('processed_files', 0)}",
            f"- total detections: {value.get('total_detections', 0)}",
            "",
            "Classes detected:",
        ]

        classes = value.get("classes_detected")
        if isinstance(classes, dict) and classes:
            lines.extend(f"  - {class_name}: {count}" for class_name, count in classes.items())
        else:
            lines.append("  - none")
        return lines

    @staticmethod
    def _display_path_name(value: Any) -> str:
        text = str(value or "Unknown")
        if text == "Unknown":
            return text
        return Path(text).name or text

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render_preview()
