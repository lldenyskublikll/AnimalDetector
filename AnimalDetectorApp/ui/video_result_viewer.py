from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl, Qt
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_paths import project_path


class VideoResultViewer(QWidget):
    """Reusable annotated-video viewer for detection pages and result pages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._videos: list[dict[str, Any]] = []
        self._video_paths: list[Path] = []
        self._updating_position = False

        self.video_widget = QVideoWidget()
        self.video_widget.setObjectName("PreviewArea")
        self.video_widget.setMinimumHeight(280)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._sync_play_button)

        self.video_list = QListWidget()
        self.video_list.setMinimumWidth(240)
        self.video_list.currentRowChanged.connect(self._show_video)

        self.previous_button = QPushButton("Previous")
        self.previous_button.setObjectName("SecondaryButton")
        self.previous_button.clicked.connect(self._select_previous_video)
        self.previous_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("SecondaryButton")
        self.next_button.clicked.connect(self._select_next_video)
        self.next_button.setEnabled(False)

        self.counter = QLabel("0 / 0")
        self.counter.setObjectName("MutedText")
        self.counter.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("PrimaryButton")
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setEnabled(False)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("SecondaryButton")
        self.stop_button.clicked.connect(lambda: self.player.stop())
        self.stop_button.setEnabled(False)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(150)
        self.details.setPlaceholderText("Select an annotated video to view its summary.")

        list_label = QLabel("Annotated Videos")
        list_label.setObjectName("MutedText")
        details_label = QLabel("Selected Video Details")
        details_label.setObjectName("MutedText")

        navigation_row = QHBoxLayout()
        navigation_row.setSpacing(8)
        navigation_row.addWidget(self.previous_button)
        navigation_row.addWidget(self.counter, 1)
        navigation_row.addWidget(self.next_button)

        playback_row = QHBoxLayout()
        playback_row.setSpacing(8)
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.stop_button)
        playback_row.addWidget(self.position_slider, 1)

        side_panel = QVBoxLayout()
        side_panel.setSpacing(8)
        side_panel.addWidget(list_label)
        side_panel.addWidget(self.video_list, 1)
        side_panel.addLayout(navigation_row)
        side_panel.addWidget(details_label)
        side_panel.addWidget(self.details, 1)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        left_panel.addWidget(self.video_widget, 1)
        left_panel.addLayout(playback_row)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addLayout(left_panel, 2)
        layout.addLayout(side_panel, 1)

    def set_videos(self, videos: list[dict[str, Any]]) -> None:
        self.player.stop()
        self._videos = []
        self._video_paths = []
        self.video_list.clear()
        self.details.clear()
        self.position_slider.setRange(0, 0)

        for video in videos:
            path_value = video.get("annotated_video_path")
            if not path_value:
                continue
            path = project_path(path_value)
            self._videos.append(video)
            self._video_paths.append(path)
            item = QListWidgetItem(video.get("video_name") or path.name)
            item.setToolTip(str(path))
            self.video_list.addItem(item)

        if self._videos:
            self.video_list.setCurrentRow(0)
        else:
            self._update_navigation(-1)
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def clear(self) -> None:
        self.set_videos([])

    def _show_video(self, row: int) -> None:
        if row < 0 or row >= len(self._videos):
            self._update_navigation(row)
            return

        path = self._video_paths[row]
        self.player.stop()
        self.position_slider.setValue(0)
        self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.details.setPlainText(self._format_video_summary(self._videos[row], path))
        self.play_button.setEnabled(path.is_file())
        self.stop_button.setEnabled(path.is_file())
        self._update_navigation(row)

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _select_previous_video(self) -> None:
        row = self.video_list.currentRow()
        if row > 0:
            self.video_list.setCurrentRow(row - 1)

    def _select_next_video(self) -> None:
        row = self.video_list.currentRow()
        if row < len(self._videos) - 1:
            self.video_list.setCurrentRow(row + 1)

    def _update_navigation(self, row: int) -> None:
        total = len(self._videos)
        current = row + 1 if 0 <= row < total else 0
        self.counter.setText(f"{current} / {total}")
        self.previous_button.setEnabled(total > 1 and row > 0)
        self.next_button.setEnabled(total > 1 and 0 <= row < total - 1)

    def _on_position_changed(self, position: int) -> None:
        if self.position_slider.isSliderDown():
            return
        self._updating_position = True
        self.position_slider.setValue(position)
        self._updating_position = False

    def _on_duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, max(0, duration))

    def _sync_play_button(self, _state: QMediaPlayer.PlaybackState | None = None) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")

    def _format_video_summary(self, video: dict[str, Any], path: Path) -> str:
        statistics = video.get("statistics", {})
        lines = [
            f"Video: {video.get('video_name') or path.name}",
            "",
            f"Source video: {self._display_path_name(video.get('source_video_path', 'Unknown'))}",
            f"Annotated video: {self._display_path_name(video.get('annotated_video_path', path.name))}",
            "",
            "Summary:",
            f"- processed frames: {self._stat(statistics, 'processed_frames')}",
            f"- total detections: {self._stat(statistics, 'total_detections')}",
            f"- average processing FPS: {self._stat(statistics, 'average_processing_fps')}",
            f"- max objects in frame: {self._stat(statistics, 'max_objects_in_frame')}",
            f"- tracking enabled: {self._stat(statistics, 'tracking_enabled')}",
            "",
            "Classes detected:",
        ]

        classes = statistics.get("class_object_detections") if isinstance(statistics, dict) else None
        if isinstance(classes, dict) and classes:
            lines.extend(f"  - {class_name}: {count}" for class_name, count in classes.items())
        else:
            lines.append("  - none")

        tracked_objects = statistics.get("unique_tracked_objects") if isinstance(statistics, dict) else None
        lines.extend(["", "Unique tracked objects:"])
        if isinstance(tracked_objects, dict) and tracked_objects:
            lines.extend(f"  - {class_name}: {count}" for class_name, count in tracked_objects.items())
        else:
            lines.append("  - none")

        unstable = statistics.get("unstable_detections") if isinstance(statistics, dict) else None
        lines.extend(["", "Unstable / suspected detections:"])
        if isinstance(unstable, list) and unstable:
            for item in unstable:
                if not isinstance(item, dict):
                    continue
                lines.append(self._format_track_summary_line(item))
        else:
            lines.append("  - none")

        frames = video.get("frames")
        labels = video.get("labels")
        lines.extend(
            [
                "",
                f"Saved frames: {len(frames) if isinstance(frames, list) else 0}",
                f"Saved label files: {len(labels) if isinstance(labels, list) else 0}",
            ]
        )
        return "\n".join(lines)

    def _format_track_summary_line(self, item: dict[str, Any]) -> str:
        reasons = ", ".join(str(reason) for reason in item.get("reasons", [])) or "none"
        class_counts = self._format_class_counts(item.get("class_counts"))
        return (
            "  - "
            f"track_id={item.get('track_id')}, "
            f"status={item.get('status')}, "
            f"dominant_class={item.get('dominant_class')}, "
            f"tracked_frames={item.get('frames')}, "
            f"dominant_class_detections={item.get('dominant_class_detections', 'unknown')}, "
            f"avg_conf={item.get('average_confidence')}, "
            f"reasons={reasons}, "
            f"class_counts={class_counts}"
        )

    def _format_class_counts(self, value: Any) -> str:
        if not isinstance(value, dict) or not value:
            return "none"
        return ", ".join(f"{class_name}:{count}" for class_name, count in sorted(value.items()))

    @staticmethod
    def _stat(statistics: Any, key: str) -> Any:
        if isinstance(statistics, dict):
            return statistics.get(key, 0)
        return 0

    @staticmethod
    def _display_path_name(value: Any) -> str:
        text = str(value or "Unknown")
        if text == "Unknown":
            return text
        return Path(text).name or text
