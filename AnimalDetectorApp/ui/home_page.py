from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class HomePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ContentPage")

        title = QLabel("Welcome to AnimalDetector")
        title.setObjectName("PageTitle")

        subtitle = QLabel("Computer vision workspace for YOLO animal detection.")
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        description = QLabel(
            "AnimalDetector is a desktop system for running YOLO-based animal detection, reviewing "
            "annotated media, and keeping detection evidence organized. The application combines "
            "photo analysis, video processing, real-time camera detection, model management, and "
            "saved result inspection in one local workspace."
        )
        description.setObjectName("HomeLead")
        description.setWordWrap(True)

        content = QWidget()
        content.setObjectName("ContentPage")
        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_layout.setContentsMargins(36, 34, 36, 34)
        content_layout.setSpacing(16)
        content_layout.addWidget(title)
        content_layout.addWidget(subtitle)
        content_layout.addSpacing(8)
        content_layout.addWidget(description)
        content_layout.addSpacing(18)
        content_layout.addWidget(
            self._section(
                "What the system does",
                "AnimalDetector is designed for practical animal monitoring workflows: it accepts "
                "photos, video files, folders of images or videos and live camera sources; applies "
                "the selected YOLO model; draws bounding boxes and class labels; calculates summary "
                "statistics; and saves the generated outputs for later inspection. The system is "
                "intended to make detection runs repeatable, easy to review, and easy to compare "
                "across different input sources or model settings.",
            )
        )
        content_layout.addWidget(
            self._section(
                "Detection modes",
                "Photo Detection: analyze a single image or a folder of images, view annotated photos, "
                "switch between results, and inspect per-image summaries and bounding-box details.\n\n"
                "Video Detection: process one video or a folder of videos, generate annotated video files, "
                "play the processed media inside the application, and review per-video statistics, detected "
                "classes, tracking details, and unstable or suspected detections.\n\n"
                "Real-Time Detection: select an available camera source, refresh the device list, run live "
                "detection, preview the annotated stream, and monitor current-frame statistics together with "
                "a rolling summary for the last five minutes.",
            )
        )
        content_layout.addWidget(
            self._section(
                "Results and reports",
                "The Detection Results window collects completed runs from photo, video, and stream "
                "detection modes. It can show saved annotated images, play annotated videos, display "
                "run parameters, summarize detected classes, and open detailed information from each "
                "run manifest. Generated result folders can include annotated media, snapshots, labels, "
                "manifest.json, report.txt, and rolling stream summaries.",
            )
        )
        content_layout.addWidget(
            self._section(
                "Model and configuration tools",
                "Settings allow you to choose the default model type and model file, configure input and "
                "output folders, tune confidence, IoU, image size, and device selection, and control whether "
                "labels, confidences, TXT files, and on-image annotations are shown or saved. The application "
                "can also download or delete base YOLO models and add or remove custom YOLO weights stored in "
                "the project workspace.",
            )
        )
        content_layout.addWidget(
            self._section(
                "Typical workflow",
                "1. Select or add the YOLO model in Settings.\n"
                "2. Choose input folders and result folders for photos, videos, or streams.\n"
                "3. Run detection in the required mode.\n"
                "4. Review annotated outputs and detailed summaries.\n"
                "5. Return to Detection Results later to inspect saved reports and manifests.",
            )
        )

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)

    @staticmethod
    def _section(title: str, body: str) -> QWidget:
        section = QWidget()
        section.setObjectName("HomeSection")

        title_label = QLabel(title)
        title_label.setObjectName("HomeSectionTitle")

        body_label = QLabel(body)
        body_label.setObjectName("HomeBody")
        body_label.setWordWrap(True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(title_label)
        layout.addWidget(body_label)
        return section
