from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.app_paths import SETTINGS_PATH, ensure_project_structure


DEFAULT_SETTINGS: dict[str, Any] = {
    "default_model_type": "custom",
    "default_model_path": "custom_weights/yolo/yolo26/yolo26m_(25_16)_(Animal_Detection_Dataset-1)_(1).pt",
    "photo_input_dir": "test_samples/photo",
    "video_input_dir": "test_samples/video",
    "photo_results_dir": "detect_results/photo",
    "video_results_dir": "detect_results/video",
    "stream_results_dir": "detect_results/stream",
    "confidence": 0.25,
    "iou": 0.7,
    "imgsz": 640,
    "device": "auto",
    "save_txt": True,
    "save_conf": True,
    "show_labels": True,
    "show_conf": True,
}


class ConfigManager:
    """Load, validate, update, and persist application settings."""

    def __init__(self, settings_path: Path = SETTINGS_PATH) -> None:
        self.settings_path = settings_path
        ensure_project_structure()
        self.settings: dict[str, Any] = self.load()

    def load(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            settings = deepcopy(DEFAULT_SETTINGS)
            self.save(settings)
            return settings

        try:
            with self.settings_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
        except (OSError, json.JSONDecodeError):
            loaded = {}

        settings = deepcopy(DEFAULT_SETTINGS)
        if isinstance(loaded, dict):
            settings.update(loaded)
        self.save(settings)
        return settings

    def save(self, settings: Mapping[str, Any] | None = None) -> None:
        if settings is not None:
            self.settings = dict(settings)

        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings_path.open("w", encoding="utf-8") as file:
            json.dump(self.settings, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def update(self, values: Mapping[str, Any]) -> None:
        self.settings.update(values)
        self.save()

    def reset(self) -> None:
        self.settings = deepcopy(DEFAULT_SETTINGS)
        self.save()

