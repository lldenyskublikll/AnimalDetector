from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.app_paths import DETECT_RESULTS_ROOT, project_path, relative_to_project


@dataclass(frozen=True)
class ResultRun:
    run_type: str
    path: Path
    manifest: dict[str, Any]

    @property
    def display_type(self) -> str:
        labels = {
            "photo": "Photo Detection",
            "video": "Video Detection",
            "stream": "Real-Time Detection",
        }
        return labels.get(self.run_type, self.run_type.title())


class ResultManager:
    """Read detection result folders created by later detection stages."""

    def __init__(self, results_root: Path = DETECT_RESULTS_ROOT) -> None:
        self.results_root = results_root

    def list_runs(self) -> list[ResultRun]:
        runs: list[ResultRun] = []
        for run_type in ("photo", "video", "stream"):
            root = self.results_root / run_type
            if not root.exists():
                continue
            for folder in sorted(root.iterdir(), reverse=True):
                if folder.is_dir():
                    runs.append(ResultRun(run_type, folder, self.read_manifest(folder)))
        return runs

    def read_manifest(self, run_folder: str | Path) -> dict[str, Any]:
        manifest_path = project_path(run_folder) / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def list_files(self, run_folder: str | Path) -> list[str]:
        root = project_path(run_folder)
        if not root.exists():
            return []
        return [
            relative_to_project(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

