from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from core.app_paths import (
    CUSTOM_WEIGHTS_ROOT,
    MODEL_FAMILIES,
    MODEL_SIZES,
    MODELS_ROOT,
    project_path,
    relative_to_project,
)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    family: str
    model_type: str
    size: str | None = None

    @property
    def relative_path(self) -> str:
        return relative_to_project(self.path)

    @property
    def display_name(self) -> str:
        label = "Base" if self.model_type == "base" else "Custom"
        return f"{label} / {self.family.upper()} / {self.name}"


class ModelManager:
    """Discover and manage base and custom YOLO weights."""

    def __init__(
        self,
        models_root: Path = MODELS_ROOT,
        custom_root: Path = CUSTOM_WEIGHTS_ROOT,
    ) -> None:
        self.models_root = models_root
        self.custom_root = custom_root
        self.ensure_directories()

    def ensure_directories(self) -> None:
        for root in (self.models_root, self.custom_root):
            for family in MODEL_FAMILIES:
                (root / family).mkdir(parents=True, exist_ok=True)

    def list_base_models(self) -> list[ModelInfo]:
        return self._list_models(self.models_root, "base")

    def list_custom_models(self) -> list[ModelInfo]:
        return self._list_models(self.custom_root, "custom")

    def list_models(self, model_type: str | None = None) -> list[ModelInfo]:
        if model_type == "base":
            return self.list_base_models()
        if model_type == "custom":
            return self.list_custom_models()
        return self.list_base_models() + self.list_custom_models()

    def model_exists(self, model_path: str | Path) -> bool:
        return project_path(model_path).is_file()

    def resolve_model(self, model_path: str | Path) -> Path:
        resolved = project_path(model_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Model file not found: {resolved}")
        return resolved

    def add_custom_model(self, source_path: str | Path, family: str) -> ModelInfo:
        family = family.lower()
        if family not in MODEL_FAMILIES:
            raise ValueError(f"Unsupported YOLO family: {family}")

        source = project_path(source_path)
        if source.suffix.lower() != ".pt":
            raise ValueError("Custom model must use the .pt format.")
        if not source.is_file():
            raise FileNotFoundError(f"Custom model file not found: {source}")

        destination = self.custom_root / family / source.name
        if destination.exists():
            raise FileExistsError(f"Custom model already exists: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return ModelInfo(
            name=destination.name,
            path=destination,
            family=family,
            model_type="custom",
            size=self._size_from_name(destination.name),
        )

    def expected_base_model_path(self, family: str, size: str) -> Path:
        family = family.lower()
        size = size.lower()
        if family not in MODEL_FAMILIES:
            raise ValueError(f"Unsupported YOLO family: {family}")
        if size not in MODEL_SIZES:
            raise ValueError(f"Unsupported YOLO size: {size}")
        return self.models_root / family / f"{family}{size}.pt"

    def base_model_exists(self, family: str, size: str) -> bool:
        return self.expected_base_model_path(family, size).is_file()

    def delete_base_model(self, family: str, size: str) -> Path:
        target = self.expected_base_model_path(family, size)
        if not target.exists():
            raise FileNotFoundError(f"Base model is already missing: {target}")
        target.unlink()
        return target

    def delete_custom_model(self, model_path: str | Path) -> Path:
        target = self.resolve_model(model_path)
        if not self._is_inside(target, self.custom_root):
            raise PermissionError("Only custom models can be deleted with this action.")
        target.unlink()
        return target

    def delete_model(self, model_path: str | Path, *, allow_base: bool = False) -> None:
        target = self.resolve_model(model_path)
        if not allow_base and self._is_inside(target, self.models_root):
            raise PermissionError("Base model deletion is disabled for this stage.")
        target.unlink()

    def _list_models(self, root: Path, model_type: str) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for family in MODEL_FAMILIES:
            family_dir = root / family
            if not family_dir.exists():
                continue
            for path in sorted(family_dir.glob("*.pt")):
                models.append(
                    ModelInfo(
                        name=path.name,
                        path=path,
                        family=family,
                        model_type=model_type,
                        size=self._size_from_name(path.name),
                    )
                )
        return models

    @staticmethod
    def _size_from_name(name: str) -> str | None:
        stem = Path(name).stem.lower()
        for family in MODEL_FAMILIES:
            prefix = family.lower()
            if stem.startswith(prefix) and len(stem) > len(prefix):
                size = stem[len(prefix)]
                if size in MODEL_SIZES:
                    return size
        return None

    @staticmethod
    def _is_inside(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False
