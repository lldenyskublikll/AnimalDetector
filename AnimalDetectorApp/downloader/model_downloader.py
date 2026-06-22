from __future__ import annotations

from pathlib import Path

from core.app_paths import MODEL_FAMILIES, MODEL_SIZES, MODELS_ROOT


def expected_model_path(family: str, size: str) -> Path:
    family = family.lower()
    size = size.lower()
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported YOLO family: {family}")
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported YOLO size: {size}")
    return MODELS_ROOT / family / f"{family}{size}.pt"


def is_model_downloaded(family: str, size: str) -> bool:
    return expected_model_path(family, size).exists()


def download_base_model(family: str, size: str) -> Path:
    """Download one base model through Ultralytics into the project model folder."""
    output_path = expected_model_path(family, size)
    if output_path.exists():
        raise FileExistsError(f"Model already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    try:
        import os
        from ultralytics import YOLO

        os.chdir(output_path.parent)
        YOLO(output_path.name)
    finally:
        import os

        os.chdir(original_cwd)

    if not output_path.exists():
        raise RuntimeError(f"Ultralytics did not create model file: {output_path}")
    return output_path


def expand_model_selection(family: str, size: str) -> list[tuple[str, str]]:
    families = MODEL_FAMILIES if family == "all" else (family.lower(),)
    sizes = MODEL_SIZES if size == "all" else (size.lower(),)
    return [(selected_family, selected_size) for selected_family in families for selected_size in sizes]


def download_base_models(family: str, size: str) -> list[Path]:
    downloaded: list[Path] = []
    for selected_family, selected_size in expand_model_selection(family, size):
        downloaded.append(download_base_model(selected_family, selected_size))
    return downloaded
