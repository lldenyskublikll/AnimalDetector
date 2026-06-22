from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.json"

MODELS_ROOT = ROOT_DIR / "models" / "yolo"
CUSTOM_WEIGHTS_ROOT = ROOT_DIR / "custom_weights" / "yolo"

TEST_SAMPLES_ROOT = ROOT_DIR / "test_samples"
DETECT_RESULTS_ROOT = ROOT_DIR / "detect_results"

MODEL_FAMILIES = ("yolo11", "yolo26")
MODEL_SIZES = ("n", "s", "m", "l", "x")


def project_path(value: str | Path) -> Path:
    """Return an absolute project path for a relative or absolute value."""
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def relative_to_project(path: str | Path) -> str:
    """Return a path relative to the project root when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def ensure_project_structure() -> None:
    """Create the folder structure required by the first project stage."""
    directories = [
        CONFIG_DIR,
        MODELS_ROOT / "yolo11",
        MODELS_ROOT / "yolo26",
        CUSTOM_WEIGHTS_ROOT / "yolo11",
        CUSTOM_WEIGHTS_ROOT / "yolo26",
        TEST_SAMPLES_ROOT / "photo",
        TEST_SAMPLES_ROOT / "video",
        DETECT_RESULTS_ROOT / "photo",
        DETECT_RESULTS_ROOT / "video",
        DETECT_RESULTS_ROOT / "stream",
        ROOT_DIR / "downloader",
        ROOT_DIR / "script_examples",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

