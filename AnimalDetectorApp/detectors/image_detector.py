from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.app_paths import project_path, relative_to_project


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".avif",
}


class ImageDetectionError(RuntimeError):
    """Raised when image detection cannot start or finish correctly."""


@dataclass(frozen=True)
class ImageDetectionOptions:
    confidence: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "auto"
    save_txt: bool = True
    save_conf: bool = True
    show_labels: bool = True
    show_conf: bool = True


@dataclass
class ImageFileResult:
    image_name: str
    verbose_summary: str
    box_descriptions: list[str]
    source_path: Path
    annotated_path: Path
    label_path: Path | None = None


@dataclass
class ImageDetectionRun:
    run_dir: Path
    model_path: Path
    source_path: Path
    total_files: int
    processed_files: int = 0
    total_detections: int = 0
    classes_detected: dict[str, int] = field(default_factory=dict)
    annotated_files: list[Path] = field(default_factory=list)
    label_files: list[Path] = field(default_factory=list)
    image_results: list[ImageFileResult] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.txt"


ProgressCallback = Callable[[int, int, str], None]


def find_image_files(source_path: str | Path) -> list[Path]:
    source = project_path(source_path)
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def run_image_detection(
    model_path: str | Path,
    source_path: str | Path,
    result_dir: str | Path,
    options: ImageDetectionOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    **parameters: Any,
) -> ImageDetectionRun:
    """Run YOLO detection for one image or all supported images in a directory."""
    options = options or ImageDetectionOptions(**parameters)
    model_file = _validate_model(model_path)
    source = _validate_source(source_path)
    image_files = find_image_files(source)
    if not image_files:
        raise ImageDetectionError(f"No supported image files found in: {source}")

    device = _resolve_device(options.device)
    run_dir = _create_run_dir(result_dir, model_file)
    _create_result_subdirs(run_dir)

    detection_run = ImageDetectionRun(
        run_dir=run_dir,
        model_path=model_file,
        source_path=source,
        total_files=len(image_files),
    )

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImageDetectionError(f"Required image detection dependency is missing: {exc}") from exc

    model = YOLO(str(model_file))

    for index, image_file in enumerate(image_files, start=1):
        if progress_callback:
            progress_callback(index - 1, len(image_files), f"Processing {image_file.name}")

        source_copy = run_dir / "source" / image_file.name
        shutil.copy2(image_file, source_copy)

        predict_kwargs: dict[str, Any] = {
            "source": str(image_file),
            "conf": options.confidence,
            "iou": options.iou,
            "imgsz": options.imgsz,
            "save": False,
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device

        results = model.predict(**predict_kwargs)
        if not results:
            continue

        result = results[0]
        annotated = result.plot(labels=options.show_labels, conf=options.show_conf)
        annotated_path = run_dir / "annotated" / image_file.name
        cv2.imwrite(str(annotated_path), annotated)
        detection_run.annotated_files.append(annotated_path)

        label_path = run_dir / "labels" / f"{image_file.stem}.txt"
        if options.save_txt:
            _write_label_file(label_path, result, options.save_conf)
            detection_run.label_files.append(label_path)
        else:
            label_path = None

        _update_summary(detection_run, result)
        detection_run.image_results.append(
            ImageFileResult(
                image_name=image_file.name,
                verbose_summary=result.verbose().strip(),
                box_descriptions=_box_descriptions(result),
                source_path=source_copy,
                annotated_path=annotated_path,
                label_path=label_path,
            )
        )
        detection_run.processed_files += 1

        if progress_callback:
            progress_callback(index, len(image_files), f"Finished {image_file.name}")

    _write_report(detection_run, options)
    _write_manifest(detection_run, options, device)
    return detection_run


def _validate_model(model_path: str | Path) -> Path:
    model_file = project_path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if model_file.suffix.lower() != ".pt":
        raise ImageDetectionError(f"Unsupported model format: {model_file.suffix}")
    return model_file


def _validate_source(source_path: str | Path) -> Path:
    source = project_path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Image source does not exist: {source}")
    if source.is_file() and source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ImageDetectionError(f"Unsupported image format: {source.suffix}")
    return source


def _resolve_device(device: str) -> str | int | None:
    normalized = (device or "auto").strip().lower()
    if normalized == "auto":
        return None
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise ImageDetectionError("CUDA was selected, but PyTorch is not installed.") from exc
        if not torch.cuda.is_available():
            raise ImageDetectionError("CUDA was selected, but CUDA is not available.")
        return 0 if normalized == "cuda" else normalized
    raise ImageDetectionError(f"Unsupported device value: {device}")


def _create_run_dir(result_dir: str | Path, model_path: Path) -> Path:
    root = project_path(result_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{model_path.stem}"
    run_dir = root / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _create_result_subdirs(run_dir: Path) -> None:
    for name in ("source", "annotated", "labels"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def _write_label_file(label_path: Path, result: Any, save_conf: bool) -> None:
    lines: list[str] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xywhn = boxes.xywhn.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        for coords, class_id, confidence in zip(xywhn, classes, confidences, strict=False):
            row = [str(int(class_id)), *(f"{value:.6f}" for value in coords)]
            if save_conf:
                row.append(f"{float(confidence):.6f}")
            lines.append(" ".join(row))

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _update_summary(detection_run: ImageDetectionRun, result: Any) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return

    names = getattr(result, "names", {})
    classes = boxes.cls.cpu().tolist()
    detection_run.total_detections += len(classes)
    for class_id in classes:
        class_name = names.get(int(class_id), str(int(class_id)))
        detection_run.classes_detected[class_name] = detection_run.classes_detected.get(class_name, 0) + 1


def _box_descriptions(result: Any) -> list[str]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    descriptions: list[str] = []
    names = getattr(result, "names", {})
    xyxy = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    for coords, class_id, confidence in zip(xyxy, classes, confidences, strict=False):
        class_id_int = int(class_id)
        class_name = names.get(class_id_int, str(class_id_int))
        coords_text = ", ".join(f"{coord:.2f}" for coord in coords)
        descriptions.append(
            f"Class: {class_name} (ID: {class_id_int}), "
            f"Confidence: {float(confidence):.4f}, Box: [{coords_text}]"
        )
    return descriptions


def _write_report(detection_run: ImageDetectionRun, options: ImageDetectionOptions) -> None:
    lines = [
        "AnimalDetector Photo Detection Report",
        f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {relative_to_project(detection_run.model_path)}",
        f"Source: {relative_to_project(detection_run.source_path)}",
        f"Result folder: {relative_to_project(detection_run.run_dir)}",
        "",
        "Parameters:",
        f"- confidence: {options.confidence}",
        f"- iou: {options.iou}",
        f"- imgsz: {options.imgsz}",
        f"- device: {options.device}",
        f"- save_txt: {options.save_txt}",
        f"- save_conf: {options.save_conf}",
        "",
        "Summary:",
        f"- total files: {detection_run.total_files}",
        f"- processed files: {detection_run.processed_files}",
        f"- total detections: {detection_run.total_detections}",
        "- classes detected:",
    ]
    if detection_run.classes_detected:
        lines.extend(f"  - {name}: {count}" for name, count in sorted(detection_run.classes_detected.items()))
    else:
        lines.append("  - none")

    lines.extend(["", "Images:"])
    if detection_run.image_results:
        for index, image_result in enumerate(detection_run.image_results, start=1):
            lines.extend(
                [
                    f"{index}. {image_result.image_name}",
                    f"   Summary: {image_result.verbose_summary or 'No detections'}",
                    "   Boxes:",
                ]
            )
            if image_result.box_descriptions:
                lines.extend(f"   - {description}" for description in image_result.box_descriptions)
            else:
                lines.append("   - none")
            lines.extend(
                [
                    f"   Source image: {relative_to_project(image_result.source_path)}",
                    f"   Annotated image: {relative_to_project(image_result.annotated_path)}",
                ]
            )
            if image_result.label_path is not None:
                lines.append(f"   Labels: {relative_to_project(image_result.label_path)}")
            lines.append("")
    else:
        lines.append("- none")

    detection_run.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    detection_run: ImageDetectionRun,
    options: ImageDetectionOptions,
    resolved_device: str | int | None,
) -> None:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest = {
        "run_type": "photo",
        "created_at": created_at,
        "model_path": relative_to_project(detection_run.model_path),
        "model_name": detection_run.model_path.name,
        "source_path": relative_to_project(detection_run.source_path),
        "result_path": relative_to_project(detection_run.run_dir),
        "parameters": {
            **asdict(options),
            "device": str(resolved_device if resolved_device is not None else options.device),
        },
        "summary": {
            "total_files": detection_run.total_files,
            "processed_files": detection_run.processed_files,
            "total_detections": detection_run.total_detections,
            "classes_detected": detection_run.classes_detected,
        },
        "images": [_manifest_image_result(image_result) for image_result in detection_run.image_results],
    }
    with detection_run.manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _manifest_image_result(image_result: ImageFileResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_name": image_result.image_name,
        "summary": image_result.verbose_summary,
        "boxes": image_result.box_descriptions,
        "source_image_path": relative_to_project(image_result.source_path),
        "annotated_image_path": relative_to_project(image_result.annotated_path),
    }
    if image_result.label_path is not None:
        payload["labels_path"] = relative_to_project(image_result.label_path)
    return payload
