from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.app_paths import project_path, relative_to_project


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
    ".webm",
}


class VideoDetectionError(RuntimeError):
    """Raised when video detection cannot start or finish correctly."""


@dataclass(frozen=True)
class VideoDetectionOptions:
    confidence: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "auto"
    save_txt: bool = True
    save_conf: bool = True
    show_labels: bool = True
    show_conf: bool = True
    save_frames: bool = True
    frame_save_stride: int = 30
    enable_tracking: bool = True
    tracker: str = "bytetrack.yaml"
    min_track_frames: int = 5
    stable_confidence: float = 0.35
    stable_class_ratio: float = 0.7
    max_class_switches: int = 1


@dataclass
class VideoFileResult:
    video_name: str
    source_path: Path
    annotated_video_path: Path
    frame_paths: list[Path] = field(default_factory=list)
    label_paths: list[Path] = field(default_factory=list)
    processed_frames: int = 0
    total_detections: int = 0
    average_processing_fps: float = 0.0
    max_objects_in_frame: int = 0
    class_frame_detections: dict[str, int] = field(default_factory=dict)
    class_object_detections: dict[str, int] = field(default_factory=dict)
    tracking_enabled: bool = False
    unique_tracked_objects: dict[str, int] = field(default_factory=dict)
    track_statistics: list[dict[str, Any]] = field(default_factory=list)
    unstable_detections: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VideoDetectionRun:
    run_dir: Path
    model_path: Path
    source_path: Path
    total_files: int
    processed_files: int = 0
    total_frames: int = 0
    total_detections: int = 0
    average_processing_fps: float = 0.0
    max_objects_in_frame: int = 0
    class_frame_detections: dict[str, int] = field(default_factory=dict)
    class_object_detections: dict[str, int] = field(default_factory=dict)
    unique_tracked_objects: dict[str, int] = field(default_factory=dict)
    unstable_detections: list[dict[str, Any]] = field(default_factory=list)
    video_results: list[VideoFileResult] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.txt"


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    frames: int = 0
    confidence_sum: float = 0.0
    class_counts: Counter[str] = field(default_factory=Counter)
    class_switches: int = 0
    last_class: str | None = None

    def update(self, frame_index: int, class_name: str, confidence: float) -> None:
        if self.last_class is not None and self.last_class != class_name:
            self.class_switches += 1
        self.last_class = class_name
        self.last_frame = frame_index
        self.frames += 1
        self.confidence_sum += confidence
        self.class_counts[class_name] += 1

    @property
    def average_confidence(self) -> float:
        if self.frames <= 0:
            return 0.0
        return self.confidence_sum / self.frames

    @property
    def dominant_class(self) -> str:
        if not self.class_counts:
            return "unknown"
        return self.class_counts.most_common(1)[0][0]

    @property
    def dominant_class_ratio(self) -> float:
        if self.frames <= 0 or not self.class_counts:
            return 0.0
        return self.class_counts[self.dominant_class] / self.frames


def find_video_files(source_path: str | Path) -> list[Path]:
    source = project_path(source_path)
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    return []


def run_video_detection(
    model_path: str | Path,
    source_path: str | Path,
    result_dir: str | Path,
    options: VideoDetectionOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    **parameters: Any,
) -> VideoDetectionRun:
    """Run YOLO detection for one video or all supported videos in a directory."""
    options = options or VideoDetectionOptions(**parameters)
    model_file = _validate_model(model_path)
    source = _validate_source(source_path)
    video_files = find_video_files(source)
    if not video_files:
        raise VideoDetectionError(f"No supported video files found in: {source}")

    device = _resolve_device(options.device)
    run_dir = _create_run_dir(result_dir, model_file)
    _create_result_subdirs(run_dir)

    detection_run = VideoDetectionRun(
        run_dir=run_dir,
        model_path=model_file,
        source_path=source,
        total_files=len(video_files),
    )

    try:
        from ultralytics import YOLO
        import cv2  # noqa: F401
    except ImportError as exc:
        raise VideoDetectionError(f"Required video detection dependency is missing: {exc}") from exc

    model = YOLO(str(model_file))

    total_elapsed = 0.0
    for file_index, video_file in enumerate(video_files, start=1):
        if progress_callback:
            progress_callback(file_index - 1, len(video_files), f"Processing {video_file.name}")

        video_result, elapsed = _process_video(model, video_file, run_dir, options, device)
        detection_run.video_results.append(video_result)
        detection_run.processed_files += 1
        detection_run.total_frames += video_result.processed_frames
        detection_run.total_detections += video_result.total_detections
        detection_run.max_objects_in_frame = max(
            detection_run.max_objects_in_frame,
            video_result.max_objects_in_frame,
        )
        _merge_counts(detection_run.class_frame_detections, video_result.class_frame_detections)
        _merge_counts(detection_run.class_object_detections, video_result.class_object_detections)
        _merge_counts(detection_run.unique_tracked_objects, video_result.unique_tracked_objects)
        detection_run.unstable_detections.extend(
            {
                **item,
                "video_name": video_result.video_name,
            }
            for item in video_result.unstable_detections
        )
        total_elapsed += elapsed

        if progress_callback:
            progress_callback(file_index, len(video_files), f"Finished {video_file.name}")

    if total_elapsed > 0:
        detection_run.average_processing_fps = detection_run.total_frames / total_elapsed

    _write_report(detection_run, options)
    _write_manifest(detection_run, options, device)
    return detection_run


def _process_video(
    model: Any,
    video_path: Path,
    run_dir: Path,
    options: VideoDetectionOptions,
    device: str | int | None,
) -> tuple[VideoFileResult, float]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoDetectionError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise VideoDetectionError(f"Could not determine video dimensions: {video_path}")

    source_copy = run_dir / "source" / video_path.name
    shutil.copy2(video_path, source_copy)

    temp_output = run_dir / "annotated_video" / f"{video_path.stem}_detected_no_audio.mp4"
    final_output = run_dir / "annotated_video" / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise VideoDetectionError(f"Could not create output video: {temp_output}")

    video_result = VideoFileResult(
        video_name=video_path.name,
        source_path=source_copy,
        annotated_video_path=final_output,
    )

    started_at = time.perf_counter()
    frame_index = 0
    track_states: dict[int, TrackState] = {}
    tracking_active = options.enable_tracking
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        inference_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": options.confidence,
            "iou": options.iou,
            "imgsz": options.imgsz,
            "save": False,
            "verbose": False,
        }
        if device is not None:
            inference_kwargs["device"] = device

        result, tracking_used = _run_frame_inference(
            model=model,
            inference_kwargs=inference_kwargs,
            tracking_active=tracking_active,
            tracker=options.tracker,
            persist=frame_index > 1,
        )
        if tracking_active and not tracking_used:
            tracking_active = False
        annotated_frame = result.plot(labels=options.show_labels, conf=options.show_conf)
        writer.write(annotated_frame)

        boxes = getattr(result, "boxes", None)
        objects_in_frame = len(boxes) if boxes is not None else 0
        video_result.processed_frames += 1
        video_result.total_detections += objects_in_frame
        video_result.max_objects_in_frame = max(video_result.max_objects_in_frame, objects_in_frame)
        _update_video_counts(video_result, result)
        if tracking_used:
            _update_track_states(track_states, result, frame_index)

        if options.save_frames and frame_index % max(1, options.frame_save_stride) == 0:
            frame_path = run_dir / "frames" / f"{video_path.stem}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            video_result.frame_paths.append(frame_path)

        if options.save_txt:
            label_path = run_dir / "labels" / f"{video_path.stem}_frame_{frame_index:06d}.txt"
            _write_label_file(label_path, result, options.save_conf)
            video_result.label_paths.append(label_path)

    capture.release()
    writer.release()
    elapsed = time.perf_counter() - started_at
    if elapsed > 0:
        video_result.average_processing_fps = video_result.processed_frames / elapsed

    video_result.tracking_enabled = bool(track_states)
    _finalize_track_statistics(video_result, track_states, options)
    _merge_original_audio(video_path, temp_output, final_output)
    return video_result, elapsed


def _run_frame_inference(
    model: Any,
    inference_kwargs: dict[str, Any],
    tracking_active: bool,
    tracker: str,
    persist: bool,
) -> tuple[Any, bool]:
    if tracking_active:
        try:
            results = model.track(
                **inference_kwargs,
                persist=persist,
                tracker=tracker,
            )
            return results[0], True
        except Exception:
            pass

    results = model.predict(**inference_kwargs)
    return results[0], False


def _validate_model(model_path: str | Path) -> Path:
    model_file = project_path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if model_file.suffix.lower() != ".pt":
        raise VideoDetectionError(f"Unsupported model format: {model_file.suffix}")
    return model_file


def _validate_source(source_path: str | Path) -> Path:
    source = project_path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Video source does not exist: {source}")
    if source.is_file() and source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise VideoDetectionError(f"Unsupported video format: {source.suffix}")
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
            raise VideoDetectionError("CUDA was selected, but PyTorch is not installed.") from exc
        if not torch.cuda.is_available():
            raise VideoDetectionError("CUDA was selected, but CUDA is not available.")
        return 0 if normalized == "cuda" else normalized
    raise VideoDetectionError(f"Unsupported device value: {device}")


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
    for name in ("source", "annotated_video", "frames", "labels"):
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


def _update_video_counts(video_result: VideoFileResult, result: Any) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return

    names = getattr(result, "names", {})
    seen_in_frame: set[str] = set()
    for class_id in boxes.cls.cpu().tolist():
        class_name = names.get(int(class_id), str(int(class_id)))
        video_result.class_object_detections[class_name] = (
            video_result.class_object_detections.get(class_name, 0) + 1
        )
        seen_in_frame.add(class_name)

    for class_name in seen_in_frame:
        video_result.class_frame_detections[class_name] = (
            video_result.class_frame_detections.get(class_name, 0) + 1
        )


def _update_track_states(track_states: dict[int, TrackState], result: Any, frame_index: int) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return

    track_ids = getattr(boxes, "id", None)
    if track_ids is None:
        return

    names = getattr(result, "names", {})
    ids = track_ids.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    for track_id, class_id, confidence in zip(ids, classes, confidences, strict=False):
        if track_id is None:
            continue
        numeric_track_id = int(track_id)
        class_name = names.get(int(class_id), str(int(class_id)))
        state = track_states.setdefault(
            numeric_track_id,
            TrackState(
                track_id=numeric_track_id,
                first_frame=frame_index,
                last_frame=frame_index,
            ),
        )
        state.update(frame_index, class_name, float(confidence))


def _finalize_track_statistics(
    video_result: VideoFileResult,
    track_states: dict[int, TrackState],
    options: VideoDetectionOptions,
) -> None:
    for state in sorted(track_states.values(), key=lambda item: item.track_id):
        status, reasons = _track_status(state, options)
        class_counts = dict(state.class_counts)
        dominant_class = state.dominant_class
        summary = {
            "track_id": state.track_id,
            "status": status,
            "reasons": reasons,
            "dominant_class": dominant_class,
            "dominant_class_detections": class_counts.get(dominant_class, 0),
            "frames": state.frames,
            "first_frame": state.first_frame,
            "last_frame": state.last_frame,
            "average_confidence": round(state.average_confidence, 4),
            "dominant_class_ratio": round(state.dominant_class_ratio, 4),
            "class_switches": state.class_switches,
            "class_counts": class_counts,
        }
        video_result.track_statistics.append(summary)

        if state.frames >= options.min_track_frames:
            video_result.unique_tracked_objects[state.dominant_class] = (
                video_result.unique_tracked_objects.get(state.dominant_class, 0) + 1
            )

        if status != "stable":
            video_result.unstable_detections.append(summary)


def _track_status(track_state: TrackState, options: VideoDetectionOptions) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if track_state.frames < options.min_track_frames:
        reasons.append("short_track")
    if track_state.average_confidence < options.stable_confidence:
        reasons.append("low_confidence")
    if track_state.dominant_class_ratio < options.stable_class_ratio:
        reasons.append("class_instability")
    if track_state.class_switches > options.max_class_switches:
        reasons.append("class_switches")

    if not reasons:
        return "stable", []
    if "short_track" in reasons and "low_confidence" in reasons:
        return "suspected_false_positive", reasons
    return "unstable_detection", reasons


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _format_class_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{class_name}:{count}" for class_name, count in sorted(value.items()))


def _format_track_summary_line(item: dict[str, Any], prefix: str = "   - ") -> str:
    reasons = ", ".join(str(reason) for reason in item.get("reasons", [])) or "none"
    return (
        prefix
        + f"track_id={item.get('track_id')}, "
        + f"status={item.get('status')}, "
        + f"dominant_class={item.get('dominant_class')}, "
        + f"tracked_frames={item.get('frames')}, "
        + f"dominant_class_detections={item.get('dominant_class_detections')}, "
        + f"avg_conf={item.get('average_confidence')}, "
        + f"reasons={reasons}, "
        + f"class_counts={_format_class_counts(item.get('class_counts'))}"
    )


def _has_audio(video_path: Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run(
        [ffmpeg, "-i", str(video_path), "-hide_banner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return "Audio:" in f"{completed.stdout}\n{completed.stderr}"


def _merge_original_audio(source_video: Path, silent_prediction: Path, final_video: Path) -> bool:
    if not _has_audio(source_video):
        silent_prediction.replace(final_video)
        return True

    try:
        import imageio_ffmpeg
    except ImportError:
        silent_prediction.replace(final_video)
        return False

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_prediction),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_video),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        silent_prediction.replace(final_video)
        return False

    silent_prediction.unlink(missing_ok=True)
    return True


def _write_report(detection_run: VideoDetectionRun, options: VideoDetectionOptions) -> None:
    lines = [
        "AnimalDetector Video Detection Report",
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
        f"- save_frames: {options.save_frames}",
        f"- frame_save_stride: {options.frame_save_stride}",
        f"- enable_tracking: {options.enable_tracking}",
        f"- tracker: {options.tracker}",
        f"- min_track_frames: {options.min_track_frames}",
        f"- stable_confidence: {options.stable_confidence}",
        f"- stable_class_ratio: {options.stable_class_ratio}",
        f"- max_class_switches: {options.max_class_switches}",
        "",
        "Summary:",
        f"- total files: {detection_run.total_files}",
        f"- processed files: {detection_run.processed_files}",
        f"- processed frames: {detection_run.total_frames}",
        f"- total detections: {detection_run.total_detections}",
        f"- average processing fps: {detection_run.average_processing_fps:.2f}",
        f"- max objects in frame: {detection_run.max_objects_in_frame}",
        f"- unstable or suspected detections: {len(detection_run.unstable_detections)}",
        "",
        "Classes detected:",
    ]
    if detection_run.class_object_detections:
        lines.extend(
            f"  - {class_name}: {count}"
            for class_name, count in sorted(detection_run.class_object_detections.items())
        )
    else:
        lines.append("  - none")

    lines.extend(["", "Unique tracked objects:"])
    if detection_run.unique_tracked_objects:
        lines.extend(
            f"  - {class_name}: {count}"
            for class_name, count in sorted(detection_run.unique_tracked_objects.items())
        )
    else:
        lines.append("  - none")

    lines.extend(["", "Videos:"])
    for index, video in enumerate(detection_run.video_results, start=1):
        lines.extend(
            [
                f"{index}. {video.video_name}",
                f"   Source video: {relative_to_project(video.source_path)}",
                f"   Annotated video: {relative_to_project(video.annotated_video_path)}",
                f"   Processed frames: {video.processed_frames}",
                f"   Total detections: {video.total_detections}",
                f"   Average processing fps: {video.average_processing_fps:.2f}",
                f"   Max objects in frame: {video.max_objects_in_frame}",
                f"   Tracking enabled: {video.tracking_enabled}",
                f"   Unstable or suspected detections: {len(video.unstable_detections)}",
                "   Classes detected:",
            ]
        )
        if video.class_object_detections:
            lines.extend(
                f"   - {class_name}: {count}"
                for class_name, count in sorted(video.class_object_detections.items())
            )
        else:
            lines.append("   - none")
        lines.append("   Unique tracked objects:")
        if video.unique_tracked_objects:
            lines.extend(
                f"   - {class_name}: {count}"
                for class_name, count in sorted(video.unique_tracked_objects.items())
            )
        else:
            lines.append("   - none")
        lines.append("   Unstable and suspected detections:")
        if video.unstable_detections:
            for item in video.unstable_detections:
                lines.append(_format_track_summary_line(item))
        else:
            lines.append("   - none")
        lines.append("")

    detection_run.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    detection_run: VideoDetectionRun,
    options: VideoDetectionOptions,
    resolved_device: str | int | None,
) -> None:
    manifest = {
        "run_type": "video",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "processed_frames": detection_run.total_frames,
            "total_detections": detection_run.total_detections,
            "classes_detected": detection_run.class_object_detections,
        },
        "video_statistics": {
            "processed_frames": detection_run.total_frames,
            "average_processing_fps": round(detection_run.average_processing_fps, 4),
            "max_objects_in_frame": detection_run.max_objects_in_frame,
            "class_frame_detections": detection_run.class_frame_detections,
            "class_object_detections": detection_run.class_object_detections,
            "unique_tracked_objects": detection_run.unique_tracked_objects,
            "unstable_detections": detection_run.unstable_detections,
        },
        "videos": [_manifest_video_result(video) for video in detection_run.video_results],
    }
    with detection_run.manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _manifest_video_result(video: VideoFileResult) -> dict[str, Any]:
    return {
        "video_name": video.video_name,
        "source_video_path": relative_to_project(video.source_path),
        "annotated_video_path": relative_to_project(video.annotated_video_path),
        "frames": [relative_to_project(path) for path in video.frame_paths],
        "labels": [relative_to_project(path) for path in video.label_paths],
        "statistics": {
            "processed_frames": video.processed_frames,
            "total_detections": video.total_detections,
            "average_processing_fps": round(video.average_processing_fps, 4),
            "max_objects_in_frame": video.max_objects_in_frame,
            "class_frame_detections": video.class_frame_detections,
            "class_object_detections": video.class_object_detections,
            "tracking_enabled": video.tracking_enabled,
            "unique_tracked_objects": video.unique_tracked_objects,
            "track_statistics": video.track_statistics,
            "unstable_detections": video.unstable_detections,
        },
    }
