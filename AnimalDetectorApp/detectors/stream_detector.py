from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.app_paths import project_path, relative_to_project


class StreamDetectionError(RuntimeError):
    """Raised when real-time detection cannot start or finish correctly."""


@dataclass(frozen=True)
class StreamDetectionOptions:
    confidence: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "auto"
    show_labels: bool = True
    show_conf: bool = True
    record_video: bool = True
    save_snapshots: bool = True
    snapshot_stride: int = 120
    summary_window_seconds: int = 300
    summary_save_interval_seconds: int = 30
    enable_tracking: bool = True
    tracker: str = "bytetrack.yaml"
    min_track_frames: int = 5
    stable_confidence: float = 0.35
    stable_class_ratio: float = 0.7
    max_class_switches: int = 1


@dataclass
class StreamFrameResult:
    annotated_frame: Any
    frame_index: int
    current_fps: float
    current_class_counts: dict[str, int]
    current_confidences: dict[str, float]
    current_detections: int
    window_summary: dict[str, Any]


@dataclass
class StreamRunResult:
    run_dir: Path
    model_path: Path
    source: int | str
    recorded_video_path: Path | None = None
    snapshot_paths: list[Path] = field(default_factory=list)
    processed_frames: int = 0
    total_detections: int = 0
    average_processing_fps: float = 0.0
    max_objects_in_frame: int = 0
    class_object_detections: dict[str, int] = field(default_factory=dict)
    class_frame_detections: dict[str, int] = field(default_factory=dict)
    unique_tracked_objects: dict[str, int] = field(default_factory=dict)
    track_statistics: list[dict[str, Any]] = field(default_factory=list)
    unstable_detections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.txt"


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
        return self.confidence_sum / self.frames if self.frames else 0.0

    @property
    def dominant_class(self) -> str:
        if not self.class_counts:
            return "unknown"
        return self.class_counts.most_common(1)[0][0]

    @property
    def dominant_class_ratio(self) -> float:
        if not self.frames or not self.class_counts:
            return 0.0
        return self.class_counts[self.dominant_class] / self.frames


class StreamDetectionSession:
    def __init__(
        self,
        model_path: str | Path,
        source: int | str,
        result_dir: str | Path,
        options: StreamDetectionOptions | None = None,
    ) -> None:
        self.model_path = _validate_model(model_path)
        self.source = source
        self.result_dir = result_dir
        self.options = options or StreamDetectionOptions()
        self.device = _resolve_device(self.options.device)
        self.run = StreamRunResult(
            run_dir=_create_run_dir(result_dir, self.model_path),
            model_path=self.model_path,
            source=source,
        )
        _create_result_subdirs(self.run.run_dir)

        self.capture: Any | None = None
        self.writer: Any | None = None
        self.model: Any | None = None
        self._started_at = 0.0
        self._last_frame_at = 0.0
        self._last_summary_save_at = 0.0
        self._window_events: deque[dict[str, Any]] = deque()
        self._window_track_events: deque[dict[str, Any]] = deque()
        self._track_states: dict[int, TrackState] = {}
        self._tracking_active = self.options.enable_tracking
        self._stopped = False

    def start(self) -> None:
        try:
            import cv2
            from ultralytics import YOLO
        except ImportError as exc:
            raise StreamDetectionError(f"Required stream detection dependency is missing: {exc}") from exc

        self.capture = _open_video_capture(cv2, self.source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise StreamDetectionError(f"Camera or stream source is not available: {self.source}")

        self.model = YOLO(str(self.model_path))
        self._started_at = time.perf_counter()
        self._last_frame_at = self._started_at
        self._last_summary_save_at = self._started_at

    def process_next_frame(self) -> StreamFrameResult:
        if self.capture is None or self.model is None:
            raise StreamDetectionError("Stream detection session is not started.")

        import cv2

        success, frame = self.capture.read()
        if not success:
            raise StreamDetectionError("Could not read a frame from the stream source.")

        frame_index = self.run.processed_frames + 1
        inference_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.options.confidence,
            "iou": self.options.iou,
            "imgsz": self.options.imgsz,
            "save": False,
            "verbose": False,
        }
        if self.device is not None:
            inference_kwargs["device"] = self.device

        result, tracking_used = self._run_inference(inference_kwargs, persist=frame_index > 1)
        if self._tracking_active and not tracking_used:
            self._tracking_active = False

        annotated_frame = result.plot(labels=self.options.show_labels, conf=self.options.show_conf)
        self._ensure_writer(annotated_frame)
        if self.writer is not None:
            self.writer.write(annotated_frame)

        now = time.perf_counter()
        current_fps = 1.0 / max(0.0001, now - self._last_frame_at)
        self._last_frame_at = now

        boxes = getattr(result, "boxes", None)
        current_detections = len(boxes) if boxes is not None else 0
        current_counts, current_confidences = _frame_class_stats(result)

        self.run.processed_frames += 1
        self.run.total_detections += current_detections
        self.run.max_objects_in_frame = max(self.run.max_objects_in_frame, current_detections)
        _merge_counts(self.run.class_object_detections, current_counts)
        for class_name in current_counts:
            self.run.class_frame_detections[class_name] = self.run.class_frame_detections.get(class_name, 0) + 1

        self._record_window_events(now, current_counts, current_confidences)
        if tracking_used:
            self._update_track_states(result, frame_index)
            self._record_window_track_events(now, result, frame_index)

        if self.options.save_snapshots and frame_index % max(1, self.options.snapshot_stride) == 0:
            snapshot_path = self.run.run_dir / "snapshots" / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)
            self.run.snapshot_paths.append(snapshot_path)

        window_summary = self.window_summary()
        if now - self._last_summary_save_at >= self.options.summary_save_interval_seconds:
            self._write_window_summary(window_summary)
            self._last_summary_save_at = now

        return StreamFrameResult(
            annotated_frame=annotated_frame,
            frame_index=frame_index,
            current_fps=current_fps,
            current_class_counts=current_counts,
            current_confidences=current_confidences,
            current_detections=current_detections,
            window_summary=window_summary,
        )

    def stop(self) -> StreamRunResult:
        if self._stopped:
            return self.run

        self._stopped = True
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        elapsed = time.perf_counter() - self._started_at if self._started_at else 0.0
        if elapsed > 0:
            self.run.average_processing_fps = self.run.processed_frames / elapsed

        self._finalize_track_statistics()
        final_summary = self.window_summary()
        self._write_window_summary(final_summary)
        _write_report(self.run, self.options, final_summary)
        _write_manifest(self.run, self.options, final_summary)
        return self.run

    def window_summary(self) -> dict[str, Any]:
        now = time.perf_counter()
        self._trim_window_events(now)
        class_counts: dict[str, int] = {}
        confidence_sums: dict[str, float] = {}
        max_objects = 0
        for event in self._window_events:
            class_name = str(event["class_name"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_sums[class_name] = confidence_sums.get(class_name, 0.0) + float(event["confidence"])
            max_objects = max(max_objects, int(event["objects_in_frame"]))

        average_confidences = {
            class_name: round(confidence_sums[class_name] / count, 4)
            for class_name, count in class_counts.items()
        }
        return {
            "window_seconds": self.options.summary_window_seconds,
            "classes_detected": class_counts,
            "average_confidence_by_class": average_confidences,
            "max_objects_in_frame": max_objects,
            "unstable_detections": self._window_unstable_detections(now),
        }

    def _run_inference(self, inference_kwargs: dict[str, Any], persist: bool) -> tuple[Any, bool]:
        assert self.model is not None
        if self._tracking_active:
            try:
                results = self.model.track(
                    **inference_kwargs,
                    persist=persist,
                    tracker=self.options.tracker,
                )
                return results[0], True
            except Exception:
                pass

        results = self.model.predict(**inference_kwargs)
        return results[0], False

    def _ensure_writer(self, annotated_frame: Any) -> None:
        if self.writer is not None or not self.options.record_video:
            return

        import cv2

        height, width = annotated_frame.shape[:2]
        fps = 25.0
        if self.capture is not None:
            fps = self.capture.get(cv2.CAP_PROP_FPS) or fps
        output_path = self.run.run_dir / "recorded_video" / "stream_detected.mp4"
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if self.writer.isOpened():
            self.run.recorded_video_path = output_path
        else:
            self.writer.release()
            self.writer = None

    def _record_window_events(
        self,
        now: float,
        current_counts: dict[str, int],
        current_confidences: dict[str, float],
    ) -> None:
        objects_in_frame = sum(current_counts.values())
        for class_name, count in current_counts.items():
            confidence = current_confidences.get(class_name, 0.0)
            for _ in range(count):
                self._window_events.append(
                    {
                        "timestamp": now,
                        "class_name": class_name,
                        "confidence": confidence,
                        "objects_in_frame": objects_in_frame,
                    }
                )
        self._trim_window_events(now)

    def _trim_window_events(self, now: float) -> None:
        cutoff = now - self.options.summary_window_seconds
        while self._window_events and self._window_events[0]["timestamp"] < cutoff:
            self._window_events.popleft()
        while self._window_track_events and self._window_track_events[0]["timestamp"] < cutoff:
            self._window_track_events.popleft()

    def _record_window_track_events(self, now: float, result: Any, frame_index: int) -> None:
        for observation in self._track_observations(result, frame_index):
            event = dict(observation)
            event["timestamp"] = now
            self._window_track_events.append(event)
        self._trim_window_events(now)

    def _update_track_states(self, result: Any, frame_index: int) -> None:
        for observation in self._track_observations(result, frame_index):
            numeric_track_id = int(observation["track_id"])
            state = self._track_states.setdefault(
                numeric_track_id,
                TrackState(numeric_track_id, frame_index, frame_index),
            )
            state.update(frame_index, str(observation["class_name"]), float(observation["confidence"]))

    def _track_observations(self, result: Any, frame_index: int) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        track_ids = getattr(boxes, "id", None)
        if track_ids is None:
            return []

        names = getattr(result, "names", {})
        ids = track_ids.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        observations: list[dict[str, Any]] = []
        for track_id, class_id, confidence in zip(ids, classes, confidences, strict=False):
            if track_id is None:
                continue
            class_name = names.get(int(class_id), str(int(class_id)))
            observations.append(
                {
                    "track_id": int(track_id),
                    "frame_index": frame_index,
                    "class_name": class_name,
                    "confidence": float(confidence),
                }
            )
        return observations

    def _finalize_track_statistics(self) -> None:
        self.run.track_statistics = []
        self.run.unique_tracked_objects = {}
        self.run.unstable_detections = self._current_unstable_detections()

        for state in sorted(self._track_states.values(), key=lambda item: item.track_id):
            summary = self._track_summary(state)
            self.run.track_statistics.append(summary)
            if state.frames >= self.options.min_track_frames:
                self.run.unique_tracked_objects[state.dominant_class] = (
                    self.run.unique_tracked_objects.get(state.dominant_class, 0) + 1
                )

    def _current_unstable_detections(self) -> list[dict[str, Any]]:
        return self._unstable_detections_from_states(self._track_states)

    def _window_unstable_detections(self, now: float) -> list[dict[str, Any]]:
        self._trim_window_events(now)
        states: dict[int, TrackState] = {}
        for event in self._window_track_events:
            track_id = int(event["track_id"])
            frame_index = int(event["frame_index"])
            state = states.setdefault(track_id, TrackState(track_id, frame_index, frame_index))
            state.update(frame_index, str(event["class_name"]), float(event["confidence"]))
        return self._unstable_detections_from_states(states)

    def _unstable_detections_from_states(self, states: dict[int, TrackState]) -> list[dict[str, Any]]:
        unstable: list[dict[str, Any]] = []
        for state in sorted(states.values(), key=lambda item: item.track_id):
            summary = self._track_summary(state)
            if summary["status"] != "stable":
                unstable.append(summary)
        return unstable

    def _track_summary(self, state: TrackState) -> dict[str, Any]:
        status, reasons = _track_status(state, self.options)
        class_counts = dict(state.class_counts)
        dominant_class = state.dominant_class
        return {
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

    def _write_window_summary(self, summary: dict[str, Any]) -> None:
        logs_dir = self.run.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
        }
        (logs_dir / "latest_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = _summary_text_lines("Last 5 Minutes Summary", summary)
        (logs_dir / "latest_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stream_detection(
    model_path: str | Path,
    source: int | str,
    result_dir: str | Path,
    **parameters: Any,
) -> Path:
    """Run a stream session until the source ends. GUI code should prefer StreamDetectionSession."""
    session = StreamDetectionSession(
        model_path=model_path,
        source=source,
        result_dir=result_dir,
        options=StreamDetectionOptions(**parameters),
    )
    session.start()
    try:
        while True:
            session.process_next_frame()
    except StreamDetectionError:
        pass
    return session.stop().run_dir


def find_camera_sources(max_index: int = 10) -> list[int]:
    """Return OpenCV camera indexes that can be opened on the current machine."""
    try:
        import cv2
    except ImportError:
        return []

    indexes: list[int] = []
    for index in range(max(0, max_index)):
        with _suppress_native_stderr():
            capture = _open_video_capture(cv2, index, allow_default_fallback=False)
        if capture.isOpened():
            indexes.append(index)
        capture.release()
    return indexes


@contextmanager
def _suppress_native_stderr() -> Any:
    try:
        sys.stderr.flush()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return

    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def _open_video_capture(cv2_module: Any, source: int | str, allow_default_fallback: bool = True) -> Any:
    if isinstance(source, int) and os.name == "nt" and hasattr(cv2_module, "CAP_DSHOW"):
        capture = cv2_module.VideoCapture(source, cv2_module.CAP_DSHOW)
        if capture.isOpened() or not allow_default_fallback:
            return capture
        capture.release()
    return cv2_module.VideoCapture(source)


def _validate_model(model_path: str | Path) -> Path:
    model_file = project_path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if model_file.suffix.lower() != ".pt":
        raise StreamDetectionError(f"Unsupported model format: {model_file.suffix}")
    return model_file


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
            raise StreamDetectionError("CUDA was selected, but PyTorch is not installed.") from exc
        if not torch.cuda.is_available():
            raise StreamDetectionError("CUDA was selected, but CUDA is not available.")
        return 0 if normalized == "cuda" else normalized
    raise StreamDetectionError(f"Unsupported device value: {device}")


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
    for name in ("recorded_video", "snapshots", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def _frame_class_stats(result: Any) -> tuple[dict[str, int], dict[str, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {}, {}

    names = getattr(result, "names", {})
    counts: dict[str, int] = {}
    confidence_sums: dict[str, float] = {}
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    for class_id, confidence in zip(classes, confidences, strict=False):
        class_name = names.get(int(class_id), str(int(class_id)))
        counts[class_name] = counts.get(class_name, 0) + 1
        confidence_sums[class_name] = confidence_sums.get(class_name, 0.0) + float(confidence)

    average_confidences = {
        class_name: confidence_sums[class_name] / count
        for class_name, count in counts.items()
    }
    return counts, average_confidences


def _track_status(track_state: TrackState, options: StreamDetectionOptions) -> tuple[str, list[str]]:
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


def _format_track_summary_line(item: dict[str, Any], prefix: str = "  - ") -> str:
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


def _summary_text_lines(title: str, summary: dict[str, Any]) -> list[str]:
    lines = [
        title,
        f"- window seconds: {summary.get('window_seconds', 0)}",
        f"- max objects in frame: {summary.get('max_objects_in_frame', 0)}",
        "",
        "Classes detected:",
    ]
    classes = summary.get("classes_detected")
    if isinstance(classes, dict) and classes:
        lines.extend(f"  - {class_name}: {count}" for class_name, count in sorted(classes.items()))
    else:
        lines.append("  - none")

    lines.extend(["", "Average confidence by class:"])
    confidences = summary.get("average_confidence_by_class")
    if isinstance(confidences, dict) and confidences:
        lines.extend(f"  - {class_name}: {confidence}" for class_name, confidence in sorted(confidences.items()))
    else:
        lines.append("  - none")

    unstable = summary.get("unstable_detections")
    lines.extend(["", "Unstable / suspected detections:"])
    if isinstance(unstable, list) and unstable:
        for item in unstable:
            if not isinstance(item, dict):
                continue
            lines.append(_format_track_summary_line(item))
    else:
        lines.append("  - none")
    return lines


def _write_report(
    run: StreamRunResult,
    options: StreamDetectionOptions,
    final_window_summary: dict[str, Any],
) -> None:
    lines = [
        "AnimalDetector Real-Time Detection Report",
        f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {relative_to_project(run.model_path)}",
        f"Source: {run.source}",
        f"Result folder: {relative_to_project(run.run_dir)}",
        "",
        "Parameters:",
    ]
    lines.extend(f"- {key}: {value}" for key, value in asdict(options).items())
    lines.extend(
        [
            "",
            "Summary:",
            "- total files: 1",
            "- processed files: 1",
            f"- processed frames: {run.processed_frames}",
            f"- total detections: {run.total_detections}",
            f"- average processing fps: {run.average_processing_fps:.2f}",
            f"- max objects in frame: {run.max_objects_in_frame}",
            f"- unstable or suspected detections: {len(run.unstable_detections)}",
            "",
            "Classes detected:",
        ]
    )
    if run.class_object_detections:
        lines.extend(f"  - {class_name}: {count}" for class_name, count in sorted(run.class_object_detections.items()))
    else:
        lines.append("  - none")

    lines.extend(["", "Unique tracked objects:"])
    if run.unique_tracked_objects:
        lines.extend(f"  - {class_name}: {count}" for class_name, count in sorted(run.unique_tracked_objects.items()))
    else:
        lines.append("  - none")

    lines.extend(["", "Recorded files:"])
    if run.recorded_video_path is not None:
        lines.append(f"- recorded video: {relative_to_project(run.recorded_video_path)}")
    else:
        lines.append("- recorded video: none")
    lines.extend(f"- snapshot: {relative_to_project(path)}" for path in run.snapshot_paths)
    lines.extend(["", *_summary_text_lines("Last 5 Minutes Summary", final_window_summary)])
    run.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    run: StreamRunResult,
    options: StreamDetectionOptions,
    final_window_summary: dict[str, Any],
) -> None:
    recorded_video = relative_to_project(run.recorded_video_path) if run.recorded_video_path else None
    video_record = {
        "video_name": run.recorded_video_path.name if run.recorded_video_path else "stream_detected.mp4",
        "source_video_path": str(run.source),
        "annotated_video_path": recorded_video,
        "frames": [relative_to_project(path) for path in run.snapshot_paths],
        "labels": [],
        "statistics": {
            "processed_frames": run.processed_frames,
            "total_detections": run.total_detections,
            "average_processing_fps": round(run.average_processing_fps, 4),
            "max_objects_in_frame": run.max_objects_in_frame,
            "class_frame_detections": run.class_frame_detections,
            "class_object_detections": run.class_object_detections,
            "tracking_enabled": bool(run.track_statistics),
            "unique_tracked_objects": run.unique_tracked_objects,
            "track_statistics": run.track_statistics,
            "unstable_detections": run.unstable_detections,
        },
    }
    manifest = {
        "run_type": "stream",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": relative_to_project(run.model_path),
        "model_name": run.model_path.name,
        "source_path": str(run.source),
        "result_path": relative_to_project(run.run_dir),
        "parameters": asdict(options),
        "summary": {
            "total_files": 1,
            "processed_files": 1,
            "processed_frames": run.processed_frames,
            "total_detections": run.total_detections,
            "classes_detected": run.class_object_detections,
        },
        "stream_statistics": {
            "processed_frames": run.processed_frames,
            "average_processing_fps": round(run.average_processing_fps, 4),
            "max_objects_in_frame": run.max_objects_in_frame,
            "class_frame_detections": run.class_frame_detections,
            "class_object_detections": run.class_object_detections,
            "unique_tracked_objects": run.unique_tracked_objects,
            "unstable_detections": run.unstable_detections,
            "last_window_summary": final_window_summary,
        },
        "recorded_video": recorded_video,
        "snapshots": [relative_to_project(path) for path in run.snapshot_paths],
        "logs": [
            relative_to_project(run.run_dir / "logs" / "latest_summary.json"),
            relative_to_project(run.run_dir / "logs" / "latest_summary.txt"),
        ],
        "videos": [video_record] if recorded_video else [],
    }
    with run.manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
