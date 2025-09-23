#!/usr/bin/env python3
"""Enhanced D435i liveness demo with depth profile, screen heuristics, and motion checks."""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs


@dataclass
class LivenessThresholds:
    min_depth_range_m: float = 0.022
    min_depth_stdev_m: float = 0.007
    min_samples: int = 120
    max_depth_m: float = 3.0

    min_center_prominence_m: float = 0.0035  # absolute floor (meters)
    min_center_prominence_ratio: float = 0.05  # relative to total range inside ROI
    max_horizontal_asymmetry_m: float = 0.12  # cheeks tolerance widened to ~12 cm

    color_mean_high: float = 235.0
    color_uniformity_std_max: float = 26.0
    color_saturation_fraction_max: float = 0.90
    color_dark_fraction_max: float = 0.95
    color_flicker_peak_to_peak: float = 70.0
    flicker_window_s: float = 2.0

    min_eye_change: float = 0.009
    min_mouth_change: float = 0.012
    min_nose_depth_change_m: float = 0.003
    min_center_shift_px: float = 2.0
    movement_window_s: float = 3.0
    min_movement_samples: int = 3


@dataclass
class LivenessConfig:
    stride: int = 3
    confidence: float = 0.6
    fps: float = 5.0
    record_seconds: int = 0
    display: bool = True
    log_to_file: bool = True
    log_path: Path = field(default_factory=lambda: Path("logs/d435i_liveness.log"))


@dataclass
class LivenessResult:
    timestamp: float
    color_image: np.ndarray
    depth_frame: rs.depth_frame
    bbox: Optional[Tuple[int, int, int, int]]
    stats: Optional[Dict[str, float]]
    depth_ok: bool
    depth_info: Dict[str, float | int | str]
    screen_ok: bool
    screen_info: Dict[str, float | int | str]
    movement_ok: bool
    movement_info: Dict[str, float | int | str]
    instant_alive: bool
    stable_alive: bool
    stability_score: float


@dataclass
class MaskInfo:
    bbox: Tuple[int, int, int, int]
    stride: int
    ellipse_mask: np.ndarray
    inner_mask: np.ndarray
    outer_mask: np.ndarray


@dataclass
class DecisionAccumulator:
    pos_gain: float = 0.25
    neg_gain: float = 0.18
    on_threshold: float = 0.65
    off_threshold: float = 0.35
    value: float = 0.0
    state: bool = False

    def update(self, positive: bool) -> Tuple[bool, float]:
        delta = self.pos_gain if positive else -self.neg_gain
        self.value = clamp(self.value + delta, 0.0, 1.0)
        if self.state:
            if self.value <= self.off_threshold:
                self.state = False
        else:
            if self.value >= self.on_threshold:
                self.state = True
        return self.state, self.value


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def bbox_from_detection(det, width: int, height: int, expansion: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
    bbox = det.location_data.relative_bounding_box
    x = bbox.xmin
    y = bbox.ymin
    w = bbox.width
    h = bbox.height
    if w <= 0 or h <= 0:
        return None
    cx = x + w / 2.0
    cy = y + h / 2.0
    w *= (1.0 + expansion)
    h *= (1.0 + expansion)
    x = cx - w / 2.0
    y = cy - h / 2.0

    x0 = int(clamp(x * width, 0, width - 1))
    y0 = int(clamp(y * height, 0, height - 1))
    x1 = int(clamp((x + w) * width, 0, width))
    y1 = int(clamp((y + h) * height, 0, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _ellipse_masks(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    if h < 2 or w < 2:
        return (np.zeros(shape, dtype=bool),) * 3
    ys, xs = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rx = max(cx, 1.0)
    ry = max(cy, 1.0)
    norm = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    ellipse = norm <= 1.0
    inner = norm <= 0.5 ** 2
    outer = (norm > 0.5 ** 2) & ellipse
    return ellipse, inner, outer


def compute_depth_metrics(
    depth_frame: rs.depth_frame,
    bbox: Tuple[int, int, int, int],
    stride: int,
    thresholds: LivenessThresholds,
) -> Tuple[Optional[Dict[str, float]], Optional[MaskInfo]]:
    depth_image = np.asanyarray(depth_frame.get_data())
    x0, y0, x1, y1 = bbox
    patch = depth_image[y0:y1, x0:x1]
    if patch.size == 0:
        return None, None
    if stride > 1:
        patch = patch[::stride, ::stride]
    patch = patch.astype(np.float32)
    depth_unit = depth_frame.get_units()
    patch *= depth_unit

    ellipse_mask, inner_mask, outer_mask = _ellipse_masks(patch.shape)
    if not ellipse_mask.any():
        return None, None

    valid = (patch > 0) & (patch < thresholds.max_depth_m) & ellipse_mask
    samples = patch[valid]
    if samples.size < thresholds.min_samples:
        return None, None

    stats: Dict[str, float] = {
        "count": float(samples.size),
        "min": float(samples.min()),
        "max": float(samples.max()),
        "mean": float(samples.mean()),
        "stdev": float(samples.std()),
    }
    stats["range"] = stats["max"] - stats["min"]

    def safe_mean(mask: np.ndarray) -> Optional[float]:
        vals = patch[mask]
        if vals.size == 0:
            return None
        return float(vals.mean())

    stats["center_mean"] = safe_mean(inner_mask & valid)
    stats["outer_mean"] = safe_mean(outer_mask & valid)
    left_mask = valid & (np.indices(patch.shape)[1] < patch.shape[1] / 2)
    right_mask = valid & (np.indices(patch.shape)[1] >= patch.shape[1] / 2)
    stats["left_mean"] = safe_mean(left_mask)
    stats["right_mean"] = safe_mean(right_mask)

    mask_info = MaskInfo(bbox=bbox, stride=stride, ellipse_mask=ellipse_mask, inner_mask=inner_mask, outer_mask=outer_mask)
    return stats, mask_info


def evaluate_depth_profile(stats: Dict[str, float], thresholds: LivenessThresholds) -> Tuple[bool, Dict[str, float]]:
    info = {
        "range": stats["range"],
        "stdev": stats["stdev"],
        "center_mean": stats.get("center_mean"),
        "outer_mean": stats.get("outer_mean"),
        "left_mean": stats.get("left_mean"),
        "right_mean": stats.get("right_mean"),
    }

    if stats["range"] < thresholds.min_depth_range_m:
        info["reason"] = "depth_range_too_small"
        return False, info
    if stats["stdev"] < thresholds.min_depth_stdev_m:
        info["reason"] = "depth_stdev_too_small"
        return False, info

    center = stats.get("center_mean")
    outer = stats.get("outer_mean")
    if center is None or outer is None:
        info["reason"] = "missing_center_outer"
        return False, info
    prominence = outer - center
    info["prominence"] = prominence
    prominence_ratio = prominence / stats["range"] if stats["range"] > 1e-6 else 0.0
    info["prominence_ratio"] = prominence_ratio
    min_required_prominence = max(
        thresholds.min_center_prominence_m,
        thresholds.min_center_prominence_ratio * stats["range"],
    )
    if prominence < min_required_prominence or prominence_ratio < thresholds.min_center_prominence_ratio:
        info["reason"] = "nose_not_prominent"
        return False, info

    left = stats.get("left_mean")
    right = stats.get("right_mean")
    if left is None or right is None:
        info["reason"] = "missing_cheeks"
        return False, info
    asymmetry = abs(left - right)
    info["asymmetry"] = asymmetry
    if asymmetry > thresholds.max_horizontal_asymmetry_m:
        info["reason"] = "cheeks_unbalanced"
        return False, info

    info["reason"] = "depth_ok"
    return True, info


def sample_color_metrics(color_image: np.ndarray, mask: MaskInfo) -> Optional[Dict[str, float]]:
    x0, y0, x1, y1 = mask.bbox
    stride = mask.stride
    roi = color_image[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    if stride > 1:
        roi = roi[::stride, ::stride]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    values = gray[mask.ellipse_mask]
    if values.size == 0:
        return None
    metrics = {
        "mean": float(values.mean()),
        "stdev": float(values.std()),
        "saturation_fraction": float(np.mean(values >= 240)),
        "dark_fraction": float(np.mean(values <= 30)),
    }
    return metrics


def evaluate_screen_suspect(
    color_metrics: Optional[Dict[str, float]],
    color_history: Deque[Tuple[float, float]],
    now: float,
    thresholds: LivenessThresholds,
) -> Tuple[bool, Dict[str, float]]:
    if not color_metrics:
        return True, {"reason": "no_color_metrics"}

    mean = color_metrics["mean"]
    stdev = color_metrics["stdev"]
    sat_frac = color_metrics["saturation_fraction"]

    suspicious = False
    reasons: List[str] = []
    if (
        mean > thresholds.color_mean_high
        and stdev < thresholds.color_uniformity_std_max
        and sat_frac > 0.2
    ):
        suspicious = True
        reasons.append("bright_uniform")
    if sat_frac > thresholds.color_saturation_fraction_max:
        suspicious = True
        reasons.append("high_saturation")
    if color_metrics.get("dark_fraction", 0.0) > thresholds.color_dark_fraction_max:
        suspicious = True
        reasons.append("very_dark")

    # flicker detection using recent brightness history
    recent = [val for (t, val) in color_history if now - t <= thresholds.flicker_window_s]
    flicker_pp = max(recent) - min(recent) if len(recent) >= 2 else 0.0
    if flicker_pp >= thresholds.color_flicker_peak_to_peak:
        suspicious = True
        reasons.append("flicker")

    info = {
        "mean": mean,
        "stdev": stdev,
        "saturation_fraction": sat_frac,
        "flicker_pp": flicker_pp,
        "reason": ",".join(reasons) if reasons else "clear",
    }
    return not suspicious, info


def _point_from_landmark(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def _aspect_ratio(indices: Tuple[int, int, int, int], landmarks, width: int, height: int) -> Optional[float]:
    top, bottom, left, right = indices
    top_pt = _point_from_landmark(landmarks.landmark[top], width, height)
    bottom_pt = _point_from_landmark(landmarks.landmark[bottom], width, height)
    left_pt = _point_from_landmark(landmarks.landmark[left], width, height)
    right_pt = _point_from_landmark(landmarks.landmark[right], width, height)

    horizontal = np.linalg.norm(left_pt - right_pt)
    vertical = np.linalg.norm(top_pt - bottom_pt)
    if horizontal < 1e-6:
        return None
    return float(vertical / horizontal)


def extract_landmark_metrics(
    face_mesh_result,
    width: int,
    height: int,
    depth_frame: rs.depth_frame,
    thresholds: LivenessThresholds,
) -> Optional[Dict[str, float]]:
    if not face_mesh_result.multi_face_landmarks:
        return None
    landmarks = face_mesh_result.multi_face_landmarks[0]

    left_eye = _aspect_ratio((159, 145, 33, 133), landmarks, width, height)
    right_eye = _aspect_ratio((386, 374, 362, 263), landmarks, width, height)
    eye_ratio = None
    if left_eye is not None and right_eye is not None:
        eye_ratio = (left_eye + right_eye) / 2.0

    mouth_ratio = _aspect_ratio((13, 14, 78, 308), landmarks, width, height)

    nose_idx = 1
    nose_landmark = landmarks.landmark[nose_idx]
    nose_x = int(clamp(nose_landmark.x * width, 0, width - 1))
    nose_y = int(clamp(nose_landmark.y * height, 0, height - 1))
    nose_depth = depth_frame.get_distance(nose_x, nose_y)
    if nose_depth <= 0 or nose_depth > thresholds.max_depth_m:
        nose_depth = None

    metrics = {
        "eye_ratio": eye_ratio,
        "mouth_ratio": mouth_ratio,
        "nose_depth": nose_depth,
    }
    return metrics


def update_movement_history(
    history: Deque[Dict[str, float]],
    metrics: Optional[Dict[str, float]],
    bbox: Tuple[int, int, int, int],
    now: float,
    thresholds: LivenessThresholds,
) -> None:
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    entry = {
        "t": now,
        "center_x": center_x,
        "center_y": center_y,
        "eye_ratio": metrics.get("eye_ratio") if metrics else None,
        "mouth_ratio": metrics.get("mouth_ratio") if metrics else None,
        "nose_depth": metrics.get("nose_depth") if metrics else None,
    }
    history.append(entry)
    # prune stale entries beyond window * 1.5 for buffer
    while history and now - history[0]["t"] > thresholds.movement_window_s * 1.5:
        history.popleft()


def _variation(values: List[float]) -> float:
    filtered = [v for v in values if v is not None]
    if len(filtered) < 2:
        return 0.0
    return float(max(filtered) - min(filtered))


def movement_liveness_ok(
    history: Deque[Dict[str, float]],
    now: float,
    thresholds: LivenessThresholds,
) -> Tuple[bool, Dict[str, float]]:
    recent = [entry for entry in history if now - entry["t"] <= thresholds.movement_window_s]
    if len(recent) < thresholds.min_movement_samples:
        return False, {"reason": "insufficient_samples", "samples": len(recent)}

    eye_var = _variation([entry["eye_ratio"] for entry in recent])
    mouth_var = _variation([entry["mouth_ratio"] for entry in recent])
    nose_var = _variation([entry["nose_depth"] for entry in recent])

    if len(recent) >= 2:
        cx_vals = [entry["center_x"] for entry in recent]
        cy_vals = [entry["center_y"] for entry in recent]
        center_shift = math.hypot(max(cx_vals) - min(cx_vals), max(cy_vals) - min(cy_vals))
    else:
        center_shift = 0.0

    movement = (
        eye_var >= thresholds.min_eye_change
        or mouth_var >= thresholds.min_mouth_change
        or nose_var >= thresholds.min_nose_depth_change_m
        or center_shift >= thresholds.min_center_shift_px
    )

    info = {
        "eye_var": eye_var,
        "mouth_var": mouth_var,
        "nose_var": nose_var,
        "center_shift": center_shift,
        "reason": "movement_ok" if movement else "movement_static",
    }
    return movement, info


class MediaPipeLiveness:
    """Single-file liveness helper that keeps all state inside one class."""

    def __init__(
        self,
        config: Optional[LivenessConfig] = None,
        thresholds: Optional[LivenessThresholds] = None,
    ) -> None:
        self.config = config or LivenessConfig()
        self.thresholds = thresholds or LivenessThresholds()
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.config.confidence,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.confidence,
        )
        self.pipe: Optional[rs.pipeline] = None
        self.align_to_color: Optional[rs.align] = None
        self.color_history: Deque[Tuple[float, float]] = deque(maxlen=180)
        self.movement_history: Deque[Dict[str, float]] = deque(maxlen=180)
        self.decision_acc = DecisionAccumulator()
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot start a closed MediaPipeLiveness instance")
        if self._started:
            return
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipe.start(cfg)
        device = profile.get_device()
        logging.info(
            "Connected to %s (S/N %s)",
            device.get_info(rs.camera_info.name),
            device.get_info(rs.camera_info.serial_number),
        )
        self.pipe = pipe
        self.align_to_color = rs.align(rs.stream.color)
        self._started = True

    def stop(self) -> None:
        if self.pipe:
            self.pipe.stop()
        self.pipe = None
        self.align_to_color = None
        self._started = False

    def close(self) -> None:
        if self._closed:
            return
        self.stop()
        if self.face_detector:
            self.face_detector.close()
            self.face_detector = None
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        self._closed = True

    def __enter__(self) -> "MediaPipeLiveness":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process(self, timeout_ms: int = 1000) -> Optional[LivenessResult]:
        if self._closed:
            raise RuntimeError("MediaPipeLiveness instance already closed")
        if not self._started:
            self.start()
        if not self.pipe or not self.align_to_color:
            raise RuntimeError("MediaPipeLiveness pipeline not started")

        frames = self.pipe.wait_for_frames(timeout_ms=timeout_ms)
        frames = self.align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        detection_result = self.face_detector.process(rgb_image)
        mesh_result = self.face_mesh.process(rgb_image)

        stats: Optional[Dict[str, float]] = None
        mask_info: Optional[MaskInfo] = None
        bbox_px: Optional[Tuple[int, int, int, int]] = None
        depth_ok = False
        depth_info: Dict[str, float | int | str] = {"reason": "no_depth"}
        screen_ok = True
        screen_info: Dict[str, float | int | str] = {"reason": "no_color_metrics"}
        movement_ok = False
        movement_info: Dict[str, float | int | str] = {"reason": "not_evaluated"}
        instant_alive = False

        detections = detection_result.detections if detection_result and detection_result.detections else []
        if detections:
            det = max(detections, key=lambda d: d.score[0])
            width = color_frame.get_width()
            height = color_frame.get_height()
            bbox_px = bbox_from_detection(det, width, height)
            if bbox_px:
                stats, mask_info = compute_depth_metrics(depth_frame, bbox_px, self.config.stride, self.thresholds)
                if stats and mask_info:
                    depth_ok, depth_info = evaluate_depth_profile(stats, self.thresholds)
                    color_metrics = sample_color_metrics(color_image, mask_info)
                    now = time.time()
                    if color_metrics:
                        self.color_history.append((now, color_metrics["mean"]))
                    screen_ok, screen_info = evaluate_screen_suspect(color_metrics, self.color_history, now, self.thresholds)

                    landmark_metrics = extract_landmark_metrics(mesh_result, width, height, depth_frame, self.thresholds)
                    update_movement_history(self.movement_history, landmark_metrics, bbox_px, now, self.thresholds)
                    movement_ok, movement_info = movement_liveness_ok(self.movement_history, now, self.thresholds)

                    instant_alive = depth_ok and screen_ok and movement_ok
                    logging.info(
                        "face_detected score=%.3f bbox=%s instant_alive=%s depth=%s screen=%s movement=%s stats=%s",
                        det.score[0],
                        bbox_px,
                        instant_alive,
                        depth_info,
                        screen_info,
                        movement_info,
                        {k: v for k, v in stats.items() if k in {"count", "min", "max", "range", "stdev", "center_mean", "outer_mean"}},
                    )
                else:
                    instant_alive = False
                    logging.info(
                        "face_detected score=%.3f bbox=%s instant_alive=%s reason=no_depth_samples",
                        det.score[0],
                        bbox_px,
                        instant_alive,
                    )
            else:
                instant_alive = False
                logging.info(
                    "face_detected score=%.3f bbox=None instant_alive=False reason=invalid_bbox",
                    det.score[0],
                )
        else:
            instant_alive = False

        stable_alive, stability_score = self.decision_acc.update(instant_alive)
        timestamp = time.time()

        return LivenessResult(
            timestamp=timestamp,
            color_image=color_image,
            depth_frame=depth_frame,
            bbox=bbox_px,
            stats=stats,
            depth_ok=depth_ok,
            depth_info=depth_info,
            screen_ok=screen_ok,
            screen_info=screen_info,
            movement_ok=movement_ok,
            movement_info=movement_info,
            instant_alive=instant_alive,
            stable_alive=stable_alive,
            stability_score=stability_score,
        )

def draw_overlay(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    instant_alive: bool,
    stable_alive: bool,
    stability_score: float,
    depth_info: Dict[str, float],
    screen_info: Dict[str, float],
    movement_info: Dict[str, float],
) -> None:
    x0, y0, x1, y1 = bbox
    color = (0, 230, 0) if stable_alive else (0, 0, 220)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

    text_lines = [
        f"live={stable_alive} inst={instant_alive} score={stability_score:.2f}",
        f"range={depth_info.get('range', 0):.3f} stdev={depth_info.get('stdev', 0):.3f}",
        f"prom={depth_info.get('prominence', 0):.3f} ratio={depth_info.get('prominence_ratio', 0):.2f} asym={depth_info.get('asymmetry', 0):.3f}",
        f"screen={screen_info.get('reason', 'n/a')} move={movement_info.get('reason', 'n/a')}",
    ]
    for idx, line in enumerate(text_lines):
        cv2.putText(
            image,
            line,
            (10, 30 + idx * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=3, help="Sub-sampling stride for ROI sampling")
    parser.add_argument("--confidence", type=float, default=0.6, help="Mediapipe detection confidence")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--fps", type=float, default=5.0, help="Status log frequency")
    parser.add_argument("--record", type=int, default=0, help="Optional recording duration in seconds (0 = run until Ctrl+C)")
    return parser.parse_args()


def setup_logging(config: LivenessConfig) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.log_to_file:
        config.log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.log_path, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def main() -> None:
    args = parse_args()
    config = LivenessConfig(
        stride=args.stride,
        confidence=args.confidence,
        fps=args.fps,
        record_seconds=args.record,
        display=not args.no_display,
    )
    setup_logging(config)
    thresholds = LivenessThresholds()

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    last_status = 0.0
    start_time = time.time()
    try:
        with MediaPipeLiveness(config=config, thresholds=thresholds) as liveness:
            while True:
                if config.record_seconds and (time.time() - start_time) >= config.record_seconds:
                    break
                try:
                    result = liveness.process(timeout_ms=1000)
                except RuntimeError as err:
                    logging.warning("Frame timeout: %s", err)
                    continue
                if result is None:
                    continue

                if result.timestamp - last_status >= (1.0 / max(config.fps, 1.0)):
                    if result.stats:
                        logging.info(
                            "status stable_alive=%s instant_alive=%s depth_ok=%s screen_ok=%s movement_ok=%s score=%.2f range=%.3f stdev=%.3f",
                            result.stable_alive,
                            result.instant_alive,
                            result.depth_ok,
                            result.screen_ok,
                            result.movement_ok,
                            result.stability_score,
                            result.stats.get("range", 0.0) if result.stats else 0.0,
                            result.stats.get("stdev", 0.0) if result.stats else 0.0,
                        )
                    else:
                        logging.info("status No reliable face/depth data detected.")
                    last_status = result.timestamp

                if config.display:
                    display = result.color_image.copy()
                    if result.bbox and result.stats:
                        draw_overlay(
                            display,
                            result.bbox,
                            result.instant_alive,
                            result.stable_alive,
                            result.stability_score,
                            result.depth_info,
                            result.screen_info,
                            result.movement_info,
                        )
                    cv2.imshow("D435i Liveness", display)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        if config.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
