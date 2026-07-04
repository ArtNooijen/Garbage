from __future__ import annotations

import argparse
from collections import Counter
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from garbage_vision.config import AppConfig, load_config
from garbage_vision.detector import DetectionResult, MotionBaselineDetector, ReferenceImageDetector
from garbage_vision.logging_setup import configure_logging
from garbage_vision.notifier import Notifier
from garbage_vision.object_detector import ObjectFinding, YoloObjectVerifier
from garbage_vision.sources import camera_frame, image_frames, video_frames

LOGGER = logging.getLogger(__name__)


class ChangedRegion:
    def __init__(self, box: tuple[int, int, int, int], area: float) -> None:
        self.box = box
        self.area = area


def save_latest_detection(output_dir: Path, name: str, frame: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    cv2.imwrite(str(path), frame)
    return path


def save_image(path: Path, frame: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return path


def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    preview = frame.copy()
    if roi is None:
        cv2.putText(
            preview,
            "DETECTION_ROI not set; full frame is checked",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        return preview

    x, y, width, height = roi
    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 255), 4)
    cv2.putText(
        preview,
        f"DETECTION_ROI {x},{y},{width},{height}",
        (x, max(30, y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return preview


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return frame
    x, y, width, height = roi
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x, frame_width - 1))
    y1 = max(0, min(y, frame_height - 1))
    x2 = max(x1 + 1, min(x1 + width, frame_width))
    y2 = max(y1 + 1, min(y1 + height, frame_height))
    return frame[y1:y2, x1:x2]


def roi_bounds(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if roi is None:
        frame_height, frame_width = frame.shape[:2]
        return 0, 0, frame_width, frame_height

    x, y, width, height = roi
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x, frame_width - 1))
    y1 = max(0, min(y, frame_height - 1))
    x2 = max(x1 + 1, min(x1 + width, frame_width))
    y2 = max(y1 + 1, min(y1 + height, frame_height))
    return x1, y1, x2, y2


def find_roi_changes(
    frame: np.ndarray,
    baseline_path: Path,
    roi: tuple[int, int, int, int] | None,
) -> list[ChangedRegion]:
    crop = crop_roi(frame, roi)
    baseline = cv2.imread(str(baseline_path))
    if baseline is None:
        return []

    baseline_crop = crop_roi(baseline, roi)
    if baseline_crop.shape[:2] != crop.shape[:2]:
        baseline_crop = cv2.resize(baseline_crop, (crop.shape[1], crop.shape[0]))

    current_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    baseline_gray = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
    baseline_gray = cv2.GaussianBlur(baseline_gray, (21, 21), 0)
    diff = cv2.absdiff(baseline_gray, current_gray)
    _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresholded = cv2.dilate(thresholded, None, iterations=2)

    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    regions: list[ChangedRegion] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append(ChangedRegion((x, y, x + width, y + height), area))
    return regions


def _intersection_ratio(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return intersection / area_a


def draw_roi_changes(
    crop: np.ndarray,
    regions: list[ChangedRegion],
    findings: list[ObjectFinding],
    roi_origin: tuple[int, int],
) -> np.ndarray:
    annotated = crop.copy()
    offset_x, offset_y = roi_origin
    local_finding_boxes = [
        (
            finding.box[0] - offset_x,
            finding.box[1] - offset_y,
            finding.box[2] - offset_x,
            finding.box[3] - offset_y,
        )
        for finding in findings
    ]

    for region in regions:
        x1, y1, x2, y2 = region.box
        overlaps_known_object = any(
            _intersection_ratio(region.box, finding_box) > 0.25
            for finding_box in local_finding_boxes
        )
        color = (0, 0, 255) if overlaps_known_object else (0, 220, 0)
        label = "changed" if overlaps_known_object else "trash candidate"
        thickness = 2 if overlaps_known_object else 4
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if overlaps_known_object else 0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_findings_on_roi_crop(
    crop: np.ndarray,
    findings: list[ObjectFinding],
    roi_origin: tuple[int, int],
) -> np.ndarray:
    annotated = crop.copy()
    offset_x, offset_y = roi_origin
    for finding in findings:
        x1, y1, x2, y2 = finding.box
        x1 -= offset_x
        x2 -= offset_x
        y1 -= offset_y
        y2 -= offset_y
        x1 = max(0, min(x1, annotated.shape[1] - 1))
        x2 = max(0, min(x2, annotated.shape[1] - 1))
        y1 = max(0, min(y1, annotated.shape[0] - 1))
        y2 = max(0, min(y2, annotated.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 3)
        label = f"{finding.name} {finding.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def filter_allowed_findings(
    findings: list[ObjectFinding],
    allowed_classes: set[str],
    trash_classes: set[str],
    dish_classes: set[str],
    dish_cup_classes: set[str],
    dish_cup_threshold: int,
) -> list[ObjectFinding]:
    primary_matches = [
        finding
        for finding in findings
        if finding.model == "primary" and finding.name in allowed_classes
    ]
    if not allowed_classes:
        primary_matches = [finding for finding in findings if finding.model == "primary"]

    dish_matches = [finding for finding in findings if finding.name in dish_classes]
    cup_matches = [finding for finding in findings if finding.name in dish_cup_classes]
    if len(cup_matches) >= dish_cup_threshold:
        dish_matches.extend(cup_matches)
    trash_matches = [finding for finding in findings if finding.name in trash_classes]

    display_findings = primary_matches + trash_matches + dish_matches
    seen: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    deduped: list[ObjectFinding] = []
    for finding in display_findings:
        key = (finding.model, finding.name, finding.box)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def format_detection_counts(
    findings: list[ObjectFinding],
    trash_classes: set[str],
    dish_classes: set[str],
    dish_cup_classes: set[str],
) -> str:
    counts = Counter(finding.name for finding in findings)
    trash_count = counts.get("afval", 0) + sum(
        counts[class_name] for class_name in trash_classes if class_name != "afval"
    )
    dish_count = counts.get("vaat", 0)
    dish_count += sum(counts[class_name] for class_name in dish_classes if class_name != "vaat")
    dish_count += sum(counts[class_name] for class_name in dish_cup_classes)
    return f"afval={trash_count} vaat={dish_count}"


def detection_counts(
    findings: list[ObjectFinding],
    trash_classes: set[str],
    dish_classes: set[str],
    dish_cup_classes: set[str],
) -> dict[str, int]:
    counts = Counter(finding.name for finding in findings)
    trash_count = counts.get("afval", 0) + sum(
        counts[class_name] for class_name in trash_classes if class_name != "afval"
    )
    dish_count = counts.get("vaat", 0)
    dish_count += sum(counts[class_name] for class_name in dish_classes if class_name != "vaat")
    dish_count += sum(counts[class_name] for class_name in dish_cup_classes)
    return {
        "afval": trash_count,
        "vaat": dish_count,
    }


def suppress_notification_reason(
    findings: list[ObjectFinding],
    suppress_classes: set[str],
) -> str | None:
    if not suppress_classes:
        return None
    detected = sorted(
        {finding.name for finding in findings if finding.name in suppress_classes}
    )
    if not detected:
        return None
    return "suppressed because detected: " + ", ".join(detected)


def build_detector(config: AppConfig) -> MotionBaselineDetector:
    if config.baseline_image.exists():
        baseline = cv2.imread(str(config.baseline_image))
        if baseline is None:
            raise RuntimeError(f"Could not read BASELINE_IMAGE: {config.baseline_image}")
        LOGGER.info("Using clean reference baseline: %s", config.baseline_image)
        return ReferenceImageDetector(
            config.detection_threshold,
            config.min_changed_area,
            baseline,
            config.detection_roi,
        )

    LOGGER.info("No clean reference baseline found; using first frame as baseline")
    return MotionBaselineDetector(
        config.detection_threshold,
        config.min_changed_area,
        config.detection_roi,
    )


def handle_frame(
    label: str,
    frame: np.ndarray,
    detector: MotionBaselineDetector,
    notifier: Notifier,
    verifier: YoloObjectVerifier,
    config: AppConfig,
) -> DetectionResult:
    result = detector.detect(frame)
    LOGGER.info(
        "%s detected=%s score=%.4f area=%s reason=%s",
        label,
        result.detected,
        result.score,
        result.changed_area,
        result.reason,
    )

    if result.detected:
        verification = verifier.verify(frame)
        display_findings = filter_allowed_findings(
            verification.findings,
            config.object_classes,
            config.trash_classes,
            config.dish_classes,
            config.dish_cup_classes,
            config.dish_cup_threshold,
        )
        LOGGER.info(
            "YOLO result accepted=%s %s",
            verification.accepted,
            format_detection_counts(
                display_findings,
                config.trash_classes,
                config.dish_classes,
                config.dish_cup_classes,
            ),
        )
        if verification.accepted:
            x1, y1, _, _ = roi_bounds(frame, config.detection_roi)
            crop = crop_roi(frame, config.detection_roi)
            regions = find_roi_changes(frame, config.baseline_image, config.detection_roi)
            crop_frame = draw_roi_changes(crop, regions, verification.findings, (x1, y1))
            crop_frame = draw_findings_on_roi_crop(crop_frame, display_findings, (x1, y1))
            full_path = save_latest_detection(config.output_dir, "latest.jpg", crop)
            crop_path = save_latest_detection(config.output_dir, "latest_roi_marked.jpg", crop_frame)
            LOGGER.info("Saved images latest=%s marked=%s", full_path, crop_path)
            suppress_reason = suppress_notification_reason(
                verification.findings,
                config.notification_suppress_classes,
            )
            if suppress_reason:
                LOGGER.info("Notification %s", suppress_reason)
            else:
                notifier.send(
                    result,
                    crop_path,
                    detection_counts(
                        display_findings,
                        config.trash_classes,
                        config.dish_classes,
                        config.dish_cup_classes,
                    ),
                )
        else:
            LOGGER.info("Change suppressed because object verification rejected it")

    return result


def run_test(config: AppConfig, source: str) -> int:
    detector = build_detector(config)
    notifier = Notifier(config, force_dry_run=True)
    verifier = YoloObjectVerifier(config)
    total = 0
    detections = 0

    if source == "images":
        frames = image_frames(config.image_dir)
    elif source == "video":
        frames = video_frames(config.video_file)
    elif source == "camera":
        frames = [("camera", camera_frame(config))]
    else:
        raise ValueError("Test source must be images, video, or camera")

    for label, frame in frames:
        total += 1
        result = handle_frame(label, frame, detector, notifier, verifier, config)
        detections += int(result.detected)

    LOGGER.info("Test complete: frames=%s detections=%s", total, detections)
    return 0


def run_prod(config: AppConfig) -> int:
    detector = build_detector(config)
    notifier = Notifier(config)
    verifier = YoloObjectVerifier(config)
    LOGGER.info(
        "Starting production loop source=%s poll_seconds=%s",
        config.camera_source,
        config.poll_seconds,
    )

    while True:
        try:
            frame = camera_frame(config)
            handle_frame("camera", frame, detector, notifier, verifier, config)
        except Exception:
            LOGGER.exception("Production camera check failed")
        time.sleep(config.poll_seconds)


def capture_baseline(config: AppConfig) -> int:
    frame = camera_frame(config)
    path = save_image(config.baseline_image, frame)
    LOGGER.info("Saved clean baseline image: %s", path)
    return 0


def preview_roi(config: AppConfig) -> int:
    if config.baseline_image.exists():
        frame = cv2.imread(str(config.baseline_image))
        if frame is None:
            raise RuntimeError(f"Could not read BASELINE_IMAGE: {config.baseline_image}")
    else:
        frame = camera_frame(config)

    config.preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = save_image(config.preview_dir / "roi_preview.jpg", draw_roi(frame, config.detection_roi))
    crop_path = save_image(config.preview_dir / "roi_crop.jpg", crop_roi(frame, config.detection_roi))
    LOGGER.info("Saved ROI preview image: %s", preview_path)
    LOGGER.info("Saved ROI crop image: %s", crop_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Garbage Vision camera detector")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["test", "prod"],
        help="Run once in test mode or continuously in production mode",
    )
    parser.add_argument(
        "--capture-baseline",
        action="store_true",
        help="Save the current camera frame as the clean reference baseline",
    )
    parser.add_argument(
        "--preview-roi",
        action="store_true",
        help="Save preview images showing the current detection area",
    )
    parser.add_argument(
        "--source",
        choices=["images", "video", "camera"],
        default="images",
        help="Input source for test mode",
    )
    return parser.parse_args()


def main() -> int:
    config = load_config()
    configure_logging(config.log_level)
    args = parse_args()
    if args.capture_baseline:
        return capture_baseline(config)
    if args.preview_roi:
        return preview_roi(config)

    mode = args.mode or config.app_mode

    if mode == "prod":
        return run_prod(config)
    return run_test(config, args.source)


if __name__ == "__main__":
    raise SystemExit(main())
