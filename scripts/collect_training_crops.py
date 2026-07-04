from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from garbage_vision.app import build_detector, crop_roi
from garbage_vision.config import load_config
from garbage_vision.logging_setup import configure_logging
from garbage_vision.sources import camera_frame

LOGGER = logging.getLogger(__name__)


def save_crop(output_dir: Path, frame, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously collect ROI crop images for later YOLO annotation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/datasets/desk-trash/images/unlabeled"),
        help="Folder where cropped images will be saved.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between camera captures.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after this many saved images. Use 0 to run forever.",
    )
    parser.add_argument(
        "--prefix",
        default="desk_roi",
        help="Filename prefix for saved images.",
    )
    parser.add_argument(
        "--only-on-change",
        action="store_true",
        help="Only save crops when the current frame differs from BASELINE_IMAGE.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    configure_logging(config.log_level)
    detector = build_detector(config) if args.only_on_change else None
    saved = 0

    LOGGER.info("Saving ROI crops to %s", args.output_dir)
    LOGGER.info("Press Ctrl-C to stop")

    while True:
        try:
            frame = camera_frame(config)
            if detector is not None:
                result = detector.detect(frame)
                LOGGER.info(
                    "change check detected=%s score=%.4f area=%s reason=%s",
                    result.detected,
                    result.score,
                    result.changed_area,
                    result.reason,
                )
                if not result.detected:
                    time.sleep(args.interval)
                    continue

            crop = crop_roi(frame, config.detection_roi)
            path = save_crop(args.output_dir, crop, args.prefix)
            saved += 1
            LOGGER.info("saved %s", path)

            if args.max_images and saved >= args.max_images:
                LOGGER.info("Reached --max-images=%s", args.max_images)
                return 0

            time.sleep(args.interval)
        except KeyboardInterrupt:
            LOGGER.info("Stopped after saving %s images", saved)
            return 0
        except Exception:
            LOGGER.exception("Capture failed")
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                LOGGER.info("Stopped after saving %s images", saved)
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
