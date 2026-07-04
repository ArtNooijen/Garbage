from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    return int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    return float(value)


def _roi(name: str) -> tuple[int, int, int, int] | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{name} must be x,y,width,height")
    x, y, width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} width and height must be positive")
    return x, y, width, height


def _csv(name: str, default: str = "") -> set[str]:
    value = os.getenv(name, default).strip()
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class AppConfig:
    app_mode: str
    log_level: str
    camera_source: str
    camera_snapshot_url: str
    camera_rtsp_url: str
    camera_username: str
    camera_password: str
    camera_verify_tls: bool
    image_dir: Path
    video_file: Path
    poll_seconds: int
    detection_threshold: float
    min_changed_area: int
    output_dir: Path
    baseline_image: Path
    preview_dir: Path
    detection_roi: tuple[int, int, int, int] | None
    object_detection_enabled: bool
    object_verify_required: bool
    object_model: str
    object_confidence: float
    object_image_size: int
    object_classes: set[str]
    notify_enabled: bool
    dry_run_notifications: bool
    webhook_url: str


def load_config() -> AppConfig:
    load_dotenv()

    return AppConfig(
        app_mode=os.getenv("APP_MODE", "test").strip().lower(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        camera_source=os.getenv("CAMERA_SOURCE", "snapshot").strip().lower(),
        camera_snapshot_url=os.getenv("CAMERA_SNAPSHOT_URL", "").strip(),
        camera_rtsp_url=os.getenv("CAMERA_RTSP_URL", "").strip(),
        camera_username=os.getenv("CAMERA_USERNAME", "").strip(),
        camera_password=os.getenv("CAMERA_PASSWORD", "").strip(),
        camera_verify_tls=_bool("CAMERA_VERIFY_TLS", True),
        image_dir=Path(os.getenv("IMAGE_DIR", "samples/images")),
        video_file=Path(os.getenv("VIDEO_FILE", "")),
        poll_seconds=_int("POLL_SECONDS", 60),
        detection_threshold=_float("DETECTION_THRESHOLD", 0.08),
        min_changed_area=_int("MIN_CHANGED_AREA", 1500),
        output_dir=Path(os.getenv("OUTPUT_DIR", "data/detections")),
        baseline_image=Path(os.getenv("BASELINE_IMAGE", "data/baseline/clean.jpg")),
        preview_dir=Path(os.getenv("PREVIEW_DIR", "data/previews")),
        detection_roi=_roi("DETECTION_ROI"),
        object_detection_enabled=_bool("OBJECT_DETECTION_ENABLED", False),
        object_verify_required=_bool("OBJECT_VERIFY_REQUIRED", False),
        object_model=os.getenv("OBJECT_MODEL", "yolov8n.pt").strip(),
        object_confidence=_float("OBJECT_CONFIDENCE", 0.35),
        object_image_size=_int("OBJECT_IMAGE_SIZE", 1280),
        object_classes=_csv(
            "OBJECT_CLASSES",
            "afval,vaat,bottle,cup,bowl,backpack,handbag,suitcase,book,cell phone",
        ),
        notify_enabled=_bool("NOTIFY_ENABLED", False),
        dry_run_notifications=_bool("DRY_RUN_NOTIFICATIONS", True),
        webhook_url=os.getenv("WEBHOOK_URL", "").strip(),
    )
