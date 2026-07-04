from __future__ import annotations

import logging
from pathlib import Path

import requests

from garbage_vision.config import AppConfig
from garbage_vision.detector import DetectionResult

LOGGER = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config: AppConfig, force_dry_run: bool = False) -> None:
        self.enabled = config.notify_enabled
        self.dry_run = force_dry_run or config.dry_run_notifications
        self.webhook_url = config.webhook_url

    def send(self, result: DetectionResult, image_path: Path | None = None) -> None:
        message = {
            "text": "Garbage Vision detection",
            "detected": result.detected,
            "score": round(result.score, 4),
            "changed_area": result.changed_area,
            "reason": result.reason,
            "image_path": str(image_path) if image_path else None,
        }

        if self.dry_run or not self.enabled:
            LOGGER.info("Notification dry run: %s", message)
            return

        if not self.webhook_url:
            LOGGER.warning("NOTIFY_ENABLED is true but WEBHOOK_URL is empty")
            return

        response = requests.post(self.webhook_url, json=message, timeout=15)
        response.raise_for_status()
        LOGGER.info("Notification sent")
