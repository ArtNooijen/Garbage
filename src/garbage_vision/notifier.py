from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import requests

from garbage_vision.config import AppConfig
from garbage_vision.detector import DetectionResult

LOGGER = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config: AppConfig, force_dry_run: bool = False) -> None:
        self.enabled = config.notify_enabled
        self.dry_run = force_dry_run or config.dry_run_notifications
        self.provider = config.notify_provider
        self.webhook_url = config.webhook_url
        self.ntfy_server = config.ntfy_server
        self.ntfy_topic = config.ntfy_topic
        self.ntfy_token = config.ntfy_token
        self.pushover_app_token = config.pushover_app_token
        self.pushover_user_key = config.pushover_user_key

    def send(
        self,
        result: DetectionResult,
        image_path: Path | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> None:
        counts = counts or {}
        message = {
            "text": "Garbage Vision detection",
            "detected": result.detected,
            "score": round(result.score, 4),
            "changed_area": result.changed_area,
            "reason": result.reason,
            "afval": counts.get("afval", 0),
            "vaat": counts.get("vaat", 0),
            "image_path": str(image_path) if image_path else None,
        }

        if self.dry_run or not self.enabled:
            LOGGER.info("Notification dry run: %s", message)
            return

        if self.provider == "ntfy":
            self._send_ntfy(message, image_path)
            return

        if self.provider == "pushover":
            self._send_pushover(message, image_path)
            return

        if self.provider != "webhook":
            LOGGER.warning("Unknown NOTIFY_PROVIDER=%s", self.provider)
            return

        self._send_webhook(message)

    def _send_webhook(self, message: dict[str, object]) -> None:
        if not self.webhook_url:
            LOGGER.warning("NOTIFY_ENABLED is true but WEBHOOK_URL is empty")
            return

        response = requests.post(self.webhook_url, json=message, timeout=15)
        response.raise_for_status()
        LOGGER.info("Webhook notification sent")

    def _send_ntfy(self, message: dict[str, object], image_path: Path | None) -> None:
        if not self.ntfy_topic:
            LOGGER.warning("NOTIFY_PROVIDER=ntfy but NTFY_TOPIC is empty")
            return

        url = f"{self.ntfy_server}/{self.ntfy_topic}"
        body = (
            f"Detection accepted: afval={message['afval']} vaat={message['vaat']} "
            f"score={message['score']} area={message['changed_area']}"
        )
        headers = {
            "Title": "Garbage Vision",
            "Message": body,
            "Priority": "default",
            "Tags": "wastebasket",
        }
        if self.ntfy_token:
            headers["Authorization"] = f"Bearer {self.ntfy_token}"

        if image_path and image_path.exists():
            headers["Filename"] = image_path.name
            with image_path.open("rb") as file:
                response = requests.put(url, data=file, headers=headers, timeout=30)
        else:
            response = requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=15)

        response.raise_for_status()
        LOGGER.info("ntfy notification sent")

    def _send_pushover(self, message: dict[str, object], image_path: Path | None) -> None:
        if not self.pushover_app_token or not self.pushover_user_key:
            LOGGER.warning(
                "NOTIFY_PROVIDER=pushover but PUSHOVER_APP_TOKEN or PUSHOVER_USER_KEY is empty"
            )
            return

        data = {
            "token": self.pushover_app_token,
            "user": self.pushover_user_key,
            "title": "Garbage Vision",
            "message": (
                f"Detection accepted: afval={message['afval']} vaat={message['vaat']}\n"
                f"score={message['score']} area={message['changed_area']}"
            ),
        }
        files = None
        file_handle = None
        try:
            if image_path and image_path.exists():
                file_handle = image_path.open("rb")
                files = {"attachment": (image_path.name, file_handle, "image/jpeg")}
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files=files,
                timeout=30,
            )
            response.raise_for_status()
            LOGGER.info("Pushover notification sent")
        finally:
            if file_handle:
                file_handle.close()
