from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from garbage_vision.config import AppConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectFinding:
    name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ObjectVerification:
    accepted: bool
    findings: list[ObjectFinding]
    reason: str


def crop_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)

    x, y, width, height = roi
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x, frame_width - 1))
    y1 = max(0, min(y, frame_height - 1))
    x2 = max(x1 + 1, min(x1 + width, frame_width))
    y2 = max(y1 + 1, min(y1 + height, frame_height))
    return frame[y1:y2, x1:x2], (x1, y1)


class YoloObjectVerifier:
    def __init__(self, config: AppConfig) -> None:
        self.enabled = config.object_detection_enabled
        self.require_match = config.object_verify_required
        self.model_name = config.object_model
        self.confidence = config.object_confidence
        self.image_size = config.object_image_size
        self.allowed_classes = config.object_classes
        self.roi = config.detection_roi
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO object detection is enabled, but ultralytics is not installed. "
                "Run: uv sync --extra yolo"
            ) from exc

        self._model = YOLO(self.model_name)
        return self._model

    def verify(self, frame: np.ndarray) -> ObjectVerification:
        if not self.enabled:
            return ObjectVerification(True, [], "object detection disabled")

        roi_frame, (offset_x, offset_y) = crop_roi(frame, self.roi)
        model = self._load_model()
        LOGGER.info(
            "Running YOLO on ROI crop only: width=%s height=%s imgsz=%s",
            roi_frame.shape[1],
            roi_frame.shape[0],
            self.image_size,
        )
        results = model.predict(
            roi_frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        names = getattr(model, "names", {})
        findings: list[ObjectFinding] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                name = str(names.get(class_id, class_id)).lower()
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                finding = ObjectFinding(
                    name=name,
                    confidence=confidence,
                    box=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
                findings.append(finding)

        if not self.allowed_classes:
            accepted = bool(findings)
        else:
            accepted = any(finding.name in self.allowed_classes for finding in findings)

        if accepted:
            labels = ", ".join(f"{item.name}:{item.confidence:.2f}" for item in findings)
            return ObjectVerification(True, findings, f"YOLO accepted: {labels}")

        if self.require_match:
            return ObjectVerification(False, findings, "YOLO found no allowed trash-like object")

        LOGGER.info("YOLO did not confirm trash, but OBJECT_VERIFY_REQUIRED=false")
        return ObjectVerification(True, findings, "YOLO did not confirm trash; verification not required")


def draw_findings(frame: np.ndarray, findings: list[ObjectFinding]) -> np.ndarray:
    annotated = frame.copy()
    for finding in findings:
        x1, y1, x2, y2 = finding.box
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
