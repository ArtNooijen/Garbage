from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    score: float
    changed_area: int
    reason: str


class MotionBaselineDetector:
    """Simple frame-to-frame change detector for proving the pipeline end to end."""

    def __init__(
        self,
        threshold: float,
        min_changed_area: int,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_changed_area = min_changed_area
        self.roi = roi
        self._baseline: np.ndarray | None = None

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        x, y, width, height = self.roi
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(x, frame_width - 1))
        y1 = max(0, min(y, frame_height - 1))
        x2 = max(x1 + 1, min(x1 + width, frame_width))
        y2 = max(y1 + 1, min(y1 + height, frame_height))
        return frame[y1:y2, x1:x2]

    def detect(self, frame: np.ndarray) -> DetectionResult:
        frame = self._crop(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._baseline is None:
            self._baseline = gray
            return DetectionResult(False, 0.0, 0, "baseline initialized")

        diff = cv2.absdiff(self._baseline, gray)
        _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresholded = cv2.dilate(thresholded, None, iterations=2)

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        changed_area = sum(cv2.contourArea(contour) for contour in contours)
        score = changed_area / float(frame.shape[0] * frame.shape[1])
        detected = score >= self.threshold and changed_area >= self.min_changed_area

        self._baseline = gray
        reason = "change threshold met" if detected else "change below threshold"
        return DetectionResult(detected, score, int(changed_area), reason)


class ReferenceImageDetector(MotionBaselineDetector):
    """Compare every frame to a fixed clean reference image."""

    def __init__(
        self,
        threshold: float,
        min_changed_area: int,
        baseline: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__(threshold, min_changed_area, roi)
        baseline = self._crop(baseline)
        gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        self._baseline = cv2.GaussianBlur(gray, (21, 21), 0)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._baseline is None:
            raise RuntimeError("Reference baseline is not initialized")

        frame = self._crop(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(self._baseline, gray)
        _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresholded = cv2.dilate(thresholded, None, iterations=2)

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        changed_area = sum(cv2.contourArea(contour) for contour in contours)
        score = changed_area / float(frame.shape[0] * frame.shape[1])
        detected = score >= self.threshold and changed_area >= self.min_changed_area
        result = DetectionResult(
            detected,
            score,
            int(changed_area),
            "change threshold met" if detected else "change below threshold",
        )
        reason = "reference " + result.reason
        return DetectionResult(result.detected, result.score, result.changed_area, reason)
