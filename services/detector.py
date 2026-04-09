from __future__ import annotations

from abc import ABC, abstractmethod

from services.models import BoundingBox, DetectionResult


class Detector(ABC):
    @abstractmethod
    def detect(self, image_ref: str) -> DetectionResult:
        raise NotImplementedError


class MockDetector(Detector):
    """Deterministic placeholder for local development and API integration."""

    def detect(self, image_ref: str) -> DetectionResult:
        if "empty" in image_ref.lower():
            return DetectionResult(boxes=[])

        boxes = [
            BoundingBox("aircraft", 0.98, 0.05, 0.10, 0.95, 0.90),
            BoundingBox("tail", 0.88, 0.72, 0.18, 0.90, 0.55),
        ]
        if "logo" in image_ref.lower():
            boxes.append(BoundingBox("logo", 0.83, 0.20, 0.25, 0.38, 0.42))
        return DetectionResult(boxes=boxes)
