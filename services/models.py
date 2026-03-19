from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class DetectionResult:
    boxes: list[BoundingBox] = field(default_factory=list)

    def best(self, label: str) -> BoundingBox | None:
        matches = [box for box in self.boxes if box.label == label]
        if not matches:
            return None
        return max(matches, key=lambda box: box.confidence)


@dataclass(slots=True)
class ClassificationResult:
    label: str | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRResult:
    text: str | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
