from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ThresholdConfig:
    detection: float = 0.25
    classification: float = 0.50
    ocr: float = 0.60


@dataclass(slots=True)
class AppConfig:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
