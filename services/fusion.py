from __future__ import annotations

from dataclasses import asdict

from services.config import ThresholdConfig
from services.models import ClassificationResult, DetectionResult, OCRResult


class PredictionFusionService:
    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds

    def fuse(
        self,
        image_ref: str,
        detections: DetectionResult,
        aircraft: ClassificationResult,
        airline: ClassificationResult,
        registration: OCRResult,
    ) -> dict:
        aircraft_box = detections.best("aircraft")
        tail_box = detections.best("tail")

        if aircraft_box is None or aircraft_box.confidence < self.thresholds.detection:
            return {
                "image": image_ref,
                "status": "no_aircraft_detected",
                "aircraft": None,
                "airline": None,
                "registration": None,
                "confidence": {},
            }

        aircraft_label = aircraft.label if aircraft.confidence >= self.thresholds.classification else None
        airline_label = airline.label if airline.confidence >= self.thresholds.classification else None
        registration_text = None
        if tail_box is not None and registration.confidence >= self.thresholds.ocr:
            registration_text = registration.text

        return {
            "image": image_ref,
            "status": "ok",
            "aircraft": aircraft_label,
            "airline": airline_label,
            "registration": registration_text,
            "confidence": {
                "detection": aircraft_box.confidence,
                "aircraft": aircraft.confidence,
                "airline": airline.confidence,
                "registration": registration.confidence if tail_box is not None else 0.0,
            },
            "detections": [asdict(box) for box in detections.boxes],
        }
