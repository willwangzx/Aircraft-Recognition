from __future__ import annotations

from services.classifier import Classifier, MockAircraftClassifier, MockAirlineClassifier
from services.config import AppConfig
from services.detector import Detector, MockDetector
from services.fusion import PredictionFusionService
from services.ocr import MockOCRReader, OCRReader


class AircraftRecognitionPipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        aircraft_classifier: Classifier | None = None,
        airline_classifier: Classifier | None = None,
        ocr_reader: OCRReader | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.detector = detector or MockDetector()
        self.aircraft_classifier = aircraft_classifier or MockAircraftClassifier()
        self.airline_classifier = airline_classifier or MockAirlineClassifier()
        self.ocr_reader = ocr_reader or MockOCRReader()
        self.fusion = PredictionFusionService(self.config.thresholds)

    def predict(self, image_ref: str) -> dict:
        detections = self.detector.detect(image_ref)
        aircraft_crop_ref = f"{image_ref}#aircraft"
        tail_crop_ref = f"{image_ref}#tail"

        aircraft = self.aircraft_classifier.classify(aircraft_crop_ref)
        airline = self.airline_classifier.classify(aircraft_crop_ref)
        registration = self.ocr_reader.read(tail_crop_ref)

        return self.fusion.fuse(image_ref, detections, aircraft, airline, registration)
