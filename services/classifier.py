from __future__ import annotations

from abc import ABC, abstractmethod

from services.models import ClassificationResult


class Classifier(ABC):
    @abstractmethod
    def classify(self, image_ref: str) -> ClassificationResult:
        raise NotImplementedError


class MockAircraftClassifier(Classifier):
    def classify(self, image_ref: str) -> ClassificationResult:
        if "boeing" in image_ref.lower():
            return ClassificationResult(label="B737-800", confidence=0.89)
        return ClassificationResult(label="A320-214", confidence=0.94)


class MockAirlineClassifier(Classifier):
    def classify(self, image_ref: str) -> ClassificationResult:
        if "delta" in image_ref.lower():
            return ClassificationResult(label="Delta Air Lines", confidence=0.91)
        return ClassificationResult(label="China Southern", confidence=0.90)
