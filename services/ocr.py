from __future__ import annotations

from abc import ABC, abstractmethod

from services.models import OCRResult


class OCRReader(ABC):
    @abstractmethod
    def read(self, image_ref: str) -> OCRResult:
        raise NotImplementedError


class MockOCRReader(OCRReader):
    def read(self, image_ref: str) -> OCRResult:
        if "unclear" in image_ref.lower():
            return OCRResult(text="B-1145", confidence=0.42)
        return OCRResult(text="B-1145", confidence=0.78)
