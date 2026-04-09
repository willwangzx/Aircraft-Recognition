from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PredictRequest:
    image_url: str | None = None
    image_path: str | None = None

    def __post_init__(self) -> None:
        if not self.image_url and not self.image_path:
            raise ValueError("Either image_url or image_path must be provided.")

    @property
    def image_ref(self) -> str:
        return self.image_url or self.image_path or ""


@dataclass(slots=True)
class BatchPredictRequest:
    items: list[PredictRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized: list[PredictRequest] = []
        for item in self.items:
            normalized.append(item if isinstance(item, PredictRequest) else PredictRequest(**item))
        if not normalized:
            raise ValueError("At least one prediction item must be provided.")
        self.items = normalized
