from __future__ import annotations

from fastapi import FastAPI

from schemas.api import BatchPredictRequest, PredictRequest
from services.pipeline import AircraftRecognitionPipeline

app = FastAPI(title="Aircraft Recognition API", version="0.1.0")
pipeline = AircraftRecognitionPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    return pipeline.predict(request.image_ref)


@app.post("/batch_predict")
def batch_predict(request: BatchPredictRequest) -> dict[str, list[dict]]:
    return {"items": [pipeline.predict(item.image_ref) for item in request.items]}
