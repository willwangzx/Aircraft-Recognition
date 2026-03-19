from fastapi.testclient import TestClient

from app.main import app
from services.pipeline import AircraftRecognitionPipeline


client = TestClient(app)


def test_pipeline_returns_prediction_for_detected_aircraft() -> None:
    pipeline = AircraftRecognitionPipeline()

    result = pipeline.predict("sample_aircraft.jpg")

    assert result["status"] == "ok"
    assert result["aircraft"] == "A320-214"
    assert result["airline"] == "China Southern"
    assert result["registration"] == "B-1145"


def test_pipeline_handles_missing_aircraft() -> None:
    pipeline = AircraftRecognitionPipeline()

    result = pipeline.predict("empty_scene.jpg")

    assert result["status"] == "no_aircraft_detected"
    assert result["aircraft"] is None
    assert result["registration"] is None


def test_pipeline_discards_low_confidence_ocr() -> None:
    pipeline = AircraftRecognitionPipeline()

    result = pipeline.predict("sample_unclear.jpg")

    assert result["status"] == "ok"
    assert result["registration"] is None


def test_predict_endpoint_accepts_image_url() -> None:
    response = client.post("/predict", json={"image_url": "https://example.com/aircraft.jpg"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["aircraft"] == "A320-214"
