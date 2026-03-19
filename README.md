# Aircraft Recognition

A modular deep learning system for commercial aircraft recognition from images.

## MVP scope

The repository now includes an implementation scaffold for the multi-stage architecture:

1. **Detection**: locate aircraft, tail, and optional logo regions.
2. **Classification**: predict aircraft type and airline from detected crops.
3. **OCR**: read registration text from the tail crop.
4. **Fusion**: combine component outputs into a single prediction payload.
5. **API**: expose inference through FastAPI.

## Project structure

```text
app/
  main.py                  FastAPI entrypoint
schemas/
  api.py                   Request/response contracts
services/
  pipeline.py              End-to-end orchestration
  models.py                Shared dataclasses for intermediate outputs
  detector.py              Detector interface + mock implementation
  classifier.py            Classifier interfaces + mock implementations
  ocr.py                   OCR interface + mock implementation
  fusion.py                Output fusion and failure handling
  config.py                Runtime configuration

docs/
  architecture.md          Implementable system architecture

tests/
  test_pipeline.py         Basic pipeline behavior tests
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

### `POST /predict`
Accepts either an `image_url` or `image_path` and returns fused predictions.

### `POST /batch_predict`
Accepts a list of prediction requests.

## Notes

The current code is a **production-oriented scaffold**, not a trained model package. The service layer is intentionally interface-driven so YOLOv8, EfficientNet, and Tesseract can be swapped in without changing the API or pipeline contracts.


## What you need to train and deploy this online

At a high level, you will need to:

1. collect and annotate a real dataset for detection, classification, and OCR,
2. train YOLOv8 for detection,
3. generate aircraft, tail, and logo crops,
4. train aircraft-type and airline classifiers,
5. integrate Tesseract or a learned OCR model for registrations,
6. replace the mock services in `services/` with real model adapters,
7. package the API in Docker, and
8. deploy it to a cloud host with monitoring, storage, and versioned model weights.

See `docs/training_and_deployment.md` for the concrete step-by-step plan.
