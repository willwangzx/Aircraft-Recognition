# Training and Deployment Guide

This repository is currently a **skeleton**. To turn it into a real aircraft-recognition product, you need to replace the mock services with trained models, package the weights, and deploy the API behind a public endpoint.

## 1. What you need before training

### Data
You need a dataset with, at minimum:
- aircraft images
- bounding boxes for `aircraft`, `tail`, and optionally `logo`
- aircraft type labels such as `A320`, `B737-800`, and `A350-900`
- airline labels such as `Delta Air Lines` or `China Southern`
- registration text labels for OCR

### Recommended directory layout

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── detection/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── classification/
├── crops/
│   ├── aircraft/
│   ├── tail/
│   └── logo/
└── metadata.csv
```

### Tooling
For a practical training pipeline, install:
- Python 3.10+
- PyTorch with CUDA if you have an NVIDIA GPU
- Ultralytics YOLOv8 for detection
- timm or torchvision for EfficientNet-based classification
- Tesseract OCR for a baseline text-recognition stage
- MLflow or Weights & Biases for experiment tracking

## 2. Step-by-step training plan

## Step 1: Train the detector
Train YOLOv8 to detect aircraft, tail, and logo boxes.

### Output
You should produce a detector checkpoint such as:

```text
artifacts/detector/yolov8-aircraft.pt
```

### What success looks like
- high recall on aircraft boxes
- acceptable precision on tail boxes
- stable validation mAP

## Step 2: Auto-generate crops
Run the detector over your labeled dataset and save:
- aircraft crops for aircraft-type classification
- aircraft or logo crops for airline classification
- tail crops for registration OCR

### Output

```text
artifacts/crops/aircraft/
artifacts/crops/logo/
artifacts/crops/tail/
```

## Step 3: Train the aircraft classifier
Train an EfficientNet classifier on aircraft crops.

### Output

```text
artifacts/classifiers/aircraft_type.pt
```

### Important considerations
- keep aircraft families balanced
- merge very rare classes if needed
- use top-1 and top-5 accuracy during evaluation

## Step 4: Train the airline classifier
Start with aircraft-body crops, then move to logo crops if airline recognition is inconsistent.

### Output

```text
artifacts/classifiers/airline.pt
```

## Step 5: Train or integrate OCR
For the MVP, install and evaluate Tesseract on tail crops.
For higher accuracy, train a CRNN or transformer OCR model.

### Output

```text
artifacts/ocr/registration_ocr.pt
```

## Step 6: Evaluate the full pipeline end-to-end
Do not stop at per-model metrics. Measure complete-system accuracy on:
- aircraft model prediction
- airline prediction
- registration extraction
- complete structured output accuracy

## 3. What you need to change in this repository

The current `services/` modules are mocks. Replace them with production adapters.

### Replace the detector
Update `services/detector.py` so `Detector.detect()`:
- loads a YOLOv8 checkpoint
- accepts an image path or downloaded image
- returns `DetectionResult` boxes from real model outputs

### Replace the classifiers
Update `services/classifier.py` so the classifier implementations:
- load trained EfficientNet weights
- preprocess crops correctly
- return calibrated confidence scores

### Replace OCR
Update `services/ocr.py` so `OCRReader.read()`:
- preprocesses tail crops
- runs Tesseract or your OCR model
- returns cleaned registration text and confidence

### Add model loading configuration
Extend `services/config.py` with:
- model file paths
- class label maps
- confidence thresholds
- device selection (`cpu` or `cuda`)

## 4. What you need to deploy online

To deploy publicly, you need more than model files.

### Application layer
- a real FastAPI app
- model-loading on startup
- request validation
- image download or upload handling
- structured logging

### Packaging
- a Dockerfile
- pinned Python dependencies
- startup command such as `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Infrastructure
- a cloud host such as AWS, GCP, Azure, RunPod, or Hugging Face Inference Endpoints
- CPU for MVP or GPU for higher throughput
- object storage for model checkpoints and sample images
- a container registry
- a domain name and TLS if exposed publicly

### Operations
- health checks
- monitoring and alerting
- API authentication or rate limiting
- versioned model releases
- rollback strategy

## 5. Recommended deployment architecture

```text
Client
  ↓
HTTPS Load Balancer
  ↓
FastAPI inference service
  ↓
Mounted or downloaded model weights
  ↓
Optional Redis queue / batch worker
  ↓
Logs + metrics + object storage
```

## 6. Fastest path to an online MVP

If you want the quickest practical version, do this in order:

1. Train only the aircraft detector and aircraft-type classifier.
2. Replace the mock detector and aircraft classifier first.
3. Keep airline and OCR optional until the first endpoint works reliably.
4. Containerize the API.
5. Deploy a single API container to a cloud VM or container platform.
6. Add monitoring and a simple `/health` endpoint check.

## 7. Suggested milestone plan

### Milestone 1: Detection + aircraft type
- one trained YOLO model
- one trained EfficientNet model
- one public `/predict` endpoint

### Milestone 2: Airline recognition
- add airline labels
- train airline classifier
- fuse into final output

### Milestone 3: Registration OCR
- train or integrate OCR
- add tail-specific preprocessing
- add confidence-based rejection

### Milestone 4: Production hardening
- Docker image
- model registry
- CI/CD
- autoscaling and monitoring

## 8. Deployment checklist

Before going live, verify all of the following:
- model weights are versioned
- inference latency is measured
- memory usage fits your instance size
- invalid images are rejected safely
- logs do not expose sensitive data
- batch and single-image APIs are both tested
- the deployed endpoint is protected by authentication if needed
