# Aircraft Recognition System Architecture

## 1. End-to-end flow

```text
Data -> Annotation -> Training -> Model Registry -> Inference API -> Client
```

The system is intentionally modular so detection, classification, and OCR can evolve independently while sharing common interfaces.

## 2. Dataset design

```text
dataset/
 ├── images/
 ├── labels/
 │    ├── detection/
 │    └── classification/
 ├── crops/
 │    ├── aircraft/
 │    ├── tail/
 │    └── logo/
 └── metadata.csv
```

### `metadata.csv`

```text
image_id,aircraft_type,airline,registration
img_001,A320,China Southern,B-1145
```

## 3. Model responsibilities

### Detector
- Recommended model: YOLOv8.
- Input: full image.
- Outputs: aircraft, tail, and optionally logo boxes.

### Aircraft classifier
- Recommended model: EfficientNet.
- Input: aircraft crop.
- Output: aircraft family or exact model.

### Airline classifier
- MVP input: aircraft crop.
- Advanced input: logo crop.
- Output: airline label.

### Registration OCR
- MVP engine: Tesseract OCR.
- Advanced model: CRNN or transformer OCR.
- Input: tail crop with preprocessing.

## 4. Training workflow

1. Train YOLO on aircraft, tail, and logo labels.
2. Generate aircraft, tail, and logo crops from detector outputs.
3. Train aircraft and airline classifiers on saved crops.
4. Train or integrate OCR on tail crops.

## 5. Runtime inference workflow

```text
Input image
    ↓
Detector
    ↓
Crop aircraft / tail / logo
    ↓
Aircraft classifier + Airline classifier + OCR
    ↓
Output fusion
```

## 6. Failure handling

- No aircraft detected -> return `null`-like prediction fields with an explanatory status.
- Tail not detected -> registration remains `null`.
- Low OCR confidence -> registration discarded.
- Low classifier confidence -> label returned only if confidence passes threshold.

## 7. Scaling roadmap

- Replace mock services with YOLOv8, EfficientNet, and Tesseract adapters.
- Add model registry and experiment tracking.
- Add batch inference workers.
- Add shared multi-task backbone for aircraft and airline prediction.
- Optimize deployment with ONNX or TensorRT.
