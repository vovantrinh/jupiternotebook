#!/bin/bash

# Full pipeline to train YOLO on original (non-augmented) MVTec data
# Steps: prepare dataset -> train YOLO

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

DATASET_SOURCE="datasets/mvtec/bottle"
YOLO_DATASET="datasets/yolo_mvtec_bottle_noaug"
YOLO_YAML="datasets/yolo_yolo_mvtec_bottle_noaug.yaml"
PROJECT_DIR="runs/yolo_mvtec_noaug"
MODEL_SIZE="n"
EPOCHS=100
BATCH=16
IMGSZ=640
DEVICE=""  # auto-detect

print_line() {
  printf '\n======================================================================\n'
  printf '%s\n' "$1"
  printf '======================================================================\n'
}

print_line "YOLO PIPELINE - ORIGINAL DATASET (NO AUGMENTATION)"
echo "Source dataset : $DATASET_SOURCE"
echo "YOLO dataset   : $YOLO_DATASET"
echo "Dataset YAML   : $YOLO_YAML"
echo "Project dir    : $PROJECT_DIR"

if [ ! -d "venv" ]; then
  echo "❌ Python virtual environment (venv) not found"
  exit 1
fi

print_line "Activating virtual environment"
source venv/bin/activate

if [ ! -d "$DATASET_SOURCE" ]; then
  echo "❌ Source dataset not found at $DATASET_SOURCE"
  exit 1
fi

if [ ! -d "$YOLO_DATASET" ]; then
  print_line "Preparing YOLO dataset (no augmentation)"
  python prepare_yolo_dataset.py \
    --source "$DATASET_SOURCE" \
    --output "$YOLO_DATASET" \
    --train_ratio 0.8 \
    --seed 42
else
  echo "✓ YOLO dataset already exists at $YOLO_DATASET"
fi

if [ ! -f "$YOLO_YAML" ]; then
  echo "❌ Dataset YAML not found at $YOLO_YAML"
  exit 1
fi

print_line "Training YOLO model"
python train_yolo.py \
  --data "$YOLO_YAML" \
  --model "$MODEL_SIZE" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$PROJECT_DIR"

print_line "Pipeline completed"
echo "Models saved under: $PROJECT_DIR"
