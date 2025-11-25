#!/bin/bash

# Pipeline to train Mask R-CNN on original (non-augmented) dataset
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

DATASET_DIR="datasets/maskrcnn_mvtec_bottle"
MODEL_ROOT="model_maskrcnn"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_DIR="$MODEL_ROOT/maskrcnn_mvtec_noaug_${TIMESTAMP}"
MVTEC_ROOT="datasets/mvtec"
CATEGORY="bottle"
EPOCHS=50
BATCH_SIZE=4
LR=0.005
CONF=0.5
DEVICE="cpu"  # force CPU due to MPS issues

print_line() {
  printf '\n======================================================================\n'
  printf '%s\n' "$1"
  printf '======================================================================\n'
}

print_line "MASK R-CNN PIPELINE - NO AUGMENTATION DATASET"
echo "Dataset dir : $DATASET_DIR"
echo "Model dir   : $MODEL_DIR"
echo "Device      : $DEVICE"

if [ ! -d "venv" ]; then
  echo "❌ Python virtual environment (venv) not found"
  exit 1
fi

print_line "Activating virtual environment"
source venv/bin/activate

if [ ! -d "$DATASET_DIR" ]; then
  print_line "Preparing Mask R-CNN dataset (original images only)"
  python prepare_maskrcnn_dataset.py \
    --mvtec_root "$MVTEC_ROOT" \
    --category "$CATEGORY" \
    --output_dir "$DATASET_DIR" \
    --train_ratio 0.8
else
  echo "✓ Dataset already exists at $DATASET_DIR"
fi

print_line "Training Mask R-CNN (no augmentation dataset)"
python train_maskrcnn.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$MODEL_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE"

print_line "Running inference on validation set"
python inference_maskrcnn.py \
  --model "$MODEL_DIR/best.pt" \
  --dataset_dir "$DATASET_DIR" \
  --split val \
  --output_dir "$MODEL_DIR/inference_val" \
  --conf "$CONF" \
  --device "$DEVICE"

print_line "Pipeline completed"
echo "Model directory : $MODEL_DIR"
echo "Metrics         : $MODEL_DIR/inference_val/metrics.json"
