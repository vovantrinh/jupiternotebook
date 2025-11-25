#!/bin/bash

# Full pipeline để train Faster R-CNN với dataset KHÔNG CÓ AUGMENTATION FILES
# Chỉ sử dụng các file gốc (filter out augmented files)

echo "======================================================================"
echo "FASTER R-CNN TRAINING PIPELINE - NO AUGMENTATION DATASET"
echo "======================================================================"
echo "⚠️  This pipeline will ONLY use original images (no augmented files)"
echo "======================================================================"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Configuration
DATA_ROOT="datasets/mvtec_balanced_noaug/bottle"
MODEL_TYPE="resnet50"
EPOCHS=30
BATCH_SIZE=16
LR=0.001
SAVE_DIR="runs/fasterrcnn_balanced"

# Step 1: Prepare dataset (NO AUGMENTATION)
echo ""
echo "======================================================================"
echo "STEP 1: Preparing Dataset (NO AUGMENTATION)"
echo "======================================================================"
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  Dataset not found at $DATA_ROOT"
    echo "Preparing balanced dataset (original files only)..."
    
    if [ ! -d "datasets/mvtec/bottle" ]; then
        echo "❌ Source dataset not found at datasets/mvtec/bottle"
        echo "Please prepare the dataset first!"
        exit 1
    fi
    
    python prepare_balanced_dataset_noaug.py \
        --source datasets/mvtec/bottle \
        --output "$DATA_ROOT" \
        --train_ratio 0.7 \
        --seed 42
    
    if [ $? -ne 0 ]; then
        echo "❌ Dataset preparation failed!"
        exit 1
    fi
    echo "✓ Dataset prepared successfully (ORIGINAL FILES ONLY)"
else
    echo "✓ Dataset found at $DATA_ROOT"
    echo "⚠️  This dataset contains ONLY original images (no augmentation)"
fi

# Step 2: Train model with NO AUGMENTATION dataset
echo ""
echo "======================================================================"
echo "STEP 2: Training Model (NO AUGMENTATION DATASET)"
echo "======================================================================"
echo "⚠️  Only using ORIGINAL images (filtering out augmented files)"
python train_balanced_noaug.py \
    --data_root "$DATA_ROOT" \
    --model_type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --save_dir "$SAVE_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 3: Find the latest model (no augmentation dataset)
echo ""
echo "======================================================================"
echo "STEP 3: Finding Latest Model"
echo "======================================================================"

# Get latest no-augmentation dataset model directory
LATEST_MODEL=$(ls -td "$SAVE_DIR"/balanced_"$MODEL_TYPE"_noaug_dataset_* 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No model found!"
    exit 1
fi

echo "Using model: $LATEST_MODEL"
BEST_MODEL="$LATEST_MODEL/best_acc.pth"

if [ ! -f "$BEST_MODEL" ]; then
    echo "❌ Best model checkpoint not found: $BEST_MODEL"
    exit 1
fi

echo "✓ Model found: $BEST_MODEL"

# Step 4: Run inference
echo ""
echo "======================================================================"
echo "STEP 4: Running Inference"
echo "======================================================================"
python inference_fasterrcnn.py \
    --checkpoint "$BEST_MODEL" \
    --data_root "$DATA_ROOT" \
    --model_type "$MODEL_TYPE" \
    --device cpu \
    --output_dir "$LATEST_MODEL/inference_val"

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi

echo "✓ Inference completed"

# Step 5: Summary
echo ""
echo "======================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo "Model directory: $LATEST_MODEL"
echo "Metrics file: $LATEST_MODEL/inference_val/metrics.json"
echo ""
echo "⚠️  This model was trained with ORIGINAL images only (no augmentation files)"
echo ""
echo "To view results:"
echo "  cat $LATEST_MODEL/inference_val/metrics.json"
echo ""
echo "To compare with other models:"
echo "  python compare_fasterrcnn_augmentation.py"
echo "======================================================================"

