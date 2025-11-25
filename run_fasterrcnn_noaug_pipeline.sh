#!/bin/bash

# Full pipeline để train Faster R-CNN không augmentation
# Bao gồm: prepare dataset -> train -> inference -> comparison

echo "======================================================================"
echo "FASTER R-CNN TRAINING PIPELINE - NO AUGMENTATION"
echo "======================================================================"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Configuration
DATA_ROOT="datasets/mvtec_balanced/bottle"
MODEL_TYPE="resnet50"
EPOCHS=30
BATCH_SIZE=16
LR=0.001
SAVE_DIR="runs/fasterrcnn_balanced"

# Step 1: Check if dataset exists
echo ""
echo "======================================================================"
echo "STEP 1: Checking Dataset"
echo "======================================================================"
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  Dataset not found at $DATA_ROOT"
    echo "Preparing balanced dataset..."
    
    if [ ! -d "datasets/mvtec/bottle" ]; then
        echo "❌ Source dataset not found at datasets/mvtec/bottle"
        echo "Please prepare the dataset first!"
        exit 1
    fi
    
    python prepare_balanced_dataset.py \
        --source datasets/mvtec/bottle \
        --output datasets/mvtec_balanced/bottle \
        --train_ratio 0.7 \
        --seed 42
    
    if [ $? -ne 0 ]; then
        echo "❌ Dataset preparation failed!"
        exit 1
    fi
    echo "✓ Dataset prepared successfully"
else
    echo "✓ Dataset found at $DATA_ROOT"
fi

# Step 2: Train model without augmentation
echo ""
echo "======================================================================"
echo "STEP 2: Training Model (NO AUGMENTATION)"
echo "======================================================================"
python train_balanced.py \
    --data_root "$DATA_ROOT" \
    --model_type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --save_dir "$SAVE_DIR" \
    --no_augmentation

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 3: Find the latest model (no augmentation)
echo ""
echo "======================================================================"
echo "STEP 3: Finding Latest Model"
echo "======================================================================"

# Get latest no-augmentation model directory
LATEST_MODEL=$(ls -td "$SAVE_DIR"/balanced_"$MODEL_TYPE"_noaug_* 2>/dev/null | head -1)

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

# Step 5: Compare with augmented model (if exists)
echo ""
echo "======================================================================"
echo "STEP 5: Comparison with Augmented Model"
echo "======================================================================"

# Find latest augmented model
LATEST_AUG_MODEL=$(ls -td "$SAVE_DIR"/balanced_"$MODEL_TYPE"_aug_* 2>/dev/null | head -1)

if [ -n "$LATEST_AUG_MODEL" ]; then
    echo "Found augmented model: $LATEST_AUG_MODEL"
    
    # Run inference on augmented model if metrics don't exist
    if [ ! -f "$LATEST_AUG_MODEL/inference_val/metrics.json" ]; then
        echo "Running inference on augmented model..."
        python inference_fasterrcnn.py \
            --checkpoint "$LATEST_AUG_MODEL/best_acc.pth" \
            --data_root "$DATA_ROOT" \
            --model_type "$MODEL_TYPE" \
            --device cpu \
            --output_dir "$LATEST_AUG_MODEL/inference_val"
    fi
    
    # Run comparison script
    if [ -f "compare_fasterrcnn_augmentation.py" ]; then
        echo "Running comparison..."
        python compare_fasterrcnn_augmentation.py
        echo "✓ Comparison completed"
    else
        echo "⚠️  Comparison script not found, skipping comparison"
    fi
else
    echo "⚠️  No augmented model found for comparison"
    echo "   Train with augmentation first to enable comparison"
fi

# Step 6: Summary
echo ""
echo "======================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo "Model directory: $LATEST_MODEL"
echo "Metrics file: $LATEST_MODEL/inference_val/metrics.json"
echo ""
echo "To view results:"
echo "  cat $LATEST_MODEL/inference_val/metrics.json"
echo ""
echo "To compare with augmented model:"
echo "  python compare_fasterrcnn_augmentation.py"
echo "======================================================================"

