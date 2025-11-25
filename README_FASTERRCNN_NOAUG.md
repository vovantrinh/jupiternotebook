# Faster R-CNN Training Pipeline - No Augmentation

Pipeline tự động để train Faster R-CNN không augmentation và so sánh với model có augmentation.

## Cấu trúc Pipeline

1. **Check/Prepare Dataset**: Kiểm tra dataset balanced, nếu chưa có thì tự động prepare
2. **Training**: Train model với `--no_augmentation` flag
3. **Inference**: Tự động chạy inference trên validation set
4. **Comparison**: So sánh với model có augmentation (nếu có)

## Cách sử dụng

### Option 1: Bash Script (Recommended)

```bash
# Activate venv
source venv/bin/activate

# Chạy pipeline
./run_fasterrcnn_noaug_pipeline.sh
```

### Option 2: Python Script

```bash
# Activate venv
source venv/bin/activate

# Chạy pipeline
python run_fasterrcnn_noaug_pipeline.py
```

## Configuration

Các tham số mặc định trong pipeline:

- **Dataset**: `datasets/mvtec_balanced/bottle`
- **Model Type**: `resnet50`
- **Epochs**: `30`
- **Batch Size**: `16`
- **Learning Rate**: `0.001`
- **Save Directory**: `runs/fasterrcnn_balanced`

## Output

Sau khi chạy pipeline, bạn sẽ có:

1. **Model Directory**: `runs/fasterrcnn_balanced/balanced_resnet50_noaug_YYYYMMDD_HHMMSS/`
   - `best_acc.pth`: Best accuracy checkpoint
   - `best_auc.pth`: Best AUC checkpoint
   - `last.pth`: Final checkpoint
   - `training_curves.png`: Training curves
   - `confusion_matrix.png`: Confusion matrix
   - `roc_curve.png`: ROC curve

2. **Inference Results**: `runs/fasterrcnn_balanced/balanced_resnet50_noaug_YYYYMMDD_HHMMSS/inference_val/`
   - `metrics.json`: Evaluation metrics (Accuracy, Precision, Recall, F1-Score, etc.)

3. **Comparison Results** (nếu có model với augmentation):
   - `fasterrcnn_augmentation_comparison.csv`: Comparison table
   - `fasterrcnn_augmentation_comparison_curves.png`: Training curves comparison

## Manual Training (nếu muốn tùy chỉnh)

```bash
source venv/bin/activate

# Train không augmentation
python train_balanced.py \
    --data_root datasets/mvtec_balanced/bottle \
    --model_type resnet50 \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --save_dir runs/fasterrcnn_balanced \
    --no_augmentation

# Inference
python inference_fasterrcnn.py \
    --checkpoint runs/fasterrcnn_balanced/balanced_resnet50_noaug_YYYYMMDD_HHMMSS/best_acc.pth \
    --data_root datasets/mvtec_balanced/bottle \
    --model_type resnet50 \
    --device cpu
```

## So sánh với Model có Augmentation

Sau khi train cả 2 models (có và không augmentation), chạy:

```bash
python compare_fasterrcnn_augmentation.py
```

Script sẽ tự động:
- Tìm models có/không augmentation
- So sánh metrics
- Vẽ training curves comparison
- Lưu kết quả ra CSV và PNG

## Metrics được so sánh

- Accuracy
- Precision
- Recall
- F1-Score
- Defect Detection Rate
- Composite Score
- Average Confidence
- Confusion Matrix (TP, FP, TN, FN)

## Notes

- Model không augmentation sẽ có suffix `noaug` trong tên thư mục
- Model có augmentation sẽ có suffix `aug` trong tên thư mục
- Pipeline tự động tìm model mới nhất để inference
- Nếu chưa có model với augmentation, pipeline vẫn chạy nhưng bỏ qua bước comparison

