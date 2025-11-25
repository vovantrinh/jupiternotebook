# SSD MVTec Anomaly Detection Pipeline

Complete pipeline for training and evaluating SSD (Single Shot Detector) models on the MVTec Anomaly Detection dataset.

## Overview

This pipeline provides:
- **Dataset Preparation**: Convert MVTec dataset to COCO format
- **Model Training**: Train SSD300 model with PyTorch
- **Evaluation**: Comprehensive metrics and visualizations
- **Inference**: Detect defects in new images

## Features

✅ Automatic GPU detection (MPS/CUDA/CPU)  
✅ COCO format annotations  
✅ Pretrained model fine-tuning  
✅ Comprehensive evaluation metrics  
✅ Jupyter notebook for analysis  
✅ Good vs Defect classification  
✅ Bounding box detection for defects  

## Requirements

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn pycocotools pillow tqdm
```

Or use the existing `venv`:
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

## Dataset Structure

After preparation, the dataset structure will be:

```
datasets/ssd_mvtec_bottle/
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── annotations.json  # COCO format
├── val/
│   ├── images/
│   │   └── ...
│   └── annotations.json
└── dataset_info.json
```

## Quick Start

### 1. Run Complete Pipeline (with Augmentation)

```bash
python run_ssd_pipeline.py --category bottle --epochs 100 --augmentation_factor 3
```

This will:
1. Prepare the dataset in COCO format
2. **Augment the dataset** (3x more images with transformations)
3. Train SSD model for 100 epochs on augmented data
4. Run inference and evaluation on validation set

### 2. Run Without Augmentation

```bash
python run_ssd_pipeline.py --category bottle --epochs 100 --skip_augment
```

### 3. Custom Training with Augmentation

```bash
python run_ssd_pipeline.py \
    --category bottle \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --augmentation_factor 5 \
    --device mps  # or cuda or cpu
```

### 4. Use Existing Dataset

If you already prepared the dataset:

```bash
python run_ssd_pipeline.py \
    --category bottle \
    --skip_prepare \
    --skip_augment \
    --epochs 100
```

## Individual Scripts

### Prepare Dataset

Convert MVTec to COCO format:

```bash
python prepare_ssd_dataset.py \
    --mvtec_root datasets/mvtec \
    --category bottle \
    --output_dir datasets/ssd_mvtec_bottle \
    --train_ratio 0.8
```

### Augment Dataset

Augment images and annotations with transformations:

```bash
python augment_ssd_dataset.py \
    --dataset_dir datasets/ssd_mvtec_bottle \
    --output_dir datasets/ssd_mvtec_bottle_augmented \
    --augmentation_factor 3
```

**Augmentation techniques:**
- Horizontal flip
- Rotate 90°, 180°, 270°
- Gaussian blur
- Brightness adjustment (bright/dark)
- Contrast adjustment

**Note:** Bounding boxes are automatically adjusted for geometric transformations.

### Train Model

Train SSD model:

```bash
python train_ssd.py \
    --dataset_dir datasets/ssd_mvtec_bottle \
    --output_dir model_ssd/ssd_mvtec_20250120 \
    --num_classes 2 \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.001 \
    --device mps \
    --pretrained
```

**Key Parameters:**
- `--num_classes`: Number of classes excluding background (default: 2 for good/defect)
- `--pretrained`: Use pretrained SSD300 weights (recommended)
- `--device`: Device to train on (auto-detect if not specified)
- `--save_every`: Save checkpoint every N epochs (default: 10)

### Run Inference

Evaluate trained model:

```bash
python inference_ssd.py \
    --model model_ssd/ssd_mvtec_20250120/best.pt \
    --dataset_dir datasets/ssd_mvtec_bottle \
    --split val \
    --output_dir model_ssd/ssd_mvtec_20250120/inference_val \
    --conf 0.5 \
    --num_classes 2 \
    --save_viz
```

**Options:**
- `--conf`: Confidence threshold (default: 0.5)
- `--save_viz`: Save visualization images with bounding boxes
- `--split`: Dataset split to evaluate (train or val)

## Model Architecture

This pipeline uses **SSD300** (Single Shot MultiBox Detector) with VGG16 backbone:
- Input size: 300x300
- Backbone: VGG16 (pretrained on ImageNet)
- Detection head: Multi-scale feature maps
- Classes: Background (0), Good (1), Defect (2)

## Evaluation Metrics

The pipeline calculates:
- **Accuracy**: Correct classifications / Total images
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Defect Detection Rate**: Same as Recall for defect class
- **Composite Score**: Weighted average of all metrics
- **Confidence**: Average confidence of detections

### Confusion Matrix

- **TP (True Positive)**: Defect correctly detected
- **FP (False Positive)**: Good image predicted as defect
- **TN (True Negative)**: Good image correctly predicted as good
- **FN (False Negative)**: Defect missed

## Analysis Notebook

Use Jupyter notebook for detailed analysis:

```bash
jupyter notebook ssd_mvtec_analysis.ipynb
```

The notebook includes:
- **Dataset comparison** (before and after augmentation)
- Dataset statistics with visualizations
- Training configuration
- Training history curves
- Model scorecard with all metrics
- Confusion matrix
- Sample predictions with visualizations

**Remember to update the paths in cell 2:**
```python
MODEL_DIR = BASE_DIR / "model_ssd/ssd_mvtec_YYYYMMDD_HHMMSS"  # Update this
DATASET_DIR = BASE_DIR / "datasets/ssd_mvtec_bottle_augmented"  # Or _bottle if no augmentation
```

## GPU Support

### Apple Silicon (M1/M2/M3)

The pipeline automatically uses MPS backend:

```bash
python run_ssd_pipeline.py --category bottle  # Auto-detects MPS
```

### NVIDIA GPU

For CUDA-enabled GPUs:

```bash
python run_ssd_pipeline.py --category bottle --device cuda
```

### CPU Only

```bash
python run_ssd_pipeline.py --category bottle --device cpu
```

## Output Structure

After running the pipeline:

```
model_ssd/ssd_mvtec_20250120_123456/
├── best.pt                 # Best model (lowest val loss)
├── last.pt                 # Final model
├── checkpoint_epoch_*.pt   # Periodic checkpoints
├── config.json             # Training configuration
├── history.json            # Training history (loss, LR)
└── inference_val/
    ├── metrics.json        # Evaluation metrics
    ├── predictions.json    # Detailed predictions
    └── visualizations/     # Annotated images (if --save_viz)
```

## Example Results

Expected metrics for MVTec bottle:
```
OVERALL METRICS
============================================================
Accuracy : 85-95%
Precision: 80-95%
Recall   : 75-90%
F1-Score : 80-92%
Defect detection rate: 75-90%
Composite: 82-93%
Confidence: 60-85%
============================================================
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pycocotools'"

**Solution:**
```bash
pip install pycocotools
```

### Issue: "RuntimeError: MPS backend not available"

**Solution:**
- Ensure you're on macOS 12.3+
- Update PyTorch: `pip install --upgrade torch torchvision`
- Or use CPU: `--device cpu`

### Issue: "Out of memory"

**Solution:**
- Reduce batch size: `--batch_size 4`
- Use smaller image size (modify code)
- Use CPU: `--device cpu`

### Issue: "No bbox found for image"

**Solution:**
- Check ground truth masks exist in `datasets/mvtec/<category>/ground_truth/`
- Verify mask images are not empty
- Some defects may have very small defects

## Best Practices

1. **Start with pretrained weights**: Use `--pretrained` flag (default)
2. **Use appropriate batch size**: 8-16 for GPU, 2-4 for CPU
3. **Monitor training**: Check `history.json` for loss curves
4. **Adjust confidence threshold**: Try 0.3-0.7 based on precision/recall trade-off
5. **Save checkpoints**: Use `--save_every 10` to save periodic checkpoints

## Comparison with YOLO

| Feature | SSD | YOLO |
|---------|-----|------|
| Architecture | VGG16 + Multi-scale | CSPDarknet + PANet |
| Input size | 300x300 | 640x640 |
| Speed | Fast | Very Fast |
| Accuracy | Good | Very Good |
| Framework | torchvision | Ultralytics |
| Setup | More manual | More automated |

## References

- MVTec Anomaly Detection Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- SSD Paper: https://arxiv.org/abs/1512.02325
- PyTorch SSD: https://pytorch.org/vision/stable/models/ssd.html

## License

This project follows the same license as the main repository.

## Contact

For issues or questions, please open an issue in the repository.

