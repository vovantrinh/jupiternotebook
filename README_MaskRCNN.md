# Mask R-CNN MVTec Anomaly Detection Pipeline

Complete pipeline for training and evaluating Mask R-CNN (Instance Segmentation) models on the MVTec Anomaly Detection dataset.

## Overview

This pipeline provides:
- **Dataset Preparation**: Convert MVTec to COCO format with segmentation masks
- **Augmentation**: Augment images and masks together
- **Model Training**: Train Mask R-CNN with PyTorch
- **Evaluation**: Comprehensive metrics and visualizations
- **Inference**: Detect and segment defects in new images

## Features

✅ Automatic GPU detection (MPS/CUDA/CPU)  
✅ COCO format with segmentation masks  
✅ Pretrained Mask R-CNN (ResNet50-FPN backbone)  
✅ Image and mask augmentation  
✅ Instance segmentation for defects  
✅ Comprehensive evaluation metrics  
✅ Good vs Defect classification  

## Requirements

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn pycocotools pillow tqdm
```

Or use the existing `venv`:
```bash
source venv/bin/activate  # Linux/Mac
```

## Quick Start

### Run Complete Pipeline (with Augmentation)

```bash
python run_maskrcnn_pipeline.py --category bottle --epochs 50 --augmentation_factor 3
```

This will:
1. Prepare the dataset in COCO format with masks
2. **Augment images and masks** (3x more data)
3. Train Mask R-CNN for 50 epochs on augmented data
4. Run inference and evaluation

### Run Without Augmentation

```bash
python run_maskrcnn_pipeline.py --category bottle --epochs 50 --skip_augment
```

## Individual Scripts

### 1. Prepare Dataset

Convert MVTec to COCO format with segmentation masks:

```bash
python prepare_maskrcnn_dataset.py \
    --mvtec_root datasets/mvtec \
    --category bottle \
    --output_dir datasets/maskrcnn_mvtec_bottle \
    --train_ratio 0.8
```

**Key differences from SSD:**
- Extracts **segmentation masks** from ground truth
- Converts masks to COCO polygon format
- Full-image instances for "good" class

### 2. Augment Dataset

Augment images and masks together:

```bash
python augment_maskrcnn_dataset.py \
    --dataset_dir datasets/maskrcnn_mvtec_bottle \
    --output_dir datasets/maskrcnn_mvtec_bottle_augmented \
    --augmentation_factor 3
```

**Augmentation techniques:**
- Horizontal flip (with mask transformation)
- Rotate 90°, 180°, 270° (with mask transformation)
- Gaussian blur
- Brightness adjustment

**Important:** Segmentation masks are automatically transformed to match image transformations!

### 3. Train Model

Train Mask R-CNN:

```bash
python train_maskrcnn.py \
    --dataset_dir datasets/maskrcnn_mvtec_bottle_augmented \
    --output_dir model_maskrcnn/maskrcnn_mvtec_20250120 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.005 \
    --device mps
```

**Key Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 4, use 2 if OOM)
- `--lr`: Learning rate (default: 0.005)
- `--device`: Device to train on (auto-detect if not specified)

**Note:** Mask R-CNN is more memory-intensive than SSD. Use smaller batch size if needed.

### 4. Run Inference

Evaluate trained model:

```bash
python inference_maskrcnn.py \
    --model model_maskrcnn/maskrcnn_mvtec_20250120/best.pt \
    --dataset_dir datasets/maskrcnn_mvtec_bottle_augmented \
    --split val \
    --output_dir model_maskrcnn/maskrcnn_mvtec_20250120/inference_val \
    --conf 0.5 \
    --device mps
```

## Model Architecture

This pipeline uses **Mask R-CNN** with ResNet50-FPN backbone:
- **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **Region Proposal Network (RPN)**: Proposes object regions
- **ROI Head**: Classifies and refines bounding boxes
- **Mask Head**: Predicts segmentation masks for each instance
- **Classes**: Background (0), Good (1), Defect (2)
- **Pretrained**: On COCO dataset, fine-tuned on MVTec

## Evaluation Metrics

The pipeline calculates:
- **Accuracy**: Correct classifications / Total images
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Defect Detection Rate**: Recall for defect class
- **Composite Score**: Weighted average of all metrics
- **Confidence**: Average confidence of detections

### Example Output

```
OVERALL METRICS
============================================================
Accuracy : 88.24%
Precision: 92.31%
Recall   : 85.71%
F1-Score : 88.89%
Defect detection rate: 85.71%
Composite: 88.16%
Confidence: 78.45%
============================================================
```

## GPU Support

### Apple Silicon (M1/M2/M3)

Auto-detects MPS backend:

```bash
python run_maskrcnn_pipeline.py --category bottle  # Auto-detects MPS
```

### NVIDIA GPU

```bash
python run_maskrcnn_pipeline.py --category bottle --device cuda
```

### CPU Only

```bash
python run_maskrcnn_pipeline.py --category bottle --device cpu
```

**Note:** Training is significantly slower on CPU. GPU is highly recommended.

## Output Structure

After running the pipeline:

```
model_maskrcnn/maskrcnn_mvtec_20250120_123456/
├── best.pt                 # Best model (lowest train loss)
├── last.pt                 # Final model
├── history.json            # Training history
└── inference_val/
    └── metrics.json        # Evaluation metrics
```

## Comparison: Mask R-CNN vs SSD vs YOLO

| Feature | Mask R-CNN | SSD | YOLO |
|---------|-----------|-----|------|
| Architecture | Two-stage | Single-stage | Single-stage |
| Output | Masks + Boxes | Boxes | Boxes |
| Accuracy | High | Medium | High |
| Speed | Slower | Fast | Very Fast |
| Memory | High | Medium | Medium |
| Best for | Segmentation | Detection | Detection |

**When to use Mask R-CNN:**
- Need pixel-level defect segmentation
- High accuracy is priority
- Have sufficient GPU memory
- Need instance masks for analysis

## Troubleshooting

### Issue: "Out of memory"

**Solution:**
- Reduce batch size: `--batch_size 2` or `--batch_size 1`
- Use smaller images (modify code)
- Use CPU: `--device cpu` (slower)

### Issue: "ModuleNotFoundError: No module named 'pycocotools'"

**Solution:**
```bash
pip install pycocotools
```

### Issue: Slow training

**Solution:**
- Use GPU (MPS or CUDA)
- Reduce epochs: `--epochs 25`
- Skip augmentation: `--skip_augment`

## Best Practices

1. **Start with pretrained weights** (default behavior)
2. **Use small batch size**: 4 for GPU, 2 for limited memory
3. **Monitor training**: Check `history.json` for loss curves
4. **Adjust confidence threshold**: Try 0.3-0.7 based on results
5. **Use augmentation**: Improves generalization significantly

## Advantages of Mask R-CNN

1. **Instance Segmentation**: Get exact defect boundaries
2. **High Accuracy**: Two-stage detection is more precise
3. **Flexible**: Can use masks or just bounding boxes
4. **Pretrained**: Good starting point from COCO weights

## Disadvantages

1. **Memory Intensive**: Requires more GPU memory
2. **Slower**: Takes longer to train and inference
3. **Complex**: More parameters to tune

## References

- Mask R-CNN Paper: https://arxiv.org/abs/1703.06870
- MVTec AD Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- PyTorch Mask R-CNN: https://pytorch.org/vision/stable/models/mask_rcnn.html

## License

This project follows the same license as the main repository.

## Contact

For issues or questions, please open an issue in the repository.

