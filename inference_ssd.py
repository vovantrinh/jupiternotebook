#!/usr/bin/env python3
"""
Inference and evaluation for SSD model on MVTec dataset.

This script performs inference on test images and calculates evaluation metrics
including accuracy, precision, recall, F1-score, and defect detection rate.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO


def create_ssd_model(num_classes):
    """
    Create SSD300 model with custom number of classes.
    """
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    
    model = ssd300_vgg16(num_classes=num_classes + 1)  # +1 for background
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1
    )
    
    return model


def load_model(model_path, num_classes=2, device='cpu'):
    """
    Load trained SSD model from checkpoint.
    """
    model = create_ssd_model(num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def get_device():
    """
    Auto-detect and return the best available device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU")
    
    return device


def inference_on_image(model, image_path, device, conf_threshold=0.5):
    """
    Run inference on a single image.
    
    Returns:
        predictions: List of detections [{'label': int, 'score': float, 'bbox': [x1, y1, x2, y2]}]
    """
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Extract predictions
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by confidence threshold
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Format results
    results = []
    for box, label, score in zip(boxes, labels, scores):
        results.append({
            'label': int(label),
            'score': float(score),
            'bbox': box.tolist()
        })
    
    return results


def evaluate_on_dataset(
    model,
    dataset_dir,
    split='val',
    device='cpu',
    conf_threshold=0.5,
    output_dir=None,
    save_visualizations=False
):
    """
    Evaluate model on dataset and calculate metrics.
    
    Args:
        model: SSD model
        dataset_dir: Path to dataset directory
        split: 'train' or 'val'
        device: Device to run inference on
        conf_threshold: Confidence threshold for detections
        output_dir: Output directory for results
        save_visualizations: Whether to save visualized predictions
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_visualizations:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
    
    # Load COCO annotations
    ann_file = split_dir / 'annotations.json'
    coco = COCO(str(ann_file))
    
    image_ids = list(coco.imgs.keys())
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {split} set")
    print(f"{'='*60}")
    print(f"Number of images: {len(image_ids)}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Metrics tracking
    all_predictions = []
    
    # Class-wise metrics (class 1: good, class 2: defect)
    # For binary classification: image is "good" if no defect detected, "defect" if defect detected
    tp_defect = 0  # True positive for defect class
    fp_defect = 0  # False positive for defect class
    tn_good = 0    # True negative (correctly predicted as good)
    fn_defect = 0  # False negative (defect missed)
    
    total_detections = 0
    total_confidence = 0.0
    
    # Process each image
    for img_id in tqdm(image_ids, desc="Inference"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = split_dir / 'images' / img_info['file_name']
        
        # Get ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Determine ground truth label (1: good, 2: defect)
        gt_labels = [ann['category_id'] for ann in anns]
        has_defect_gt = 2 in gt_labels
        
        # Run inference
        predictions = inference_on_image(model, img_path, device, conf_threshold)
        
        # Determine predicted label
        pred_labels = [p['label'] for p in predictions]
        has_defect_pred = 2 in pred_labels
        
        # Calculate confidence (average of all detections)
        if predictions:
            avg_conf = np.mean([p['score'] for p in predictions])
            total_detections += len(predictions)
            total_confidence += sum(p['score'] for p in predictions)
        else:
            avg_conf = 0.0
        
        # Update metrics
        if has_defect_gt and has_defect_pred:
            # True positive: defect correctly detected
            tp_defect += 1
        elif has_defect_gt and not has_defect_pred:
            # False negative: defect missed
            fn_defect += 1
        elif not has_defect_gt and has_defect_pred:
            # False positive: good image predicted as defect
            fp_defect += 1
        elif not has_defect_gt and not has_defect_pred:
            # True negative: good image correctly predicted as good
            tn_good += 1
        
        # Store prediction result
        all_predictions.append({
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'gt_has_defect': has_defect_gt,
            'pred_has_defect': has_defect_pred,
            'predictions': predictions,
            'confidence': avg_conf
        })
        
        # Save visualization if requested
        if save_visualizations and output_dir:
            img = cv2.imread(str(img_path))
            
            for pred in predictions:
                bbox = pred['bbox']
                label = pred['label']
                score = pred['score']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw box
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                class_name = 'good' if label == 1 else 'defect'
                text = f"{class_name}: {score:.2f}"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
            
            # Save
            vis_path = vis_dir / img_info['file_name']
            cv2.imwrite(str(vis_path), img)
    
    # Calculate metrics
    total = tp_defect + fp_defect + tn_good + fn_defect
    
    # Accuracy: (TP + TN) / Total
    accuracy = (tp_defect + tn_good) / total if total > 0 else 0
    
    # Precision: TP / (TP + FP)
    precision = tp_defect / (tp_defect + fp_defect) if (tp_defect + fp_defect) > 0 else 0
    
    # Recall: TP / (TP + FN)
    recall = tp_defect / (tp_defect + fn_defect) if (tp_defect + fn_defect) > 0 else 0
    
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Defect Detection Rate (same as Recall for defect class)
    defect_detection_rate = recall
    
    # Composite Score: weighted average of accuracy, precision, recall, f1
    composite_score = (accuracy * 0.3 + precision * 0.2 + recall * 0.3 + f1 * 0.2)
    
    # Average confidence
    avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'defect_detection_rate': defect_detection_rate,
        'composite_score': composite_score,
        'confidence': avg_confidence,
        'tp_defect': tp_defect,
        'fp_defect': fp_defect,
        'tn_good': tn_good,
        'fn_defect': fn_defect,
        'total_images': total,
        'total_detections': total_detections,
        'conf_threshold': conf_threshold
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"Accuracy : {accuracy*100:>6.2f}%")
    print(f"Precision: {precision*100:>6.2f}%")
    print(f"Recall   : {recall*100:>6.2f}%")
    print(f"F1-Score : {f1*100:>6.2f}%")
    print(f"Defect detection rate: {defect_detection_rate*100:>6.2f}%")
    print(f"Composite: {composite_score*100:>6.2f}%")
    print(f"Confidence: {avg_confidence*100:>6.2f}%")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (Defect detected):     {tp_defect}")
    print(f"  FP (False defect):        {fp_defect}")
    print(f"  TN (Good correct):        {tn_good}")
    print(f"  FN (Defect missed):       {fn_defect}")
    print(f"  Total images:             {total}")
    
    # Save results
    if output_dir:
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed predictions
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_dir}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SSD model on MVTec dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to SSD dataset directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (excluding background)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (auto-detect if not specified)')
    parser.add_argument('--save_viz', action='store_true',
                        help='Save visualization images')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, num_classes=args.num_classes, device=device)
    print("✓ Model loaded successfully")
    
    try:
        evaluate_on_dataset(
            model=model,
            dataset_dir=args.dataset_dir,
            split=args.split,
            device=device,
            conf_threshold=args.conf,
            output_dir=args.output_dir,
            save_visualizations=args.save_viz
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

