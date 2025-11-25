#!/usr/bin/env python3
"""
Inference script for Faster R-CNN classification model
Evaluates on validation dataset and saves metrics in JSON format compatible with other models
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import get_model


def load_model(checkpoint_path: str, model_type: str = 'resnet50', device: str = 'cpu'):
    """Load trained Faster R-CNN model from checkpoint"""
    device = torch.device(device)
    model = get_model(model_type=model_type, num_classes=2, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"✓ Model type: {model_type}")
    print(f"✓ Device: {device}")
    
    return model, device


def get_image_transform(img_size: int = 224):
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_val_images(data_root: str) -> List[Tuple[str, int]]:
    """
    Load validation images with labels
    Returns: List of (image_path, label) tuples where label: 0=good, 1=defect
    """
    val_dir = Path(data_root) / 'val'
    images = []
    
    # Load good images (label 0)
    good_dir = val_dir / 'good'
    if good_dir.exists():
        for img_name in sorted(good_dir.glob('*.png')) + sorted(good_dir.glob('*.jpg')):
            images.append((str(img_name), 0))
    
    # Load defect images (label 1)
    defect_dir = val_dir / 'defect'
    if defect_dir.exists():
        for img_name in sorted(defect_dir.glob('*.png')) + sorted(defect_dir.glob('*.jpg')):
            images.append((str(img_name), 1))
    
    print(f"Loaded {len(images)} validation images")
    num_good = sum(1 for _, label in images if label == 0)
    num_defect = sum(1 for _, label in images if label == 1)
    print(f"  - Good: {num_good}, Defect: {num_defect}")
    
    return images


def evaluate_on_dataset(
    model,
    data_root: str,
    device: str = 'cpu',
    img_size: int = 224,
    output_dir: str = None
):
    """
    Evaluate model on validation dataset and calculate metrics
    Returns metrics dictionary compatible with other models
    """
    model.eval()
    transform = get_image_transform(img_size)
    
    # Load validation images
    val_images = load_val_images(data_root)
    
    if len(val_images) == 0:
        raise ValueError(f"No validation images found in {data_root}/val")
    
    # Initialize counters
    tp_defect = 0  # True Positive: predicted defect, actual defect
    fp_defect = 0  # False Positive: predicted defect, actual good
    tn_good = 0    # True Negative: predicted good, actual good
    fn_defect = 0  # False Negative: predicted good, actual defect
    
    total_confidence = 0.0
    total_predictions = 0
    
    print(f"\n{'='*60}")
    print(f"Evaluating on validation set")
    print(f"{'='*60}")
    print(f"Total images: {len(val_images)}")
    
    # Process each image
    with torch.no_grad():
        for img_path, gt_label in tqdm(val_images, desc="Inference"):
            # Load and preprocess image
            try:
                image = Image.open(img_path).convert('RGB')
                tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                pred_label = int(torch.argmax(probs))
                confidence = float(probs[pred_label].item())
                
                total_confidence += confidence
                total_predictions += 1
                
                # Update confusion matrix
                # gt_label: 0=good, 1=defect
                # pred_label: 0=good, 1=defect
                
                if gt_label == 1:  # Actual defect
                    if pred_label == 1:  # Predicted defect
                        tp_defect += 1
                    else:  # Predicted good
                        fn_defect += 1
                else:  # Actual good (gt_label == 0)
                    if pred_label == 1:  # Predicted defect
                        fp_defect += 1
                    else:  # Predicted good
                        tn_good += 1
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate metrics
    total = tp_defect + fp_defect + tn_good + fn_defect
    
    accuracy = (tp_defect + tn_good) / total if total > 0 else 0
    precision = tp_defect / (tp_defect + fp_defect) if (tp_defect + fp_defect) > 0 else 0
    recall = tp_defect / (tp_defect + fn_defect) if (tp_defect + fn_defect) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    defect_detection_rate = recall  # Same as recall for defect class
    composite_score = (accuracy * 0.3 + precision * 0.2 + recall * 0.3 + f1_score * 0.2)
    avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
    
    # Create metrics dictionary (compatible with other models)
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'defect_detection_rate': float(defect_detection_rate),
        'composite_score': float(composite_score),
        'confidence': float(avg_confidence),
        'tp_defect': int(tp_defect),
        'fp_defect': int(fp_defect),
        'tn_good': int(tn_good),
        'fn_defect': int(fn_defect),
        'total_images': int(total)
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"Accuracy : {accuracy*100:>6.2f}%")
    print(f"Precision: {precision*100:>6.2f}%")
    print(f"Recall   : {recall*100:>6.2f}%")
    print(f"F1-Score : {f1_score*100:>6.2f}%")
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
    
    # Save metrics
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Inference for Faster R-CNN classification model')
    parser.add_argument('--checkpoint', required=True, 
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_root', required=True,
                       help='Path to dataset root (should contain val/good and val/defect)')
    parser.add_argument('--model_type', default='resnet50', choices=['resnet50', 'fpn'],
                       help='Model architecture type')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for inference')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for inference')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save metrics.json (default: runs/fasterrcnn_balanced/inference_val)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        checkpoint_name = Path(args.checkpoint).parent.name
        args.output_dir = f'runs/fasterrcnn_balanced/{checkpoint_name}/inference_val'
    
    # Load model
    model, device = load_model(args.checkpoint, args.model_type, args.device)
    
    # Evaluate
    metrics = evaluate_on_dataset(
        model=model,
        data_root=args.data_root,
        device=device,
        img_size=args.img_size,
        output_dir=args.output_dir
    )
    
    print("\n✓ Inference completed!")


if __name__ == '__main__':
    main()

