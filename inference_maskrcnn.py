#!/usr/bin/env python3
"""Inference and evaluation for Mask R-CNN model."""

import sys
import json
import argparse
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO


def load_model(model_path, num_classes=2, device='cpu'):
    model = maskrcnn_resnet50_fpn(num_classes=num_classes + 1)
    
    # Replace classifiers
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes + 1
    )
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes + 1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def evaluate_on_dataset(model, dataset_dir, split='val', device='cpu', conf_threshold=0.5, output_dir=None):
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split
    ann_file = split_dir / 'annotations.json'
    
    coco = COCO(str(ann_file))
    image_ids = list(coco.imgs.keys())
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {split} set")
    print(f"{'='*60}")
    print(f"Images: {len(image_ids)}")
    
    tp_defect, fp_defect, tn_good, fn_defect = 0, 0, 0, 0
    total_detections, total_confidence = 0, 0.0
    
    for img_id in tqdm(image_ids, desc="Inference"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = split_dir / 'images' / img_info['file_name']
        
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_tensor)
        
        pred = predictions[0]
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        mask = scores >= conf_threshold
        labels = labels[mask]
        scores = scores[mask]
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_labels = [ann['category_id'] for ann in anns]
        
        has_defect_gt = 2 in gt_labels
        has_defect_pred = 2 in labels
        
        if predictions:
            total_detections += len(labels)
            total_confidence += sum(scores)
        
        if has_defect_gt and has_defect_pred:
            tp_defect += 1
        elif has_defect_gt and not has_defect_pred:
            fn_defect += 1
        elif not has_defect_gt and has_defect_pred:
            fp_defect += 1
        else:
            tn_good += 1
    
    total = tp_defect + fp_defect + tn_good + fn_defect
    accuracy = (tp_defect + tn_good) / total if total > 0 else 0
    precision = tp_defect / (tp_defect + fp_defect) if (tp_defect + fp_defect) > 0 else 0
    recall = tp_defect / (tp_defect + fn_defect) if (tp_defect + fn_defect) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    defect_detection_rate = recall
    composite_score = (accuracy * 0.3 + precision * 0.2 + recall * 0.3 + f1 * 0.2)
    avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
    
    # Convert numpy types to Python native types for JSON serialization
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'defect_detection_rate': float(defect_detection_rate),
        'composite_score': float(composite_score),
        'confidence': float(avg_confidence),
        'tp_defect': int(tp_defect),
        'fp_defect': int(fp_defect),
        'tn_good': int(tn_good),
        'fn_defect': int(fn_defect),
        'total_images': int(total)
    }
    
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
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Results saved to: {output_dir}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    
    device = args.device or ('mps' if torch.backends.mps.is_available() else 
                            'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.model, args.num_classes, device)
    evaluate_on_dataset(model, args.dataset_dir, args.split, device, args.conf, args.output_dir)
