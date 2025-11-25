#!/usr/bin/env python3
"""
Inference và đánh giá YOLO model cho MVTec dataset
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json


def read_yolo_label(label_path):
    """Đọc YOLO label file"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return boxes


def yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format sang xyxy"""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return (x_min, y_min, x_max, y_max)


def calculate_iou(box1, box2):
    """Tính IoU giữa 2 boxes (xyxy format)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def evaluate_on_dataset(model, dataset_dir, split='val', conf_threshold=0.25, iou_threshold=0.5):
    """
    Đánh giá model trên dataset
    
    Returns:
        metrics: dict với các metrics
    """
    images_dir = os.path.join(dataset_dir, split, 'images')
    labels_dir = os.path.join(dataset_dir, split, 'labels')
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    images = [f for f in os.listdir(images_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nEvaluating on {len(images)} images from {split} set...")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Statistics
    total_images = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_images = 0
    
    # Per class statistics
    class_stats = {
        0: {'tp': 0, 'fp': 0, 'fn': 0},  # good
        1: {'tp': 0, 'fp': 0, 'fn': 0}   # defect
    }
    
    results_list = []
    
    for img_name in tqdm(images):
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        # Đọc ground truth
        gt_boxes = read_yolo_label(label_path)
        
        # Load image để lấy kích thước
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Predict
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        # Parse predictions
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append({
                    'class_id': cls,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
        
        # Match predictions với ground truth
        matched_gt = set()
        matched_pred = set()
        matched_any_defect = False
        
        # Match defect boxes (class 1)
        for i, pred in enumerate(pred_boxes):
            if pred['class_id'] == 1:  # defect
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(gt_boxes):
                    if gt['class_id'] == 1 and j not in matched_gt:
                        gt_box = yolo_to_xyxy(
                            gt['x_center'], gt['y_center'],
                            gt['width'], gt['height'],
                            img_width, img_height
                        )
                        iou = calculate_iou(pred['box'], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    class_stats[1]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
                else:
                    false_positives += 1
                    class_stats[1]['fp'] += 1
        
        # Count false negatives (unmatched ground truth defects)
        for j, gt in enumerate(gt_boxes):
            if gt['class_id'] == 1 and j not in matched_gt:
                false_negatives += 1
                class_stats[1]['fn'] += 1
        
        # Count false positives (unmatched predictions)
        for i, pred in enumerate(pred_boxes):
            if pred['class_id'] == 1 and i not in matched_pred:
                false_positives += 1
                class_stats[1]['fp'] += 1
        
        gt_defect_count = sum(1 for b in gt_boxes if b['class_id'] == 1)
        has_defect_pred = any(p['class_id'] == 1 for p in pred_boxes)
        
        # Good images (no defects) - nếu không có prediction defect thì đúng
        if gt_defect_count == 0:  # Good image
            if not has_defect_pred:
                true_positives += 1
                class_stats[0]['tp'] += 1
                correct_images += 1
            else:
                false_positives += len([p for p in pred_boxes if p['class_id'] == 1])
                class_stats[0]['fp'] += len([p for p in pred_boxes if p['class_id'] == 1])
        else:
            if len(matched_gt) > 0:
                correct_images += 1
        
        total_images += 1
        
        # Store result
        results_list.append({
            'image': img_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'gt_defects': sum(1 for b in gt_boxes if b['class_id'] == 1),
            'pred_defects': sum(1 for b in pred_boxes if b['class_id'] == 1)
        })
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per class metrics
    class_metrics = {}
    for class_id, stats in class_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        class_metrics[class_id] = {
            'precision': prec,
            'recall': rec,
            'f1': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    accuracy = correct_images / total_images if total_images > 0 else 0
    composite_score = (precision + recall) / 2.0 if (precision is not None and recall is not None) else 0
    
    defect_detection_rate = class_metrics.get(1, {}).get('recall', 0)
    
    metrics = {
        'total_images': total_images,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'score': composite_score,
        'defect_detection_rate': defect_detection_rate,
        'class_metrics': class_metrics,
        'results': results_list
    }
    
    return metrics


def inference_on_images(model, images_dir, output_dir, conf_threshold=0.25, save_images=True):
    """Inference trên thư mục ảnh và lưu kết quả"""
    os.makedirs(output_dir, exist_ok=True)
    
    images = [f for f in os.listdir(images_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nRunning inference on {len(images)} images...")
    print(f"Output directory: {output_dir}")
    
    results = []
    
    for img_name in tqdm(images):
        img_path = os.path.join(images_dir, img_name)
        
        # Predict
        pred_results = model.predict(img_path, conf=conf_threshold, save=False, verbose=False)
        
        # Parse results
        if len(pred_results) > 0 and pred_results[0].boxes is not None:
            boxes_info = []
            for box in pred_results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = 'good' if cls == 0 else 'defect'
                boxes_info.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].cpu().tolist()
                })
            
            results.append({
                'image': img_name,
                'predictions': boxes_info
            })
            
            # Save annotated image
            if save_images:
                annotated_img = pred_results[0].plot()
                output_path = os.path.join(output_dir, f"pred_{img_name}")
                cv2.imwrite(output_path, annotated_img)
        else:
            results.append({
                'image': img_name,
                'predictions': []
            })
    
    # Save results to JSON
    results_json_path = os.path.join(output_dir, 'results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary text
    summary_path = os.path.join(output_dir, 'results.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("INFERENCE RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['image']}\n")
            if result['predictions']:
                for pred in result['predictions']:
                    f.write(f"  - {pred['class']}: {pred['confidence']:.4f} "
                           f"bbox: {pred['bbox']}\n")
            else:
                f.write("  - No detections\n")
            f.write("\n")
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - Images: {len([r for r in results if r['predictions']])} with detections")
    print(f"  - JSON: {results_json_path}")
    print(f"  - Summary: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference and evaluate YOLO model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--dataset', type=str, default='datasets/yolo_mvtec_bottle',
                       help='Dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--images_dir', type=str, default=None,
                       help='Custom images directory for inference (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for inference results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for evaluation')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not run inference')
    parser.add_argument('--inference_only', action='store_true',
                       help='Only run inference, do not evaluate')
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    print("="*60)
    print("YOLO INFERENCE & EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    
    model = YOLO(args.model)
    
    # Evaluate on dataset
    if not args.inference_only:
        if os.path.exists(args.dataset):
            print(f"\nEvaluating on dataset: {args.dataset}")
            metrics = evaluate_on_dataset(
                model, args.dataset, split=args.split,
                conf_threshold=args.conf, iou_threshold=args.iou
            )
            
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Total images: {metrics['total_images']}")
            print(f"\nOverall Metrics:")
            print(f"  Accuracy : {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall   : {metrics['recall']:.4f}")
            print(f"  F1 Score : {metrics['f1']:.4f}")
            print(f"  Defect Detection Rate: {metrics['defect_detection_rate']:.4f}")
            print(f"\n  TP: {metrics['true_positives']}")
            print(f"  FP: {metrics['false_positives']}")
            print(f"  FN: {metrics['false_negatives']}")
            
            print("\nOVERALL METRICS")
            print("="*60)
            print(f"Accuracy : {metrics['accuracy']*100:.2f}%")
            print(f"Precision: {metrics['precision']*100:.2f}%")
            print(f"Recall   : {metrics['recall']*100:.2f}%")
            print(f"F1-Score : {metrics['f1']*100:.2f}%")
            print(f"Composite: {metrics['score']*100:.2f}%")
            print(f"Defect detection rate: {metrics['defect_detection_rate']*100:.2f}%")
            
            print(f"\nPer-Class Metrics:")
            for class_id, class_metric in metrics['class_metrics'].items():
                class_name = 'good' if class_id == 0 else 'defect'
                print(f"\n  {class_name.upper()}:")
                print(f"    Precision: {class_metric['precision']:.4f}")
                print(f"    Recall: {class_metric['recall']:.4f}")
                print(f"    F1: {class_metric['f1']:.4f}")
                print(f"    TP: {class_metric['tp']}, FP: {class_metric['fp']}, FN: {class_metric['fn']}")
            
            # Save metrics
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                metrics_path = os.path.join(args.output_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✓ Metrics saved to {metrics_path}")
        else:
            print(f"⚠️  Dataset not found: {args.dataset}, skipping evaluation")
    
    # Inference on custom images
    if not args.eval_only:
        if args.images_dir:
            if not os.path.exists(args.images_dir):
                raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
            
            output_dir = args.output_dir or f"runs/yolo_mvtec/inference_{args.split}"
            inference_on_images(
                model, args.images_dir, output_dir,
                conf_threshold=args.conf, save_images=True
            )
        elif args.dataset:
            # Inference on dataset split
            images_dir = os.path.join(args.dataset, args.split, 'images')
            if os.path.exists(images_dir):
                output_dir = args.output_dir or f"runs/yolo_mvtec/inference_{args.split}"
                inference_on_images(
                    model, images_dir, output_dir,
                    conf_threshold=args.conf, save_images=True
                )
    
    print("\n✓ Completed!")


if __name__ == '__main__':
    main()


