#!/usr/bin/env python3
"""
Prepare MVTec dataset for SSD training in COCO format.

This script converts the MVTec Anomaly Detection dataset into COCO format
suitable for training SSD object detection models. It generates bounding boxes
from ground truth masks and creates train/val splits.

Output structure:
    datasets/ssd_mvtec_<category>/
        ├── train/
        │   ├── images/
        │   └── annotations.json (COCO format)
        └── val/
            ├── images/
            └── annotations.json (COCO format)
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm


def mask_to_bbox(mask_path):
    """
    Convert binary mask to bounding box in COCO format.
    Returns: [x, y, width, height] or None if no object found
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get bounding box for all contours
    x_coords = []
    y_coords = []
    
    for contour in contours:
        if len(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
    
    if len(x_coords) == 0:
        return None
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # COCO format: [x, y, width, height]
    width = x_max - x_min
    height = y_max - y_min
    
    return [float(x_min), float(y_min), float(width), float(height)]


def create_coco_annotation(image_id, annotation_id, bbox, category_id, area=None):
    """
    Create a COCO format annotation dictionary.
    """
    if area is None:
        area = bbox[2] * bbox[3]  # width * height
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "segmentation": []  # Can be empty for bbox-only detection
    }


def prepare_ssd_dataset(
    mvtec_root,
    category,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    """
    Convert MVTec dataset to COCO format for SSD training.
    
    Args:
        mvtec_root: Path to MVTec dataset root
        category: Category name (e.g., 'bottle')
        output_dir: Output directory for SSD dataset
        train_ratio: Ratio of train split
        seed: Random seed for reproducibility
    
    Returns:
        output_dir: Path to created dataset
        class_names: List of class names
    """
    np.random.seed(seed)
    
    mvtec_root = Path(mvtec_root)
    category_path = mvtec_root / category
    output_dir = Path(output_dir)
    
    if not category_path.exists():
        raise ValueError(f"Category path not found: {category_path}")
    
    print(f"\n{'='*60}")
    print(f"Preparing SSD dataset for MVTec/{category}")
    print(f"{'='*60}")
    print(f"Source: {category_path}")
    print(f"Output: {output_dir}")
    
    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    
    # Class definitions (COCO format requires category IDs starting from 1)
    # Class 0: background (implicit)
    # Class 1: good
    # Class 2: defect
    categories = [
        {"id": 1, "name": "good", "supercategory": "object"},
        {"id": 2, "name": "defect", "supercategory": "object"}
    ]
    class_names = ["good", "defect"]
    
    # Initialize COCO format dictionaries
    coco_train = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    coco_val = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_id = 1
    annotation_id = 1
    
    # Process test images (has ground truth labels)
    test_dir = category_path / 'test'
    gt_dir = category_path / 'ground_truth'
    
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    # Get all defect types
    defect_types = [d for d in test_dir.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(defect_types)} defect types:")
    for dt in defect_types:
        print(f"  - {dt.name}")
    
    # Collect all samples
    all_samples = []
    
    for defect_type in defect_types:
        images = sorted(defect_type.glob('*.png'))
        
        for img_path in images:
            # Determine if it's good or defect
            is_good = (defect_type.name == 'good')
            
            # Get mask path if it's a defect
            mask_path = None
            if not is_good and gt_dir.exists():
                mask_path = gt_dir / defect_type.name / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    print(f"Warning: Mask not found for {img_path.name}, skipping...")
                    continue
            
            all_samples.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'defect_type': defect_type.name,
                'is_good': is_good
            })
    
    # Shuffle and split
    np.random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples)} images")
    print(f"  Val:   {len(val_samples)} images")
    
    # Process train samples
    print("\nProcessing training set...")
    for sample in tqdm(train_samples):
        img_path = sample['image_path']
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping...")
            continue
        
        height, width = img.shape[:2]
        
        # Copy image
        new_img_name = f"{image_id:06d}.jpg"
        new_img_path = output_dir / 'train' / 'images' / new_img_name
        cv2.imwrite(str(new_img_path), img)
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": new_img_name,
            "width": width,
            "height": height,
            "original_file": str(img_path.relative_to(mvtec_root)),
            "defect_type": sample['defect_type']
        }
        coco_train['images'].append(image_info)
        
        # Add annotation
        if sample['is_good']:
            # For good images, create a full-image bounding box with class 'good'
            bbox = [0.0, 0.0, float(width), float(height)]
            ann = create_coco_annotation(image_id, annotation_id, bbox, category_id=1)
            coco_train['annotations'].append(ann)
            annotation_id += 1
        else:
            # For defect images, extract bbox from mask
            mask_path = sample['mask_path']
            bbox = mask_to_bbox(mask_path)
            
            if bbox is not None:
                ann = create_coco_annotation(image_id, annotation_id, bbox, category_id=2)
                coco_train['annotations'].append(ann)
                annotation_id += 1
            else:
                print(f"Warning: No bbox found for {img_path.name}")
        
        image_id += 1
    
    # Process val samples
    print("\nProcessing validation set...")
    for sample in tqdm(val_samples):
        img_path = sample['image_path']
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping...")
            continue
        
        height, width = img.shape[:2]
        
        # Copy image
        new_img_name = f"{image_id:06d}.jpg"
        new_img_path = output_dir / 'val' / 'images' / new_img_name
        cv2.imwrite(str(new_img_path), img)
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": new_img_name,
            "width": width,
            "height": height,
            "original_file": str(img_path.relative_to(mvtec_root)),
            "defect_type": sample['defect_type']
        }
        coco_val['images'].append(image_info)
        
        # Add annotation
        if sample['is_good']:
            bbox = [0.0, 0.0, float(width), float(height)]
            ann = create_coco_annotation(image_id, annotation_id, bbox, category_id=1)
            coco_val['annotations'].append(ann)
            annotation_id += 1
        else:
            mask_path = sample['mask_path']
            bbox = mask_to_bbox(mask_path)
            
            if bbox is not None:
                ann = create_coco_annotation(image_id, annotation_id, bbox, category_id=2)
                coco_val['annotations'].append(ann)
                annotation_id += 1
            else:
                print(f"Warning: No bbox found for {img_path.name}")
        
        image_id += 1
    
    # Save COCO annotation files
    print("\nSaving COCO annotation files...")
    with open(output_dir / 'train' / 'annotations.json', 'w') as f:
        json.dump(coco_train, f, indent=2)
    
    with open(output_dir / 'val' / 'annotations.json', 'w') as f:
        json.dump(coco_val, f, indent=2)
    
    # Save dataset info
    dataset_info = {
        "category": category,
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_images": len(coco_train['images']),
        "val_images": len(coco_val['images']),
        "train_annotations": len(coco_train['annotations']),
        "val_annotations": len(coco_val['annotations']),
        "created_at": datetime.now().isoformat()
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Dataset preparation completed!")
    print(f"{'='*60}")
    print(f"Train images: {len(coco_train['images'])}")
    print(f"Train annotations: {len(coco_train['annotations'])}")
    print(f"Val images: {len(coco_val['images'])}")
    print(f"Val annotations: {len(coco_val['annotations'])}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Output directory: {output_dir}")
    
    return output_dir, class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare MVTec dataset for SSD training')
    parser.add_argument('--mvtec_root', type=str, default='datasets/mvtec',
                        help='Path to MVTec dataset root')
    parser.add_argument('--category', type=str, default='bottle',
                        help='MVTec category to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: datasets/ssd_mvtec_<category>)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio for train split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f'datasets/ssd_mvtec_{args.category}'
    
    try:
        prepare_ssd_dataset(
            mvtec_root=args.mvtec_root,
            category=args.category,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

