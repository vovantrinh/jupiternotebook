#!/usr/bin/env python3
"""
Augment Mask R-CNN dataset with COCO format annotations including segmentation masks.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def transform_segmentation(segmentation, transform_func, orig_shape):
    """Transform segmentation polygons based on image transformation."""
    new_segmentation = []
    for polygon in segmentation:
        # Reshape to (N, 2)
        points = np.array(polygon).reshape(-1, 2)
        # Apply transformation
        transformed_points = transform_func(points, orig_shape)
        # Flatten back
        new_polygon = transformed_points.flatten().tolist()
        if len(new_polygon) >= 6:  # Need at least 3 points
            new_segmentation.append(new_polygon)
    return new_segmentation


def flip_h_points(points, shape):
    """Flip points horizontally."""
    h, w = shape[:2]
    points_new = points.copy()
    points_new[:, 0] = w - points[:, 0]
    return points_new


def rotate_90_points(points, shape):
    """Rotate points 90° clockwise."""
    h, w = shape[:2]
    points_new = np.zeros_like(points)
    points_new[:, 0] = points[:, 1]
    points_new[:, 1] = w - points[:, 0]
    return points_new


def rotate_180_points(points, shape):
    """Rotate points 180°."""
    h, w = shape[:2]
    points_new = np.zeros_like(points)
    points_new[:, 0] = w - points[:, 0]
    points_new[:, 1] = h - points[:, 1]
    return points_new


def rotate_270_points(points, shape):
    """Rotate points 270° clockwise."""
    h, w = shape[:2]
    points_new = np.zeros_like(points)
    points_new[:, 0] = h - points[:, 1]
    points_new[:, 1] = points[:, 0]
    return points_new


def augment_maskrcnn_dataset(dataset_dir, output_dir=None, augmentation_factor=3, seed=42):
    """Augment Mask R-CNN dataset with segmentation masks."""
    np.random.seed(seed)
    
    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        output_dir = Path(str(dataset_dir) + '_augmented')
    else:
        output_dir = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Augmenting Mask R-CNN Dataset")
    print(f"{'='*60}")
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    
    augmentations = [
        ('flip_h', cv2.flip, {'flipCode': 1}, flip_h_points),
        ('rotate_90', cv2.rotate, {'rotateCode': cv2.ROTATE_90_CLOCKWISE}, rotate_90_points),
        ('rotate_180', cv2.rotate, {'rotateCode': cv2.ROTATE_180}, rotate_180_points),
        ('rotate_270', cv2.rotate, {'rotateCode': cv2.ROTATE_90_COUNTERCLOCKWISE}, rotate_270_points),
        ('blur', lambda img, **k: cv2.GaussianBlur(img, (5, 5), 0), {}, None),
        ('bright', lambda img, **k: cv2.convertScaleAbs(img, alpha=1.3, beta=0), {}, None),
    ]
    
    stats = {'train': {}, 'val': {}}
    
    for split in ['train', 'val']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        
        ann_file = split_dir / 'annotations.json'
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        output_split_dir = output_dir / split
        output_images_dir = output_split_dir / 'images'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        new_coco_data = {
            'images': [],
            'annotations': [],
            'categories': coco_data['categories']
        }
        
        next_image_id = 1
        next_ann_id = 1
        
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        print(f"\nProcessing {split} set...")
        for img_info in tqdm(coco_data['images']):
            img_path = split_dir / 'images' / img_info['file_name']
            if not img_path.exists():
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            anns = img_to_anns.get(img_info['id'], [])
            
            # Original
            orig_img_name = f"{next_image_id:06d}_orig.jpg"
            cv2.imwrite(str(output_images_dir / orig_img_name), image)
            
            new_img_info = deepcopy(img_info)
            new_img_info['id'] = next_image_id
            new_img_info['file_name'] = orig_img_name
            new_coco_data['images'].append(new_img_info)
            
            for ann in anns:
                new_ann = deepcopy(ann)
                new_ann['id'] = next_ann_id
                new_ann['image_id'] = next_image_id
                new_coco_data['annotations'].append(new_ann)
                next_ann_id += 1
            
            next_image_id += 1
            
            # Augmentations
            selected_augs = np.random.choice(
                len(augmentations),
                size=min(augmentation_factor, len(augmentations)),
                replace=False
            )
            
            for aug_idx in selected_augs:
                aug_name, aug_func, aug_params, transform_points = augmentations[aug_idx]
                
                aug_image = aug_func(image.copy(), **aug_params)
                aug_img_name = f"{next_image_id:06d}_{aug_name}.jpg"
                cv2.imwrite(str(output_images_dir / aug_img_name), aug_image)
                
                h, w = aug_image.shape[:2]
                new_img_info = {
                    'id': next_image_id,
                    'file_name': aug_img_name,
                    'width': w,
                    'height': h,
                    'augmentation': aug_name
                }
                new_coco_data['images'].append(new_img_info)
                
                for ann in anns:
                    new_ann = deepcopy(ann)
                    new_ann['id'] = next_ann_id
                    new_ann['image_id'] = next_image_id
                    
                    if transform_points:
                        new_seg = transform_segmentation(
                            ann['segmentation'], 
                            transform_points, 
                            image.shape
                        )
                        new_ann['segmentation'] = new_seg
                        
                        # Recalculate bbox from segmentation
                        all_x, all_y = [], []
                        for poly in new_seg:
                            for i in range(0, len(poly), 2):
                                all_x.append(poly[i])
                                all_y.append(poly[i+1])
                        if all_x:
                            new_ann['bbox'] = [
                                float(min(all_x)), float(min(all_y)),
                                float(max(all_x) - min(all_x)),
                                float(max(all_y) - min(all_y))
                            ]
                    
                    new_coco_data['annotations'].append(new_ann)
                    next_ann_id += 1
                
                next_image_id += 1
        
        with open(output_split_dir / 'annotations.json', 'w') as f:
            json.dump(new_coco_data, f, indent=2)
        
        stats[split] = {
            'original': len(coco_data['images']),
            'total': len(new_coco_data['images'])
        }
    
    with open(output_dir / 'augmentation_info.json', 'w') as f:
        json.dump({'augmentation_factor': augmentation_factor, 'stats': stats}, f, indent=2)
    
    print(f"\n✓ Augmentation completed: {output_dir}")
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--augmentation_factor', type=int, default=3)
    args = parser.parse_args()
    
    augment_maskrcnn_dataset(args.dataset_dir, args.output_dir, args.augmentation_factor)
