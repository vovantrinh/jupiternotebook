#!/usr/bin/env python3
"""
Augment SSD dataset with COCO format annotations.

This script performs image augmentation (flip, rotate, blur, brightness)
while updating the bounding box annotations accordingly.
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


def flip_horizontal(image, bbox):
    """
    Flip image horizontally and adjust bbox.
    
    Args:
        image: Image array
        bbox: [x, y, width, height] in COCO format
    
    Returns:
        flipped_image, flipped_bbox
    """
    h, w = image.shape[:2]
    flipped_image = cv2.flip(image, 1)
    
    x, y, width, height = bbox
    # New x position after horizontal flip
    new_x = w - (x + width)
    flipped_bbox = [new_x, y, width, height]
    
    return flipped_image, flipped_bbox


def rotate_90(image, bbox):
    """
    Rotate image 90 degrees clockwise and adjust bbox.
    
    Args:
        image: Image array
        bbox: [x, y, width, height] in COCO format
    
    Returns:
        rotated_image, rotated_bbox
    """
    h, w = image.shape[:2]
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    x, y, width, height = bbox
    # After 90° clockwise rotation:
    # new_x = y
    # new_y = w - (x + width)
    # new_width = height
    # new_height = width
    new_x = y
    new_y = w - (x + width)
    new_width = height
    new_height = width
    
    rotated_bbox = [new_x, new_y, new_width, new_height]
    
    return rotated_image, rotated_bbox


def rotate_180(image, bbox):
    """
    Rotate image 180 degrees and adjust bbox.
    """
    h, w = image.shape[:2]
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    
    x, y, width, height = bbox
    new_x = w - (x + width)
    new_y = h - (y + height)
    rotated_bbox = [new_x, new_y, width, height]
    
    return rotated_image, rotated_bbox


def rotate_270(image, bbox):
    """
    Rotate image 270 degrees clockwise (90° counter-clockwise) and adjust bbox.
    """
    h, w = image.shape[:2]
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    x, y, width, height = bbox
    # After 90° counter-clockwise rotation:
    new_x = h - (y + height)
    new_y = x
    new_width = height
    new_height = width
    
    rotated_bbox = [new_x, new_y, new_width, new_height]
    
    return rotated_image, rotated_bbox


def apply_blur(image, bbox):
    """
    Apply Gaussian blur to image (bbox unchanged).
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred, bbox


def adjust_brightness(image, bbox, factor=1.2):
    """
    Adjust image brightness (bbox unchanged).
    """
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted, bbox


def adjust_contrast(image, bbox, factor=1.2):
    """
    Adjust image contrast (bbox unchanged).
    """
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted, bbox


def augment_ssd_dataset(
    dataset_dir,
    output_dir=None,
    augmentation_factor=3,
    seed=42
):
    """
    Augment SSD dataset with transformations.
    
    Args:
        dataset_dir: Path to original SSD dataset
        output_dir: Output directory (default: dataset_dir + '_augmented')
        augmentation_factor: Number of augmented versions per image
        seed: Random seed
    
    Returns:
        output_dir: Path to augmented dataset
        stats: Augmentation statistics
    """
    np.random.seed(seed)
    
    dataset_dir = Path(dataset_dir)
    
    if output_dir is None:
        output_dir = Path(str(dataset_dir) + '_augmented')
    else:
        output_dir = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Augmenting SSD Dataset")
    print(f"{'='*60}")
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentation factor: {augmentation_factor}")
    
    # Define augmentation functions
    augmentations = [
        ('flip_h', flip_horizontal),
        ('rotate_90', rotate_90),
        ('rotate_180', rotate_180),
        ('rotate_270', rotate_270),
        ('blur', apply_blur),
        ('bright', lambda img, box: adjust_brightness(img, box, 1.3)),
        ('dark', lambda img, box: adjust_brightness(img, box, 0.7)),
        ('contrast', lambda img, box: adjust_contrast(img, box, 1.3)),
    ]
    
    stats = {'train': {}, 'val': {}}
    
    # Process each split
    for split in ['train', 'val']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping...")
            continue
        
        # Load COCO annotations
        ann_file = split_dir / 'annotations.json'
        if not ann_file.exists():
            print(f"Warning: {ann_file} not found, skipping...")
            continue
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create output directories
        output_split_dir = output_dir / split
        output_images_dir = output_split_dir / 'images'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize new COCO data
        new_coco_data = {
            'images': [],
            'annotations': [],
            'categories': coco_data['categories']
        }
        
        # Track IDs
        next_image_id = 1
        next_ann_id = 1
        
        # Statistics
        original_count = len(coco_data['images'])
        augmented_count = 0
        
        # Create image_id to annotations mapping
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        print(f"\nProcessing {split} set...")
        print(f"Original images: {original_count}")
        
        # Process each image
        for img_info in tqdm(coco_data['images'], desc=f"Augmenting {split}"):
            img_path = split_dir / 'images' / img_info['file_name']
            
            if not img_path.exists():
                print(f"Warning: {img_path} not found, skipping...")
                continue
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Cannot read {img_path}, skipping...")
                continue
            
            # Get annotations for this image
            anns = img_to_anns.get(img_info['id'], [])
            
            # Copy original image and annotations
            orig_img_name = f"{next_image_id:06d}_orig.jpg"
            cv2.imwrite(str(output_images_dir / orig_img_name), image)
            
            # Add original image info
            new_img_info = {
                'id': next_image_id,
                'file_name': orig_img_name,
                'width': img_info['width'],
                'height': img_info['height'],
                'original_id': img_info['id'],
                'augmentation': 'original'
            }
            new_coco_data['images'].append(new_img_info)
            
            # Add original annotations
            for ann in anns:
                new_ann = deepcopy(ann)
                new_ann['id'] = next_ann_id
                new_ann['image_id'] = next_image_id
                new_coco_data['annotations'].append(new_ann)
                next_ann_id += 1
            
            next_image_id += 1
            
            # Apply augmentations
            selected_augs = np.random.choice(
                len(augmentations),
                size=min(augmentation_factor, len(augmentations)),
                replace=False
            )
            
            for aug_idx in selected_augs:
                aug_name, aug_func = augmentations[aug_idx]
                
                # Augment image and each annotation
                aug_image = image.copy()
                aug_anns = []
                
                # Apply augmentation to all bounding boxes
                for ann in anns:
                    bbox = ann['bbox']
                    aug_image_temp, aug_bbox = aug_func(image, bbox)
                    
                    # Create augmented annotation
                    new_ann = deepcopy(ann)
                    new_ann['id'] = next_ann_id
                    new_ann['image_id'] = next_image_id
                    new_ann['bbox'] = aug_bbox
                    new_ann['area'] = aug_bbox[2] * aug_bbox[3]
                    aug_anns.append(new_ann)
                    next_ann_id += 1
                
                # Apply augmentation to image
                aug_image, _ = aug_func(image, [0, 0, 0, 0])  # Dummy bbox for image-only augs
                
                # Save augmented image
                aug_img_name = f"{next_image_id:06d}_{aug_name}.jpg"
                cv2.imwrite(str(output_images_dir / aug_img_name), aug_image)
                
                # Add augmented image info
                h, w = aug_image.shape[:2]
                new_img_info = {
                    'id': next_image_id,
                    'file_name': aug_img_name,
                    'width': w,
                    'height': h,
                    'original_id': img_info['id'],
                    'augmentation': aug_name
                }
                new_coco_data['images'].append(new_img_info)
                
                # Add augmented annotations
                new_coco_data['annotations'].extend(aug_anns)
                
                next_image_id += 1
                augmented_count += 1
        
        # Save augmented COCO annotations
        output_ann_file = output_split_dir / 'annotations.json'
        with open(output_ann_file, 'w') as f:
            json.dump(new_coco_data, f, indent=2)
        
        total_count = len(new_coco_data['images'])
        
        print(f"✓ {split} set augmented:")
        print(f"  Original: {original_count} images")
        print(f"  Augmented: {augmented_count} images")
        print(f"  Total: {total_count} images")
        
        stats[split] = {
            'original': original_count,
            'augmented': augmented_count,
            'total': total_count
        }
    
    # Save augmentation info
    aug_info = {
        'augmentation_factor': augmentation_factor,
        'augmentations': [name for name, _ in augmentations],
        'stats': stats,
        'created_at': datetime.now().isoformat()
    }
    
    with open(output_dir / 'augmentation_info.json', 'w') as f:
        json.dump(aug_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Augmentation completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    return output_dir, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment SSD dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to original SSD dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: dataset_dir + "_augmented")')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                        help='Number of augmented versions per image')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    try:
        augment_ssd_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            augmentation_factor=args.augmentation_factor,
            seed=args.seed
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

