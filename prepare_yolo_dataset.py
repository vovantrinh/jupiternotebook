#!/usr/bin/env python3
"""
Script để chuẩn bị dataset YOLO format từ MVTec dataset
- Convert ground truth masks thành bounding boxes
- Tạo YOLO format labels (class_id x_center y_center width height - normalized)
- Tạo cấu trúc: train/images, train/labels, val/images, val/labels
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image


def mask_to_bbox(mask_path):
    """
    Convert mask thành bounding box (x_min, y_min, x_max, y_max)
    Returns None nếu không có object trong mask
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Sử dụng boundingRect cho tất cả contours và lấy bbox lớn nhất
    # Cách này an toàn hơn và xử lý được mọi trường hợp
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
    
    return (x_min, y_min, x_max, y_max)


def bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bbox (x_min, y_min, x_max, y_max) sang YOLO format
    (class_id x_center y_center width height) - normalized
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Normalize
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Đảm bảo trong khoảng [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return (x_center, y_center, width, height)


def prepare_yolo_dataset(
    source_dir='datasets/mvtec/bottle',
    output_dir='datasets/yolo_mvtec_bottle',
    train_ratio=0.8,
    seed=42
):
    """
    Chuẩn bị dataset YOLO format từ MVTec dataset
    
    Args:
        source_dir: Thư mục chứa dataset gốc
        output_dir: Thư mục output (sẽ tạo với prefix yolo_)
        train_ratio: Tỷ lệ chia train (0.8 = 80% train, 20% val)
        seed: Random seed
    """
    random.seed(seed)
    
    print("="*60)
    print("CHUẨN BỊ DATASET YOLO FORMAT")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Train ratio: {train_ratio}")
    
    # Tạo thư mục output
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Class mapping: 0 = good, 1 = defect
    class_names = ['good', 'defect']
    
    # 1. Xử lý GOOD images (không có bounding box)
    print("\n1. Processing GOOD images...")
    good_images = []
    
    # Từ train/good
    train_good_source = os.path.join(source_dir, 'train', 'good')
    if os.path.exists(train_good_source):
        for img in os.listdir(train_good_source):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                good_images.append(os.path.join(train_good_source, img))
    
    # Từ test/good
    test_good_source = os.path.join(source_dir, 'test', 'good')
    if os.path.exists(test_good_source):
        for img in os.listdir(test_good_source):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                good_images.append(os.path.join(test_good_source, img))
    
    print(f"Found {len(good_images)} good images")
    
    # Shuffle và split
    random.shuffle(good_images)
    train_split = int(len(good_images) * train_ratio)
    train_good_images = good_images[:train_split]
    val_good_images = good_images[train_split:]
    
    # Copy good images và tạo empty labels (không có defect)
    print(f"Processing {len(train_good_images)} train good images...")
    for img_path in tqdm(train_good_images):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Copy image
        img_name = os.path.basename(img_path)
        # Convert to JPG for YOLO
        img_name_jpg = os.path.splitext(img_name)[0] + '.jpg'
        dst_img_path = os.path.join(train_images_dir, img_name_jpg)
        img.convert('RGB').save(dst_img_path, 'JPEG', quality=95)
        
        # Tạo empty label file (không có bounding box)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)
        with open(label_path, 'w') as f:
            pass  # Empty file - no defects
    
    print(f"Processing {len(val_good_images)} val good images...")
    for img_path in tqdm(val_good_images):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Copy image
        img_name = os.path.basename(img_path)
        img_name_jpg = os.path.splitext(img_name)[0] + '.jpg'
        dst_img_path = os.path.join(val_images_dir, img_name_jpg)
        img.convert('RGB').save(dst_img_path, 'JPEG', quality=95)
        
        # Tạo empty label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(val_labels_dir, label_name)
        with open(label_path, 'w') as f:
            pass  # Empty file
    
    # 2. Xử lý DEFECT images (có bounding box từ ground truth masks)
    print("\n2. Processing DEFECT images...")
    defect_data = []  # (img_path, mask_path)
    
    test_dir = os.path.join(source_dir, 'test')
    ground_truth_dir = os.path.join(source_dir, 'ground_truth')
    
    defect_categories = [d for d in os.listdir(test_dir) 
                        if os.path.isdir(os.path.join(test_dir, d)) and d != 'good']
    
    for category in defect_categories:
        category_img_dir = os.path.join(test_dir, category)
        category_mask_dir = os.path.join(ground_truth_dir, category)
        
        if not os.path.exists(category_mask_dir):
            print(f"⚠️  Warning: No ground truth masks for {category}, skipping...")
            continue
        
        for img_name in os.listdir(category_img_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_img_dir, img_name)
                # Tìm mask tương ứng
                mask_name = os.path.splitext(img_name)[0] + '_mask.png'
                mask_path = os.path.join(category_mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    defect_data.append((img_path, mask_path))
                else:
                    print(f"⚠️  Warning: No mask found for {img_name}")
    
    print(f"Found {len(defect_data)} defect images with masks")
    
    # Shuffle và split
    random.shuffle(defect_data)
    train_split = int(len(defect_data) * train_ratio)
    train_defect_data = defect_data[:train_split]
    val_defect_data = defect_data[train_split:]
    
    # Process train defect images
    print(f"Processing {len(train_defect_data)} train defect images...")
    for img_path, mask_path in tqdm(train_defect_data):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Copy image
        img_name = os.path.basename(img_path)
        img_name_jpg = os.path.splitext(img_name)[0] + '.jpg'
        dst_img_path = os.path.join(train_images_dir, img_name_jpg)
        img.convert('RGB').save(dst_img_path, 'JPEG', quality=95)
        
        # Convert mask to bounding box
        bbox = mask_to_bbox(mask_path)
        if bbox is None:
            print(f"⚠️  Warning: No bbox found for {img_name}, creating empty label")
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(train_labels_dir, label_name)
            with open(label_path, 'w') as f:
                pass
        else:
            # Convert to YOLO format
            x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)
            
            # Write label file (class_id = 1 for defect)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(train_labels_dir, label_name)
            with open(label_path, 'w') as f:
                f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Process val defect images
    print(f"Processing {len(val_defect_data)} val defect images...")
    for img_path, mask_path in tqdm(val_defect_data):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Copy image
        img_name = os.path.basename(img_path)
        img_name_jpg = os.path.splitext(img_name)[0] + '.jpg'
        dst_img_path = os.path.join(val_images_dir, img_name_jpg)
        img.convert('RGB').save(dst_img_path, 'JPEG', quality=95)
        
        # Convert mask to bounding box
        bbox = mask_to_bbox(mask_path)
        if bbox is None:
            print(f"⚠️  Warning: No bbox found for {img_name}, creating empty label")
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(val_labels_dir, label_name)
            with open(label_path, 'w') as f:
                pass
        else:
            # Convert to YOLO format
            x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)
            
            # Write label file (class_id = 1 for defect)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(val_labels_dir, label_name)
            with open(label_path, 'w') as f:
                f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"TRAIN:")
    print(f"  Good images:   {len(train_good_images)}")
    print(f"  Defect images: {len(train_defect_data)}")
    print(f"  Total:         {len(train_good_images) + len(train_defect_data)} images")
    print(f"\nVALIDATION:")
    print(f"  Good images:   {len(val_good_images)}")
    print(f"  Defect images: {len(val_defect_data)}")
    print(f"  Total:         {len(val_good_images) + len(val_defect_data)} images")
    print(f"\nClasses: {class_names}")
    print("="*60)
    # Tạo file YAML config
    yaml_path = os.path.join(os.path.dirname(output_dir), f"yolo_{os.path.basename(output_dir)}.yaml")
    yaml_content = f"""# YOLO Dataset Configuration for MVTec
# Path: {yaml_path}

path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

# Classes
names:
  0: good
  1: defect

# Number of classes
nc: 2
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Dataset YOLO format prepared successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    
    return output_dir, class_names, yaml_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from MVTec')
    parser.add_argument('--source', type=str, default='datasets/mvtec/bottle',
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, default='datasets/yolo_mvtec_bottle',
                       help='Output dataset directory (with yolo prefix)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train ratio (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    try:
        output_dir, class_names, yaml_path = prepare_yolo_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        print(f"\n✓ Success! Dataset ready at: {output_dir}")
        print(f"✓ YAML config: {yaml_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

