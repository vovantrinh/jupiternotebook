#!/usr/bin/env python3
"""
Augment dataset YOLO format
- Augment ảnh và bounding boxes tương ứng
- Lưu vào cùng thư mục với ảnh gốc
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageStat
from tqdm import tqdm
import shutil

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_yolo_label(label_path):
    """Đọc YOLO label file, trả về list of (class_id, x_center, y_center, width, height)"""
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
                boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def write_yolo_label(label_path, boxes):
    """Ghi YOLO label file"""
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height in boxes:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format sang pixel coordinates"""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return (x_min, y_min, x_max, y_max)


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert pixel coordinates sang YOLO format"""
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return (x_center, y_center, width, height)


def rotate_image_and_boxes(img, boxes, degrees):
    """Xoay ảnh và bounding boxes"""
    img_width, img_height = img.size
    
    # Xoay ảnh
    median_color = tuple(int(x) for x in ImageStat.Stat(img).median)
    rotated_img = img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=median_color)
    
    # Xoay bounding boxes
    rotated_boxes = []
    center_x, center_y = img_width / 2, img_height / 2
    angle_rad = np.radians(degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    for class_id, x_center, y_center, width, height in boxes:
        # Convert to pixel coordinates
        x_min, y_min, x_max, y_max = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
        
        # Get 4 corners
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max)
        ]
        
        # Rotate corners
        rotated_corners = []
        for x, y in corners:
            # Translate to origin
            x -= center_x
            y -= center_y
            # Rotate
            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a
            # Translate back
            x_new += center_x
            y_new += center_y
            rotated_corners.append((x_new, y_new))
        
        # Get new bounding box
        x_coords = [c[0] for c in rotated_corners]
        y_coords = [c[1] for c in rotated_corners]
        x_min_new = max(0, min(x_coords))
        y_min_new = max(0, min(y_coords))
        x_max_new = min(img_width, max(x_coords))
        y_max_new = min(img_height, max(y_coords))
        
        # Convert back to YOLO format
        if x_max_new > x_min_new and y_max_new > y_min_new:
            x_center_new, y_center_new, width_new, height_new = bbox_to_yolo(
                int(x_min_new), int(y_min_new), int(x_max_new), int(y_max_new),
                img_width, img_height
            )
            rotated_boxes.append((class_id, x_center_new, y_center_new, width_new, height_new))
    
    return rotated_img, rotated_boxes


def flip_horizontal_image_and_boxes(img, boxes):
    """Lật ngang ảnh và bounding boxes"""
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    flipped_boxes = []
    for class_id, x_center, y_center, width, height in boxes:
        # Flip x_center
        x_center_new = 1.0 - x_center
        flipped_boxes.append((class_id, x_center_new, y_center, width, height))
    
    return flipped_img, flipped_boxes


def flip_vertical_image_and_boxes(img, boxes):
    """Lật dọc ảnh và bounding boxes"""
    flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    flipped_boxes = []
    for class_id, x_center, y_center, width, height in boxes:
        # Flip y_center
        y_center_new = 1.0 - y_center
        flipped_boxes.append((class_id, x_center, y_center_new, width, height))
    
    return flipped_img, flipped_boxes


def blur_image(img):
    """Làm mờ ảnh (không ảnh hưởng đến bounding boxes)"""
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def augment_image_and_label(img_path, label_path, output_img_dir, output_label_dir, suffix, transform_fn):
    """Augment một ảnh và label tương ứng"""
    img_name = os.path.basename(img_path)
    img_stem = os.path.splitext(img_name)[0]
    img_ext = '.jpg'
    
    output_img_path = os.path.join(output_img_dir, f"{img_stem}_{suffix}{img_ext}")
    output_label_path = os.path.join(output_label_dir, f"{img_stem}_{suffix}.txt")
    
    # Skip nếu đã tồn tại
    if os.path.exists(output_img_path):
        return
    
    # Đọc ảnh và labels
    img = Image.open(img_path).convert('RGB')
    boxes = read_yolo_label(label_path)
    
    # Apply transform
    aug_img, aug_boxes = transform_fn(img, boxes)
    
    # Lưu ảnh
    aug_img.save(output_img_path, 'JPEG', quality=95)
    
    # Lưu labels
    write_yolo_label(output_label_path, aug_boxes)


def augment_split(split_dir, split_name='train'):
    """Augment một split (train hoặc val)"""
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"⚠️  Không tìm thấy {images_dir}")
        return
    
    images = [f for f in os.listdir(images_dir) 
              if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    
    if not images:
        print(f"⚠️  Không có ảnh trong {images_dir}")
        return
    
    print(f"\nAugmenting {len(images)} images in {split_name}...")
    
    # Định nghĩa các transforms
    transforms = [
        ("flip_h", lambda img, boxes: flip_horizontal_image_and_boxes(img, boxes)),
        ("flip_v", lambda img, boxes: flip_vertical_image_and_boxes(img, boxes)),
        ("rot_p15", lambda img, boxes: rotate_image_and_boxes(img, boxes, 15)),
        ("rot_n15", lambda img, boxes: rotate_image_and_boxes(img, boxes, -15)),
        ("blur", lambda img, boxes: (blur_image(img), boxes)),  # Blur không đổi boxes
    ]
    
    for img_name in tqdm(images):
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        for suffix, transform_fn in transforms:
            augment_image_and_label(
                img_path, label_path, images_dir, labels_dir, suffix, transform_fn
            )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment YOLO dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/yolo_mvtec_bottle",
        help="Đường dẫn tới dataset YOLO",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Các split cần augment (mặc định chỉ train)",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"❌ Không tìm thấy {data_root}")
    
    print("="*60)
    print("AUGMENT YOLO DATASET")
    print("="*60)
    print(f"Dataset: {data_root}")
    print(f"Splits: {args.splits}")
    
    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"⚠️  Bỏ qua {split_dir}, không tồn tại")
            continue
        augment_split(split_dir, split)
    
    print("\n✓ Hoàn tất augment")


if __name__ == "__main__":
    main()


