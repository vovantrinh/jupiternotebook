"""
Script để chuẩn bị dataset cân bằng KHÔNG CÓ AUGMENTATION
Chỉ copy các file gốc (không có suffix augmentation)
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def prepare_balanced_dataset_noaug(
    source_dir='datasets/mvtec/bottle',
    output_dir='datasets/mvtec_balanced_noaug/bottle',
    train_ratio=0.7,
    seed=42
):
    """
    Chuẩn bị dataset cân bằng từ MVTec dataset - CHỈ FILE GỐC (KHÔNG AUGMENTATION)
    
    Args:
        source_dir: Thư mục chứa dataset gốc
        output_dir: Thư mục output
        train_ratio: Tỷ lệ chia train (0.7 = 70% train, 30% val)
        seed: Random seed
    """
    random.seed(seed)
    
    print(f"Preparing balanced dataset (NO AUGMENTATION)...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Train ratio: {train_ratio}")
    print("⚠️  Only copying ORIGINAL files (no augmented files)")
    
    # Tạo thư mục output
    train_good_dir = os.path.join(output_dir, 'train', 'good')
    train_defect_dir = os.path.join(output_dir, 'train', 'defect')
    val_good_dir = os.path.join(output_dir, 'val', 'good')
    val_defect_dir = os.path.join(output_dir, 'val', 'defect')
    
    for dir_path in [train_good_dir, train_defect_dir, val_good_dir, val_defect_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Augmentation suffixes để filter
    aug_suffixes = ['_blur', '_flip_h', '_flip_v', '_rot_n15', '_rot_p15', 
                   '_bright', '_contrast', '_noise', '_aug', '_rot']
    
    def is_original_file(filename):
        """Kiểm tra xem file có phải là file gốc không"""
        for suffix in aug_suffixes:
            if suffix in filename:
                return False
        return True
    
    # 1. Lấy good images từ train và test - CHỈ FILE GỐC
    print("\n1. Processing GOOD images (original only)...")
    good_images = []
    
    # Từ train/good
    train_good_source = os.path.join(source_dir, 'train', 'good')
    if os.path.exists(train_good_source):
        for img in os.listdir(train_good_source):
            if img.endswith(('.png', '.jpg', '.jpeg')) and is_original_file(img):
                good_images.append(os.path.join(train_good_source, img))
    
    # Từ test/good
    test_good_source = os.path.join(source_dir, 'test', 'good')
    if os.path.exists(test_good_source):
        for img in os.listdir(test_good_source):
            if img.endswith(('.png', '.jpg', '.jpeg')) and is_original_file(img):
                good_images.append(os.path.join(test_good_source, img))
    
    print(f"Found {len(good_images)} original good images")
    
    # Shuffle và split
    random.shuffle(good_images)
    train_split = int(len(good_images) * train_ratio)
    train_good_images = good_images[:train_split]
    val_good_images = good_images[train_split:]
    
    # Copy good images
    print(f"Copying {len(train_good_images)} to train/good...")
    for img_path in tqdm(train_good_images):
        dst_path = os.path.join(train_good_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
        os.chmod(dst_path, 0o644)
    
    print(f"Copying {len(val_good_images)} to val/good...")
    for img_path in tqdm(val_good_images):
        dst_path = os.path.join(val_good_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
        os.chmod(dst_path, 0o644)
    
    # 2. Lấy defect images từ test - CHỈ FILE GỐC
    print("\n2. Processing DEFECT images (original only)...")
    defect_images = []
    
    test_dir = os.path.join(source_dir, 'test')
    defect_categories = [d for d in os.listdir(test_dir) 
                        if os.path.isdir(os.path.join(test_dir, d)) and d != 'good']
    
    for category in defect_categories:
        category_path = os.path.join(test_dir, category)
        for img in os.listdir(category_path):
            if img.endswith(('.png', '.jpg', '.jpeg')) and is_original_file(img):
                defect_images.append(os.path.join(category_path, img))
    
    print(f"Found {len(defect_images)} original defect images from {len(defect_categories)} categories: {defect_categories}")
    
    # Shuffle và split
    random.shuffle(defect_images)
    train_split = int(len(defect_images) * train_ratio)
    train_defect_images = defect_images[:train_split]
    val_defect_images = defect_images[train_split:]
    
    # Copy defect images
    print(f"Copying {len(train_defect_images)} to train/defect...")
    for img_path in tqdm(train_defect_images):
        dst_path = os.path.join(train_defect_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
        os.chmod(dst_path, 0o644)
    
    print(f"Copying {len(val_defect_images)} to val/defect...")
    for img_path in tqdm(val_defect_images):
        dst_path = os.path.join(val_defect_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
        os.chmod(dst_path, 0o644)
    
    # Summary
    print("\n" + "="*60)
    print("DATASET SUMMARY (NO AUGMENTATION)")
    print("="*60)
    print(f"TRAIN:")
    print(f"  Good:   {len(train_good_images)} images (original only)")
    print(f"  Defect: {len(train_defect_images)} images (original only)")
    print(f"  Total:  {len(train_good_images) + len(train_defect_images)} images")
    print(f"\nVALIDATION:")
    print(f"  Good:   {len(val_good_images)} images (original only)")
    print(f"  Defect: {len(val_defect_images)} images (original only)")
    print(f"  Total:  {len(val_good_images) + len(val_defect_images)} images")
    print(f"\nBalance ratio:")
    print(f"  Train - Good:Defect = {len(train_good_images)}:{len(train_defect_images)} "
          f"({len(train_good_images)/(len(train_good_images)+len(train_defect_images))*100:.1f}%:"
          f"{len(train_defect_images)/(len(train_good_images)+len(train_defect_images))*100:.1f}%)")
    print("="*60)
    print(f"\n✓ Dataset prepared successfully (ORIGINAL FILES ONLY)!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare balanced dataset - NO AUGMENTATION')
    parser.add_argument('--source', type=str, default='datasets/mvtec/bottle',
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, default='datasets/mvtec_balanced_noaug/bottle',
                       help='Output dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Train ratio (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    prepare_balanced_dataset_noaug(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

