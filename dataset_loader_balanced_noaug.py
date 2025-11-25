"""
Dataset loader cho balanced dataset (good và defect trong cùng train/val)
CHỈ LOAD CÁC FILE GỐC - KHÔNG CÓ AUGMENTATION
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class BalancedMVTecDatasetNoAug(Dataset):
    """Dataset loader cho MVTec với good và defect - CHỈ LOAD FILE GỐC (không augmentation)"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Đường dẫn đến thư mục dataset balanced
            split: 'train' hoặc 'val'
            transform: Transform để áp dụng cho ảnh
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split)
        
        # Augmentation suffixes để filter ra
        aug_suffixes = ['_blur', '_flip_h', '_flip_v', '_rot_n15', '_rot_p15', 
                       '_bright', '_contrast', '_noise', '_blur', '_flip']
        
        # Load good images (label 0) - chỉ file gốc
        good_dir = os.path.join(split_dir, 'good')
        if os.path.exists(good_dir):
            for img_name in os.listdir(good_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    # Chỉ lấy file gốc (không có augmentation suffix)
                    is_original = True
                    for suffix in aug_suffixes:
                        if suffix in img_name:
                            is_original = False
                            break
                    
                    if is_original:
                        self.images.append(os.path.join(good_dir, img_name))
                        self.labels.append(0)
        
        # Load defect images (label 1) - chỉ file gốc
        defect_dir = os.path.join(split_dir, 'defect')
        if os.path.exists(defect_dir):
            for img_name in os.listdir(defect_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    # Chỉ lấy file gốc (không có augmentation suffix)
                    # File gốc thường có format: 000.png, 001.png, etc. (chỉ số + extension)
                    is_original = True
                    for suffix in aug_suffixes:
                        if suffix in img_name:
                            is_original = False
                            break
                    
                    if is_original:
                        self.images.append(os.path.join(defect_dir, img_name))
                        self.labels.append(1)
        
        num_good = sum(1 for l in self.labels if l == 0)
        num_defect = sum(1 for l in self.labels if l == 1)
        
        print(f"Loaded {len(self.images)} ORIGINAL images (no augmentation) from {split} set")
        print(f"  - Good: {num_good}, Defect: {num_defect}")
        if len(self.images) > 0:
            print(f"  - Balance: {num_good/(num_good+num_defect)*100:.1f}% good, "
                  f"{num_defect/(num_good+num_defect)*100:.1f}% defect")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_train=True, img_size=224):
    """Tạo transforms cho training và validation - KHÔNG CÓ AUGMENTATION"""
    # Cả train và val đều không có augmentation, chỉ resize và normalize
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform


def create_dataloaders(data_root, batch_size=16, img_size=224, num_workers=4):
    """Tạo train và val dataloaders - CHỈ LOAD FILE GỐC"""
    
    train_transform = get_transforms(is_train=True, img_size=img_size)
    val_transform = get_transforms(is_train=False, img_size=img_size)
    
    train_dataset = BalancedMVTecDatasetNoAug(
        root_dir=data_root,
        split='train',
        transform=train_transform
    )
    
    val_dataset = BalancedMVTecDatasetNoAug(
        root_dir=data_root,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loader
    data_root = 'datasets/mvtec_balanced_noaug/bottle'
    train_loader, val_loader = create_dataloaders(data_root, batch_size=8)
    
    print("\n--- Testing DataLoader (No Augmentation) ---")
    for images, labels in train_loader:
        print(f"Train batch - Images shape: {images.shape}, Labels: {labels}")
        break
    
    for images, labels in val_loader:
        print(f"Val batch - Images shape: {images.shape}, Labels: {labels}")
        break

