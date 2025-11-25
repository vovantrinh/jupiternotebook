"""
Dataset loader cho balanced dataset (good và defect trong cùng train/val)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class BalancedMVTecDataset(Dataset):
    """Dataset loader cho MVTec với good và defect trong cả train và val"""
    
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
        
        # Load good images (label 0)
        good_dir = os.path.join(split_dir, 'good')
        if os.path.exists(good_dir):
            for img_name in os.listdir(good_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(good_dir, img_name))
                    self.labels.append(0)
        
        # Load defect images (label 1)
        defect_dir = os.path.join(split_dir, 'defect')
        if os.path.exists(defect_dir):
            for img_name in os.listdir(defect_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(defect_dir, img_name))
                    self.labels.append(1)
        
        num_good = sum(1 for l in self.labels if l == 0)
        num_defect = sum(1 for l in self.labels if l == 1)
        
        print(f"Loaded {len(self.images)} images from {split} set")
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


def get_transforms(is_train=True, img_size=224, use_augmentation=True):
    """Tạo transforms cho training và validation"""
    if is_train and use_augmentation:
        # Training với augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Training không augmentation hoặc validation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(data_root, batch_size=16, img_size=224, num_workers=4, use_augmentation=True):
    """Tạo train và val dataloaders"""
    
    train_transform = get_transforms(is_train=True, img_size=img_size, use_augmentation=use_augmentation)
    val_transform = get_transforms(is_train=False, img_size=img_size, use_augmentation=False)
    
    train_dataset = BalancedMVTecDataset(
        root_dir=data_root,
        split='train',
        transform=train_transform
    )
    
    val_dataset = BalancedMVTecDataset(
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
    data_root = 'datasets/mvtec_balanced/bottle'
    train_loader, val_loader = create_dataloaders(data_root, batch_size=8)
    
    print("\n--- Testing DataLoader ---")
    for images, labels in train_loader:
        print(f"Train batch - Images shape: {images.shape}, Labels: {labels}")
        break
    
    for images, labels in val_loader:
        print(f"Val batch - Images shape: {images.shape}, Labels: {labels}")
        break



