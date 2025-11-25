#!/usr/bin/env python3
"""
Train SSD (Single Shot Detector) model on MVTec dataset.

This script trains a SSD300 model using PyTorch and torchvision on the
prepared MVTec dataset in COCO format.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO


class MVTecCOCODataset(Dataset):
    """
    Dataset class for MVTec in COCO format.
    """
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        
        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Get category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image ID
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = self.root_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


def get_device():
    """
    Auto-detect and return the best available device.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU (training will be slow)")
    
    return device


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    return tuple(zip(*batch))


def create_ssd_model(num_classes, pretrained=True):
    """
    Create SSD300 model with custom number of classes.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: SSD300 model
    """
    if pretrained:
        # Load pretrained model
        weights = SSD300_VGG16_Weights.DEFAULT
        model = ssd300_vgg16(weights=weights)
        
        # Get input channels for classification head
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = [4, 6, 6, 6, 4, 4]
        
        # Replace classification head with custom number of classes
        # num_classes + 1 for background
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1  # +1 for background
        )
    else:
        # Create model from scratch
        model = ssd300_vgg16(num_classes=num_classes + 1)
    
    return model


def train_ssd(
    dataset_dir,
    output_dir,
    num_classes=2,
    epochs=100,
    batch_size=8,
    lr=0.001,
    device=None,
    pretrained=True,
    save_every=10
):
    """
    Train SSD model on MVTec dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        output_dir: Output directory for model and logs
        num_classes: Number of classes (excluding background)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on (None for auto-detect)
        pretrained: Use pretrained weights
        save_every: Save checkpoint every N epochs
    
    Returns:
        model: Trained model
        output_dir: Path to output directory
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training SSD Model")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Pretrained: {pretrained}")
    
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    # Create datasets
    train_dataset = MVTecCOCODataset(
        root_dir=dataset_dir / 'train' / 'images',
        annotation_file=dataset_dir / 'train' / 'annotations.json'
    )
    
    val_dataset = MVTecCOCODataset(
        root_dir=dataset_dir / 'val' / 'images',
        annotation_file=dataset_dir / 'val' / 'annotations.json'
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print(f"\nCreating SSD300 model...")
    model = create_ssd_model(num_classes=num_classes, pretrained=pretrained)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, targets in pbar:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Update stats
            train_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                model.train()  # Need to be in train mode to get losses
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
                
                val_loss += losses.item()
                pbar.set_postfix({'loss': losses.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, output_dir / 'best.pt')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  ✓ Saved checkpoint")
        
        print()
    
    # Save final model
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
    }, output_dir / 'last.pt')
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training config
    config = {
        'num_classes': num_classes,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'pretrained': pretrained,
        'device': str(device),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'best_val_loss': best_val_loss,
        'completed_at': datetime.now().isoformat()
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    
    return model, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SSD model on MVTec dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to SSD dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: model_ssd/ssd_mvtec_TIMESTAMP)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (excluding background)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (auto-detect if not specified)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'model_ssd/ssd_mvtec_{timestamp}'
    
    try:
        train_ssd(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            pretrained=args.pretrained,
            save_every=args.save_every
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

