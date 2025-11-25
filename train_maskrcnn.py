#!/usr/bin/env python3
"""Train Mask R-CNN model on MVTec dataset."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from pycocotools.coco import COCO
import cv2
import numpy as np
from tqdm import tqdm


class MVTecCOCODataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = Path(root_dir)
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root_dir / img_info['file_name']
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id])
        }
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


def train_maskrcnn(dataset_dir, output_dir, num_classes=2, epochs=50, batch_size=4, lr=0.005, device=None):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Mask R-CNN Model")
    print(f"{'='*60}")
    
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    print(f"Device: {device}")
    
    train_dataset = MVTecCOCODataset(
        dataset_dir / 'train' / 'images',
        dataset_dir / 'train' / 'annotations.json'
    )
    
    val_dataset = MVTecCOCODataset(
        dataset_dir / 'val' / 'images',
        dataset_dir / 'val' / 'annotations.json'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    
    # Load pretrained Mask R-CNN
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Replace classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes + 1
    )
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes + 1
    )
    
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        avg_train_loss = train_loss / len(train_loader)
        lr_scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
        
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      output_dir / 'best.pt')
    
    torch.save({'model_state_dict': model.state_dict()}, output_dir / 'last.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training completed: {output_dir}")
    return model, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    
    train_maskrcnn(args.dataset_dir, args.output_dir, args.num_classes, 
                  args.epochs, args.batch_size, args.lr, args.device)
