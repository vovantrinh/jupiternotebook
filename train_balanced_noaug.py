"""
Script training với balanced dataset - CHỈ SỬ DỤNG FILE GỐC (KHÔNG AUGMENTATION)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
from datetime import datetime
import json

from dataset_loader_balanced_noaug import create_dataloaders
from model import get_model


def train_model(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tạo thư mục lưu kết quả
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'balanced_{args.model_type}_noaug_dataset_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    print(f"⚠️  Training with ORIGINAL images only (no augmentation files)")
    
    # Load data - CHỈ LOAD FILE GỐC
    print(f"\nLoading dataset from {args.data_root}...")
    print("⚠️  Only loading ORIGINAL images (filtering out augmented files)")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Load model
    print(f"\nInitializing model: {args.model_type}")
    model = get_model(
        model_type=args.model_type,
        num_classes=2,
        pretrained=True
    ).to(device)
    
    # Loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }
    
    best_acc = 0.0
    best_auc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("⚠️  NO DATA AUGMENTATION - Using original images only")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs[:, 1].cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        try:
            val_auc = roc_auc_score(all_labels, all_probabilities)
        except:
            val_auc = 0.0
        
        # Update history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_auc'].append(float(val_auc))
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        
        # Save best models
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, val_auc, save_dir, 'best_acc.pth')
            print(f"✓ Saved best accuracy model: {val_acc:.2f}%")
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_acc, val_auc, save_dir, 'best_auc.pth')
            print(f"✓ Saved best AUC model: {val_auc:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, val_acc, val_auc, save_dir, 'last.pth')
    
    # Save history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Validation Set")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=['Good', 'Defect'],
        digits=4
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, save_dir)
    
    print(f"\n✓ Training completed!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Results saved to: {save_dir}")


def save_checkpoint(model, optimizer, epoch, acc, auc, save_dir, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
        'auc': auc
    }
    torch.save(checkpoint, os.path.join(save_dir, filename))


def plot_training_curves(history, save_dir):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # AUC
    axes[2].plot(epochs, history['val_auc'], 'g-', label='Val AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Validation AUC')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(labels, predictions, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Good', 'Defect'],
        yticklabels=['Good', 'Defect']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Faster RCNN with Balanced Dataset - NO AUGMENTATION FILES')
    
    parser.add_argument('--data_root', type=str, default='datasets/mvtec_balanced_noaug/bottle',
                       help='Dataset directory')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet50', 'fpn'], help='Model type')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='runs/fasterrcnn_balanced',
                       help='Save directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Configuration - NO AUGMENTATION DATASET")
    print("="*60)
    print("⚠️  This script will ONLY use original images (no augmented files)")
    print("="*60)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*60)
    
    train_model(args)

