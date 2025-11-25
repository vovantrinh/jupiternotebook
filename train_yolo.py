#!/usr/bin/env python3
"""
Train YOLO model cho MVTec dataset
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import torch


def get_device(device=''):
    """
    Tự động detect và chọn device tốt nhất
    - Nếu device được chỉ định, sử dụng device đó
    - Nếu không, tự động chọn: MPS (M1 Mac) > CUDA > CPU
    """
    if device:
        return device
    
    # Kiểm tra MPS (MacBook M1/M2/M3)
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) detected and available")
        return 'mps'
    elif torch.cuda.is_available():
        print("✓ CUDA GPU detected and available")
        return '0'
    else:
        print("⚠️  No GPU available, using CPU")
        return 'cpu'


def train_yolo(
    data_yaml,
    model_size='n',  # n, s, m, l, x
    epochs=100,
    imgsz=640,
    batch=16,
    device='',
    project='runs/yolo_mvtec',
    name=None,
    **kwargs
):
    """
    Train YOLO model
    
    Args:
        data_yaml: Đường dẫn đến file YAML config dataset
        model_size: Kích thước model ('n', 's', 'm', 'l', 'x')
        epochs: Số epochs
        imgsz: Kích thước ảnh input
        batch: Batch size
        device: Device ('', 'mps', '0', 'cpu', etc.) - '' = auto detect
        project: Thư mục project
        name: Tên experiment (None = tự động)
        **kwargs: Các tham số khác
    """
    # Auto detect device nếu chưa chỉ định
    selected_device = get_device(device)
    
    print("="*60)
    print("TRAIN YOLO MODEL")
    print("="*60)
    print(f"Data YAML: {data_yaml}")
    print(f"Model size: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {selected_device} {'(auto-detected)' if not device else '(manual)'}")
    print(f"Project: {project}")
    print("="*60)
    
    # Tạo tên experiment nếu chưa có
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"yolo{model_size}_{timestamp}"
    
    # Load model
    model_name = f"yolo11{model_size}.pt"  # Sử dụng YOLO11
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print(f"\nStarting training on device: {selected_device}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=selected_device,
        project=project,
        name=name,
        **kwargs
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best model saved at: {results.save_dir}")
    print(f"Results directory: {os.path.join(project, name)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for MVTec dataset')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (nano, small, medium, large, xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='',
                       help='Device (mps for M1 Mac, 0 for CUDA, cpu, or empty for auto-detect)')
    parser.add_argument('--project', type=str, default='runs/yolo_mvtec',
                       help='Project directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=0.8,
                       help='Warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1,
                       help='Warmup initial bias lr')
    parser.add_argument('--box', type=float, default=7.5,
                       help='Box loss gain')
    parser.add_argument('--cls', type=float, default=0.5,
                       help='Class loss gain')
    parser.add_argument('--dfl', type=float, default=1.5,
                       help='DFL loss gain')
    parser.add_argument('--pose', type=float, default=12.0,
                       help='Pose loss gain')
    parser.add_argument('--kobj', type=float, default=2.0,
                       help='Keypoint obj loss gain')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing (0.0-0.1)')
    parser.add_argument('--nbs', type=int, default=64,
                       help='Nominal batch size')
    parser.add_argument('--hsv_h', type=float, default=0.015,
                       help='Image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                       help='Image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                       help='Image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0,
                       help='Image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1,
                       help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0,
                       help='Image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0,
                       help='Image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                       help='Image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.0,
                       help='Segment copy-paste (probability)')
    
    args = parser.parse_args()
    
    # Kiểm tra file data yaml
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data YAML file not found: {args.data}")
    
    # Tạo thư mục project
    os.makedirs(args.project, exist_ok=True)
    
    # Train
    train_kwargs = {
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'warmup_bias_lr': args.warmup_bias_lr,
        'box': args.box,
        'cls': args.cls,
        'dfl': args.dfl,
        'pose': args.pose,
        'kobj': args.kobj,
        'label_smoothing': args.label_smoothing,
        'nbs': args.nbs,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'copy_paste': args.copy_paste,
    }
    
    train_yolo(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        **train_kwargs
    )


if __name__ == '__main__':
    main()

