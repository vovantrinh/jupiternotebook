#!/usr/bin/env python3
"""
Script chính để chạy toàn bộ pipeline YOLO:
1. Chuẩn bị dataset YOLO format
2. Augment dataset
3. Train model
4. Inference và đánh giá
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / "venv/bin/python"


def get_python_executable():
    """
    Return the Python interpreter inside the local venv if available.
    Fallback to current interpreter otherwise.
    """
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


PYTHON_BIN = get_python_executable()


def run_command(cmd, description):
    """Chạy một command và hiển thị kết quả"""
    print("\n" + "="*60)
    print(description)
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        return False
    print(f"\n✓ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run complete YOLO pipeline for MVTec dataset')
    
    # Dataset preparation
    parser.add_argument('--source', type=str, default='datasets/mvtec/bottle',
                       help='Source MVTec dataset directory')
    parser.add_argument('--output', type=str, default='datasets/yolo_mvtec_bottle',
                       help='Output YOLO dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train ratio (0-1)')
    
    # Augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Augment dataset after preparation')
    parser.add_argument('--augment_splits', nargs='+', default=['train'],
                       help='Splits to augment')
    
    # Training
    parser.add_argument('--train', action='store_true',
                       help='Train model after dataset preparation')
    parser.add_argument('--model_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='',
                       help='Device (mps for M1 Mac, 0 for CUDA, cpu, or empty for auto-detect)')
    
    # Inference
    parser.add_argument('--inference', action='store_true',
                       help='Run inference after training')
    parser.add_argument('--eval_split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Split to evaluate')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for inference')
    
    # Pipeline control
    parser.add_argument('--skip_prepare', action='store_true',
                       help='Skip dataset preparation')
    parser.add_argument('--skip_augment', action='store_true',
                       help='Skip augmentation')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference')
    
    # Model path (for inference only)
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (for inference only)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO PIPELINE FOR MVTEC DATASET")
    print("="*60)
    print(f"Source dataset: {args.source}")
    print(f"Output dataset: {args.output}")
    
    # Auto-detect device nếu chưa chỉ định
    if not args.device:
        if torch.backends.mps.is_available():
            print(f"✓ Auto-detected: MPS (Apple Silicon GPU) - MacBook M1/M2/M3")
            args.device = 'mps'
        elif torch.cuda.is_available():
            print(f"✓ Auto-detected: CUDA GPU")
            args.device = '0'
        else:
            print(f"⚠️  Auto-detected: CPU (no GPU available)")
            args.device = 'cpu'
    else:
        print(f"Device: {args.device} (manual)")
    print()
    
    # 1. Prepare dataset
    data_yaml = None
    if not args.skip_prepare:
        cmd = [
            PYTHON_BIN, 'prepare_yolo_dataset.py',
            '--source', args.source,
            '--output', args.output,
            '--train_ratio', str(args.train_ratio)
        ]
        
        if not run_command(cmd, "1. Preparing YOLO dataset"):
            return 1
        
        # Tìm file YAML được tạo
        yaml_path = os.path.join('datasets', f"yolo_{Path(args.output).name}.yaml")
        if os.path.exists(yaml_path):
            data_yaml = yaml_path
        else:
            # Tạo data yaml nếu chưa có
            yaml_content = f"""# YOLO Dataset Configuration
path: {os.path.abspath(args.output)}
train: train/images
val: val/images

# Classes
names:
  0: good
  1: defect

# Number of classes
nc: 2
"""
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            data_yaml = yaml_path
            print(f"✓ Created data YAML: {data_yaml}")
    else:
        print("\n⏭️  Skipping dataset preparation")
        # Tìm file YAML hiện có
        yaml_path = os.path.join('datasets', f"yolo_{Path(args.output).name}.yaml")
        if os.path.exists(yaml_path):
            data_yaml = yaml_path
    
    # 2. Augment dataset
    if args.augment and not args.skip_augment:
        cmd = [
            PYTHON_BIN, 'augment_yolo_dataset.py',
            '--data_root', args.output,
            '--splits'
        ] + args.augment_splits
        
        if not run_command(cmd, "2. Augmenting dataset"):
            return 1
    else:
        print("\n⏭️  Skipping augmentation")
    
    # 3. Train model
    if args.train and not args.skip_train:
        if data_yaml is None:
            data_yaml = os.path.join('datasets', f"yolo_{Path(args.output).name}.yaml")
        
        if not os.path.exists(data_yaml):
            print(f"❌ Data YAML not found: {data_yaml}")
            print("Please run dataset preparation first or create YAML manually")
            return 1
        
        # Tìm model path từ training results
        project_dir = 'runs/yolo_mvtec'
        model_name = f"yolo{args.model_size}_*"
        
        cmd = [
            PYTHON_BIN, 'train_yolo.py',
            '--data', data_yaml,
            '--model', args.model_size,
            '--epochs', str(args.epochs),
            '--batch', str(args.batch),
            '--imgsz', str(args.imgsz),
            '--project', project_dir,
            '--device', args.device  # Luôn truyền device (có thể là auto-detect)
        ]
        
        if not run_command(cmd, "3. Training YOLO model"):
            return 1
        
        # Tìm best model
        import glob
        model_pattern = os.path.join(project_dir, f"yolo{args.model_size}_*", 'weights', 'best.pt')
        model_files = glob.glob(model_pattern)
        if model_files:
            args.model_path = model_files[0]
            print(f"\n✓ Found best model: {args.model_path}")
        else:
            print(f"\n⚠️  Best model not found, using last checkpoint")
            last_pattern = os.path.join(project_dir, f"yolo{args.model_size}_*", 'weights', 'last.pt')
            last_files = glob.glob(last_pattern)
            if last_files:
                args.model_path = last_files[0]
    else:
        print("\n⏭️  Skipping training")
    
    # 4. Inference and evaluation
    if args.inference and not args.skip_inference:
        if args.model_path is None:
            # Tìm model mới nhất
            import glob
            model_pattern = os.path.join('runs/yolo_mvtec', '*', 'weights', 'best.pt')
            model_files = sorted(glob.glob(model_pattern), key=os.path.getmtime, reverse=True)
            if model_files:
                args.model_path = model_files[0]
                print(f"\n✓ Using latest model: {args.model_path}")
            else:
                print("\n❌ No trained model found. Please train first or specify --model_path")
                return 1
        
        if not os.path.exists(args.model_path):
            print(f"\n❌ Model not found: {args.model_path}")
            return 1
        
        output_dir = f"runs/yolo_mvtec/inference_{args.eval_split}"
        
        cmd = [
            PYTHON_BIN, 'inference_yolo.py',
            '--model', args.model_path,
            '--dataset', args.output,
            '--split', args.eval_split,
            '--output_dir', output_dir,
            '--conf', str(args.conf)
        ]
        
        if not run_command(cmd, "4. Inference and evaluation"):
            return 1
    else:
        print("\n⏭️  Skipping inference")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    if args.model_path:
        print(f"\nTrained model: {args.model_path}")
    
    if args.inference and not args.skip_inference:
        print(f"\nInference results: runs/yolo_mvtec/inference_{args.eval_split}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

