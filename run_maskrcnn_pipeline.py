#!/usr/bin/env python3
"""Complete Mask R-CNN pipeline for MVTec anomaly detection."""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import torch


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        sys.exit(result.returncode)
    print(f"\n✓ {description} completed")
    return result


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description='Run Mask R-CNN pipeline for MVTec')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--mvtec_root', type=str, default='datasets/mvtec')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--augmentation_factor', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--skip_prepare', action='store_true')
    parser.add_argument('--skip_augment', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')
    args = parser.parse_args()
    
    if args.device is None:
        args.device = get_device()
        print(f"\n✓ Auto-detected device: {args.device}")
    
    venv_python = Path('venv/bin/python')
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable
    
    base_dataset_dir = f"datasets/maskrcnn_mvtec_{args.category}"
    augmented_dataset_dir = f"datasets/maskrcnn_mvtec_{args.category}_augmented"
    dataset_dir = augmented_dataset_dir if not args.skip_augment else base_dataset_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f"model_maskrcnn/maskrcnn_mvtec_{timestamp}"
    
    print(f"\n{'='*60}")
    print("MASK R-CNN PIPELINE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Category:      {args.category}")
    print(f"Training dataset: {dataset_dir}")
    print(f"Model dir:     {model_dir}")
    print(f"Augmentation:  {'Yes' if not args.skip_augment else 'No'}")
    print(f"Epochs:        {args.epochs}")
    print(f"Device:        {args.device}")
    print(f"{'='*60}")
    
    # Step 1: Prepare
    if not args.skip_prepare:
        run_command([
            python_cmd, 'prepare_maskrcnn_dataset.py',
            '--mvtec_root', args.mvtec_root,
            '--category', args.category,
            '--output_dir', base_dataset_dir,
            '--train_ratio', str(args.train_ratio)
        ], "Step 1: Prepare Dataset")
    
    # Step 2: Augment
    if not args.skip_augment:
        run_command([
            python_cmd, 'augment_maskrcnn_dataset.py',
            '--dataset_dir', base_dataset_dir,
            '--output_dir', augmented_dataset_dir,
            '--augmentation_factor', str(args.augmentation_factor)
        ], "Step 2: Augment Dataset")
    
    # Step 3: Train
    if not args.skip_train:
        run_command([
            python_cmd, 'train_maskrcnn.py',
            '--dataset_dir', dataset_dir,
            '--output_dir', model_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--device', args.device
        ], "Step 3: Train Model")
    
    # Step 4: Inference
    if not args.skip_inference:
        inference_dir = f"{model_dir}/inference_val"
        run_command([
            python_cmd, 'inference_maskrcnn.py',
            '--model', f"{model_dir}/best.pt",
            '--dataset_dir', dataset_dir,
            '--split', 'val',
            '--output_dir', inference_dir,
            '--conf', str(args.conf),
            '--device', args.device
        ], "Step 4: Inference")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print(f"\nModel: {model_dir}")
    print(f"Metrics: {model_dir}/inference_val/metrics.json")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        sys.exit(1)
