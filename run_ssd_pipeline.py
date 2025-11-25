#!/usr/bin/env python3
"""
Complete SSD pipeline for MVTec anomaly detection.

This script orchestrates the entire pipeline:
1. Prepare dataset (convert MVTec to COCO format)
2. Train SSD model
3. Run inference and evaluation

Usage:
    python run_ssd_pipeline.py --category bottle --epochs 100
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import torch


def run_command(cmd, description):
    """
    Run a shell command and handle errors.
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✓ {description} completed successfully")
    return result


def get_device():
    """
    Auto-detect and return the best available device.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description='Run complete SSD pipeline for MVTec anomaly detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python run_ssd_pipeline.py --category bottle
  
  # Custom epochs and batch size
  python run_ssd_pipeline.py --category bottle --epochs 50 --batch_size 16
  
  # Specify GPU device
  python run_ssd_pipeline.py --category bottle --device cuda
  
  # Skip dataset preparation (use existing dataset)
  python run_ssd_pipeline.py --category bottle --skip_prepare
        """
    )
    
    # Dataset arguments
    parser.add_argument('--category', type=str, default='bottle',
                        help='MVTec category to process')
    parser.add_argument('--mvtec_root', type=str, default='datasets/mvtec',
                        help='Path to MVTec dataset root')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio for train/val split')
    
    # Augmentation arguments
    parser.add_argument('--augmentation_factor', type=int, default=3,
                        help='Number of augmented versions per image')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (excluding background)')
    
    # Inference arguments
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for inference')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detect if not specified)')
    
    # Pipeline control
    parser.add_argument('--skip_prepare', action='store_true',
                        help='Skip dataset preparation step')
    parser.add_argument('--skip_augment', action='store_true',
                        help='Skip augmentation step')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training step')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference step')
    
    # Output arguments
    parser.add_argument('--save_viz', action='store_true',
                        help='Save visualization images during inference')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_device()
        print(f"\n✓ Auto-detected device: {args.device}")
    
    # Auto-detect Python interpreter (prefer venv if available)
    venv_python = Path('venv/bin/python')
    if venv_python.exists():
        python_cmd = str(venv_python)
        print(f"✓ Using venv Python: {python_cmd}")
    else:
        python_cmd = sys.executable
        print(f"✓ Using system Python: {python_cmd}")
    
    # Define paths
    base_dataset_dir = f"datasets/ssd_mvtec_{args.category}"
    augmented_dataset_dir = f"datasets/ssd_mvtec_{args.category}_augmented"
    
    # Use augmented dataset if not skipping augmentation
    if not args.skip_augment:
        dataset_dir = augmented_dataset_dir
    else:
        dataset_dir = base_dataset_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f"model_ssd/ssd_mvtec_{timestamp}"
    
    print(f"\n{'='*60}")
    print("SSD PIPELINE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Category:      {args.category}")
    print(f"Base dataset:  {base_dataset_dir}")
    print(f"Training dataset: {dataset_dir}")
    print(f"Model dir:     {model_dir}")
    print(f"Augmentation:  {'Yes' if not args.skip_augment else 'No'}")
    if not args.skip_augment:
        print(f"Aug factor:    {args.augmentation_factor}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device:        {args.device}")
    print(f"Confidence:    {args.conf}")
    print(f"{'='*60}")
    
    # Step 1: Prepare dataset
    if not args.skip_prepare:
        cmd = [
            python_cmd,
            'prepare_ssd_dataset.py',
            '--mvtec_root', args.mvtec_root,
            '--category', args.category,
            '--output_dir', base_dataset_dir,
            '--train_ratio', str(args.train_ratio)
        ]
        
        run_command(cmd, "Step 1: Prepare SSD Dataset")
    else:
        print(f"\n⏭️  Skipping dataset preparation (using existing dataset)")
    
    # Step 2: Augment dataset
    if not args.skip_augment:
        cmd = [
            python_cmd,
            'augment_ssd_dataset.py',
            '--dataset_dir', base_dataset_dir,
            '--output_dir', augmented_dataset_dir,
            '--augmentation_factor', str(args.augmentation_factor)
        ]
        
        run_command(cmd, "Step 2: Augment SSD Dataset")
    else:
        print(f"\n⏭️  Skipping augmentation (using original dataset)")
    
    # Step 3: Train model
    if not args.skip_train:
        cmd = [
            python_cmd,
            'train_ssd.py',
            '--dataset_dir', dataset_dir,
            '--output_dir', model_dir,
            '--num_classes', str(args.num_classes),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--device', args.device,
            '--pretrained'
        ]
        
        run_command(cmd, "Step 3: Train SSD Model")
    else:
        print(f"\n⏭️  Skipping training")
        # Need to specify existing model directory
        print("⚠️  Warning: You need to specify an existing model directory for inference")
    
    # Step 4: Run inference and evaluation
    if not args.skip_inference:
        model_path = f"{model_dir}/best.pt"
        inference_dir = f"{model_dir}/inference_val"
        
        cmd = [
            python_cmd,
            'inference_ssd.py',
            '--model', model_path,
            '--dataset_dir', dataset_dir,
            '--split', 'val',
            '--output_dir', inference_dir,
            '--conf', str(args.conf),
            '--num_classes', str(args.num_classes),
            '--device', args.device
        ]
        
        if args.save_viz:
            cmd.append('--save_viz')
        
        run_command(cmd, "Step 4: Inference and Evaluation")
    else:
        print(f"\n⏭️  Skipping inference")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nModel directory: {model_dir}")
    print(f"  - best.pt: Best model checkpoint")
    print(f"  - last.pt: Final model checkpoint")
    print(f"  - config.json: Training configuration")
    print(f"  - history.json: Training history")
    
    if not args.skip_inference:
        print(f"\nInference results: {model_dir}/inference_val")
        print(f"  - metrics.json: Evaluation metrics")
        print(f"  - predictions.json: Detailed predictions")
        if args.save_viz:
            print(f"  - visualizations/: Annotated images")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. View training history:")
    print(f"   cat {model_dir}/history.json")
    print(f"\n2. View evaluation metrics:")
    print(f"   cat {model_dir}/inference_val/metrics.json")
    print(f"\n3. Analyze results in Jupyter Notebook:")
    print(f"   jupyter notebook ssd_mvtec_analysis.ipynb")
    print(f"\n4. Run inference on new images:")
    print(f"   python inference_ssd.py --model {model_dir}/best.pt \\")
    print(f"       --dataset_dir {dataset_dir} --split val")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

