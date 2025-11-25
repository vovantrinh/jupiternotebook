#!/usr/bin/env python3
"""
Full pipeline để train Faster R-CNN không augmentation
Bao gồm: prepare dataset -> train -> inference -> comparison
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def run_command(cmd, description):
    """Chạy command và hiển thị output"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ {description} failed!")
        sys.exit(1)
    
    print(f"✓ {description} completed")
    return result

def check_dataset(data_root):
    """Kiểm tra dataset có tồn tại không"""
    if not Path(data_root).exists():
        print(f"⚠️  Dataset not found at {data_root}")
        print("Preparing balanced dataset...")
        
        source_dir = "datasets/mvtec/bottle"
        if not Path(source_dir).exists():
            print(f"❌ Source dataset not found at {source_dir}")
            print("Please prepare the dataset first!")
            sys.exit(1)
        
        run_command(
            [
                "python", "prepare_balanced_dataset.py",
                "--source", source_dir,
                "--output", data_root,
                "--train_ratio", "0.7",
                "--seed", "42"
            ],
            "Preparing balanced dataset"
        )
    else:
        print(f"✓ Dataset found at {data_root}")

def train_model(data_root, model_type, epochs, batch_size, lr, save_dir):
    """Train model không augmentation"""
    cmd = [
        "python", "train_balanced.py",
        "--data_root", data_root,
        "--model_type", model_type,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--save_dir", save_dir,
        "--no_augmentation"
    ]
    
    run_command(cmd, "Training model (NO AUGMENTATION)")

def find_latest_model(save_dir, model_type, suffix="noaug"):
    """Tìm model mới nhất"""
    base_path = Path(save_dir)
    pattern = f"balanced_{model_type}_{suffix}_*"
    
    models = sorted(base_path.glob(pattern), key=os.path.getmtime, reverse=True)
    
    if not models:
        print(f"❌ No model found matching pattern: {pattern}")
        sys.exit(1)
    
    latest_model = models[0]
    best_model = latest_model / "best_acc.pth"
    
    if not best_model.exists():
        print(f"❌ Best model checkpoint not found: {best_model}")
        sys.exit(1)
    
    return latest_model, best_model

def run_inference(checkpoint, data_root, model_type, output_dir):
    """Chạy inference"""
    cmd = [
        "python", "inference_fasterrcnn.py",
        "--checkpoint", str(checkpoint),
        "--data_root", data_root,
        "--model_type", model_type,
        "--device", "cpu",
        "--output_dir", str(output_dir)
    ]
    
    run_command(cmd, "Running inference")

def load_metrics(metrics_file):
    """Load metrics từ file JSON"""
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def print_summary(model_dir, metrics_file):
    """In summary của kết quả"""
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Model directory: {model_dir}")
    print(f"Metrics file: {metrics_file}")
    
    metrics = load_metrics(metrics_file)
    if metrics:
        print("\nModel Performance:")
        print(f"  Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
        print(f"  Precision: {metrics.get('precision', 0)*100:.2f}%")
        print(f"  Recall: {metrics.get('recall', 0)*100:.2f}%")
        print(f"  F1-Score: {metrics.get('f1_score', 0)*100:.2f}%")
        print(f"  Composite Score: {metrics.get('composite_score', 0)*100:.2f}%")
    
    print(f"\n{'='*80}")

def compare_with_augmented(save_dir, model_type):
    """So sánh với model có augmentation nếu có"""
    base_path = Path(save_dir)
    aug_pattern = f"balanced_{model_type}_aug_*"
    
    aug_models = sorted(base_path.glob(aug_pattern), key=os.path.getmtime, reverse=True)
    
    if aug_models:
        latest_aug = aug_models[0]
        aug_metrics_file = latest_aug / "inference_val" / "metrics.json"
        
        # Run inference on augmented model if needed
        if not aug_metrics_file.exists():
            print(f"\nRunning inference on augmented model: {latest_aug}")
            run_inference(
                latest_aug / "best_acc.pth",
                "datasets/mvtec_balanced/bottle",
                model_type,
                latest_aug / "inference_val"
            )
        
        # Run comparison
        if Path("compare_fasterrcnn_augmentation.py").exists():
            print(f"\n{'='*80}")
            print("Running Comparison")
            print(f"{'='*80}")
            run_command(
                ["python", "compare_fasterrcnn_augmentation.py"],
                "Comparison with augmented model"
            )
        else:
            print("⚠️  Comparison script not found")
    else:
        print("\n⚠️  No augmented model found for comparison")
        print("   Train with augmentation first to enable comparison")

def main():
    """Main pipeline"""
    print("="*80)
    print("FASTER R-CNN TRAINING PIPELINE - NO AUGMENTATION")
    print("="*80)
    
    # Configuration
    config = {
        'data_root': 'datasets/mvtec_balanced/bottle',
        'model_type': 'resnet50',
        'epochs': 30,
        'batch_size': 16,
        'lr': 0.001,
        'save_dir': 'runs/fasterrcnn_balanced'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Check dataset
    check_dataset(config['data_root'])
    
    # Step 2: Train model
    train_model(
        config['data_root'],
        config['model_type'],
        config['epochs'],
        config['batch_size'],
        config['lr'],
        config['save_dir']
    )
    
    # Step 3: Find latest model
    model_dir, best_model = find_latest_model(
        config['save_dir'],
        config['model_type'],
        suffix='noaug'
    )
    print(f"\n✓ Using model: {model_dir}")
    
    # Step 4: Run inference
    inference_dir = model_dir / "inference_val"
    run_inference(
        best_model,
        config['data_root'],
        config['model_type'],
        inference_dir
    )
    
    # Step 5: Compare with augmented model
    compare_with_augmented(config['save_dir'], config['model_type'])
    
    # Step 6: Print summary
    metrics_file = inference_dir / "metrics.json"
    print_summary(model_dir, metrics_file)
    
    print("\nTo view detailed results:")
    print(f"  cat {metrics_file}")
    print("\nTo compare with augmented model:")
    print("  python compare_fasterrcnn_augmentation.py")

if __name__ == '__main__':
    main()

