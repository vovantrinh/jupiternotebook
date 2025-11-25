# Hướng Dẫn Training Faster RCNN cho MVTec Dataset

## Mục Lục
1. [Cài Đặt](#cài-đặt)
2. [Cấu Trúc Dataset](#cấu-trúc-dataset)
3. [Training](#training)
4. [Inference](#inference)
5. [Các Tham Số](#các-tham-số)

---

## Cài Đặt

### 1. Kích hoạt môi trường ảo
```bash
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install torch torchvision pillow numpy scikit-learn matplotlib seaborn tqdm
```

Hoặc sử dụng file requirements:
```bash
pip install -r requirements_fasterrcnn.txt
```

---

## Cấu Trúc Dataset

Dataset MVTec phải có cấu trúc như sau:
```
datasets/mvtec/bottle/
├── train/
│   └── good/          # Chỉ có ảnh good trong train
│       ├── 000.png
│       ├── 001.png
│       └── ...
└── test/
    ├── good/          # Ảnh good trong test
    │   ├── 000.png
    │   └── ...
    ├── broken_large/  # Ảnh defect
    ├── broken_small/
    └── contamination/
```

---

## Training

### 1. Training Cơ Bản (ResNet50 Backbone)

```bash
python train_faster_rcnn.py \
    --data_root datasets/mvtec/bottle \
    --model_type resnet50 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001
```

### 2. Training với FPN Backbone

```bash
python train_faster_rcnn.py \
    --data_root datasets/mvtec/bottle \
    --model_type fpn \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001
```

### 3. Training với Cấu Hình Tùy Chỉnh

```bash
python train_faster_rcnn.py \
    --data_root datasets/mvtec/bottle \
    --model_type resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --img_size 224 \
    --lr 0.0001 \
    --optimizer adamw \
    --scheduler cosine \
    --weight_decay 1e-4 \
    --save_dir runs/my_experiment
```

### 4. Training Nhanh (Test)

```bash
python train_faster_rcnn.py \
    --data_root datasets/mvtec/bottle \
    --model_type resnet50 \
    --epochs 10 \
    --batch_size 8
```

---

## Inference

### 1. Test Một Ảnh

```bash
python inference.py \
    --checkpoint runs/fasterrcnn_train/best_acc.pth \
    --model_type resnet50 \
    --image datasets/mvtec/bottle/test/good/000.png \
    --save_dir runs/inference \
    --visualize
```

### 2. Test Toàn Bộ Thư Mục Test

```bash
python inference.py \
    --checkpoint runs/fasterrcnn_train/best_acc.pth \
    --model_type resnet50 \
    --folder datasets/mvtec/bottle/test/good \
    --save_dir runs/inference_good \
    --visualize
```

### 3. Test Ảnh Defect

```bash
python inference.py \
    --checkpoint runs/fasterrcnn_train/best_acc.pth \
    --model_type resnet50 \
    --folder datasets/mvtec/bottle/test/broken_large \
    --save_dir runs/inference_defect
```

---

## Các Tham Số

### Training Parameters

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|-----------------|
| `--data_root` | Đường dẫn đến dataset | `datasets/mvtec/bottle` |
| `--model_type` | Loại model: `resnet50` hoặc `fpn` | `resnet50` |
| `--epochs` | Số epochs | `50` |
| `--batch_size` | Batch size | `16` |
| `--img_size` | Kích thước ảnh input | `224` |
| `--lr` | Learning rate | `0.001` |
| `--optimizer` | Optimizer: `adam`, `sgd`, `adamw` | `adam` |
| `--scheduler` | Scheduler: `plateau`, `cosine`, `none` | `plateau` |
| `--weight_decay` | Weight decay | `1e-4` |
| `--pretrained` | Sử dụng pretrained weights | `True` |
| `--save_dir` | Thư mục lưu kết quả | `runs/fasterrcnn_train` |

### Inference Parameters

| Tham số | Mô tả | Bắt buộc |
|---------|-------|----------|
| `--checkpoint` | Đường dẫn đến model checkpoint | ✓ |
| `--model_type` | Loại model | ✓ |
| `--image` | Đường dẫn đến một ảnh | |
| `--folder` | Đường dẫn đến thư mục ảnh | |
| `--save_dir` | Thư mục lưu kết quả | |
| `--visualize` | Visualize kết quả | |

---

## Kết Quả Training

Sau khi training, các file kết quả sẽ được lưu trong thư mục `runs/fasterrcnn_train/`:

- `best_acc.pth` - Model có accuracy tốt nhất
- `best_auc.pth` - Model có AUC tốt nhất
- `last.pth` - Model ở epoch cuối
- `training_curves.png` - Đồ thị loss, accuracy, AUC
- `confusion_matrix.png` - Ma trận nhầm lẫn
- `roc_curve.png` - Đường cong ROC

---

## Ví Dụ Workflow Đầy Đủ

```bash
# 1. Kích hoạt venv
source venv/bin/activate

# 2. Training model
python train_faster_rcnn.py \
    --data_root datasets/mvtec/bottle \
    --model_type resnet50 \
    --epochs 50 \
    --batch_size 16

# 3. Test một ảnh good
python inference.py \
    --checkpoint runs/fasterrcnn_train/fasterrcnn_resnet50_*/best_acc.pth \
    --model_type resnet50 \
    --image datasets/mvtec/bottle/test/good/000.png \
    --save_dir runs/test_good \
    --visualize

# 4. Test một ảnh defect
python inference.py \
    --checkpoint runs/fasterrcnn_train/fasterrcnn_resnet50_*/best_acc.pth \
    --model_type resnet50 \
    --image datasets/mvtec/bottle/test/broken_large/000.png \
    --save_dir runs/test_defect \
    --visualize

# 5. Test toàn bộ test set
python inference.py \
    --checkpoint runs/fasterrcnn_train/fasterrcnn_resnet50_*/best_acc.pth \
    --model_type resnet50 \
    --folder datasets/mvtec/bottle/test \
    --save_dir runs/test_all \
    --visualize
```

---

## Lưu Ý

1. **GPU**: Model sẽ tự động sử dụng GPU nếu có. Nếu không có GPU, training sẽ chậm hơn trên CPU.

2. **Memory**: Nếu gặp lỗi out of memory, giảm `batch_size` xuống (ví dụ: 8 hoặc 4).

3. **Learning Rate**: 
   - ResNet50: `lr=0.001` thường hoạt động tốt
   - FPN: Có thể thử `lr=0.0001` nếu training không ổn định

4. **Epochs**: 
   - 50 epochs thường đủ cho dataset nhỏ
   - Có thể tăng lên 100-200 epochs cho kết quả tốt hơn

5. **Model Selection**:
   - `resnet50`: Nhanh hơn, ít tham số hơn
   - `fpn`: Chậm hơn nhưng có thể cho kết quả tốt hơn

---

## Troubleshooting

### Lỗi: "RuntimeError: CUDA out of memory"
```bash
# Giảm batch size
python train_faster_rcnn.py --batch_size 8

# Hoặc giảm image size
python train_faster_rcnn.py --img_size 128
```

### Lỗi: "FileNotFoundError: Dataset not found"
```bash
# Kiểm tra đường dẫn dataset
ls datasets/mvtec/bottle/train/good
ls datasets/mvtec/bottle/test
```

### Model không hội tụ
```bash
# Thử learning rate nhỏ hơn
python train_faster_rcnn.py --lr 0.0001 --optimizer adamw
```

---

## Contact

Nếu có vấn đề hoặc câu hỏi, vui lòng liên hệ hoặc tạo issue.



