# Hướng Dẫn Training YOLO cho MVTec Dataset

## Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Cài Đặt](#cài-đặt)
3. [Cấu Trúc Dataset](#cấu-trúc-dataset)
4. [Pipeline](#pipeline)
5. [Sử Dụng](#sử-dụng)

---

## Tổng Quan

Pipeline này bao gồm:
1. **Chuẩn bị dataset YOLO format**: Convert MVTec dataset sang YOLO format với bounding boxes từ ground truth masks
2. **Augmentation**: Augment ảnh và bounding boxes tương ứng
3. **Training**: Train YOLO model
4. **Inference & Evaluation**: Inference và đánh giá model

---

## Cài Đặt

### 1. Kích hoạt môi trường ảo
```bash
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install ultralytics opencv-python numpy pillow tqdm
```

Hoặc nếu đã có requirements:
```bash
pip install -r requirements_fasterrcnn.txt
pip install ultralytics opencv-python
```

---

## Cấu Trúc Dataset

### Dataset gốc (MVTec)
```
datasets/mvtec/bottle/
├── train/
│   └── good/          # Ảnh good
├── test/
│   ├── good/         # Ảnh good
│   ├── broken_large/ # Ảnh defect
│   ├── broken_small/
│   └── contamination/
└── ground_truth/     # Masks cho defect images
    ├── broken_large/
    ├── broken_small/
    └── contamination/
```

### Dataset YOLO format (sau khi prepare)
```
datasets/yolo_mvtec_bottle/
├── train/
│   ├── images/       # Ảnh train (JPG)
│   └── labels/       # Labels YOLO format (.txt)
└── val/
    ├── images/       # Ảnh validation (JPG)
    └── labels/       # Labels YOLO format (.txt)
```

### Format Label YOLO
Mỗi file `.txt` chứa:
```
class_id x_center y_center width height
```
- Tất cả giá trị được normalize về [0, 1]
- `class_id`: 0 = good, 1 = defect
- Ảnh good không có bounding box (file rỗng hoặc không có file)

---

## Pipeline

### Cách 1: Chạy toàn bộ pipeline tự động (tự động sử dụng GPU M1)

```bash
python run_yolo_pipeline.py \
    --source datasets/mvtec/bottle \
    --output datasets/yolo_mvtec_bottle \
    --augment \
    --train \
    --inference \
    --model_size n \
    --epochs 100 \
    --batch 16
    # Tự động detect và sử dụng GPU M1 (MPS) nếu có
    # Hoặc chỉ định thủ công: --device mps
    # Script tự động dùng venv/bin/python nếu tồn tại (không cần kích hoạt thủ công)
```

### Cách 2: Chạy từng bước riêng lẻ

#### Bước 1: Chuẩn bị dataset YOLO format
```bash
python prepare_yolo_dataset.py \
    --source datasets/mvtec/bottle \
    --output datasets/yolo_mvtec_bottle \
    --train_ratio 0.8
```

#### Bước 2: Augment dataset (tùy chọn)
```bash
python augment_yolo_dataset.py \
    --data_root datasets/yolo_mvtec_bottle \
    --splits train
```

#### Bước 3: Train model
```bash
python train_yolo.py \
    --data datasets/yolo_mvtec_bottle.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

#### Bước 4: Inference và đánh giá
```bash
python inference_yolo.py \
    --model runs/yolo_mvtec/yolo11n_*/weights/best.pt \
    --dataset datasets/yolo_mvtec_bottle \
    --split val \
    --conf 0.25
```

---

## Sử Dụng

### 1. Chuẩn bị Dataset

Script `prepare_yolo_dataset.py` sẽ:
- Đọc ảnh good từ `train/good` và `test/good`
- Đọc ảnh defect từ `test/` (broken_large, broken_small, contamination)
- Convert ground truth masks thành bounding boxes
- Chia train/val theo tỷ lệ
- Tạo YOLO format labels

**Tham số:**
- `--source`: Thư mục dataset gốc
- `--output`: Thư mục output (tự động có prefix yolo_)
- `--train_ratio`: Tỷ lệ train (mặc định 0.8)

### 2. Augmentation

Script `augment_yolo_dataset.py` sẽ:
- Lật ngang (flip_h)
- Lật dọc (flip_v)
- Xoay ±15° (rot_p15, rot_n15)
- Làm mờ (blur)
- Tự động cập nhật bounding boxes tương ứng

**Tham số:**
- `--data_root`: Thư mục dataset YOLO
- `--splits`: Các split cần augment (mặc định: train)

### 3. Training

Script `train_yolo.py` sử dụng Ultralytics YOLO11:
- Model sizes: n (nano), s (small), m (medium), l (large), x (xlarge)
- Tự động validation
- Lưu best model và last checkpoint

**Tham số chính:**
- `--data`: Đường dẫn file YAML config dataset
- `--model`: Kích thước model (n, s, m, l, x)
- `--epochs`: Số epochs
- `--batch`: Batch size
- `--imgsz`: Kích thước ảnh input
- `--device`: Device (`mps` cho MacBook M1/M2/M3, `0` cho CUDA, `cpu`, hoặc để trống = auto-detect)

**Lưu ý về GPU:**
- **MacBook M1/M2/M3**: Tự động detect và sử dụng MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Tự động detect và sử dụng CUDA
- **Không có GPU**: Tự động fallback về CPU
- Có thể chỉ định thủ công: `--device mps` (M1 Mac) hoặc `--device 0` (CUDA)

**Kết quả training:**
- Model được lưu tại: `runs/yolo_mvtec/yolo{size}_{timestamp}/weights/best.pt`
- Training curves và metrics trong thư mục runs

### 4. Inference & Evaluation

Script `inference_yolo.py` sẽ:
- Inference trên dataset split
- Tính toán metrics: Precision, Recall, F1
- Lưu ảnh kết quả với bounding boxes
- Lưu metrics vào JSON

**Tham số:**
- `--model`: Đường dẫn model đã train (.pt)
- `--dataset`: Thư mục dataset YOLO
- `--split`: Split để đánh giá (train, val)
- `--conf`: Confidence threshold (mặc định 0.25)
- `--iou`: IoU threshold cho evaluation (mặc định 0.5)

**Kết quả:**
- Ảnh inference: `runs/yolo_mvtec/inference_{split}/pred_*.jpg`
- Metrics: `runs/yolo_mvtec/inference_{split}/metrics.json`
- Summary: `runs/yolo_mvtec/inference_{split}/results.txt`

---

## Ví Dụ Đầy Đủ

```bash
# 1. Chuẩn bị dataset
python prepare_yolo_dataset.py \
    --source datasets/mvtec/bottle \
    --output datasets/yolo_mvtec_bottle \
    --train_ratio 0.8

# 2. Augment (tùy chọn)
python augment_yolo_dataset.py \
    --data_root datasets/yolo_mvtec_bottle \
    --splits train

# 3. Train (tự động sử dụng GPU M1 nếu có)
python train_yolo.py \
    --data datasets/yolo_mvtec_bottle.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
    # --device mps  # Tự động detect, hoặc chỉ định thủ công cho M1 Mac

# 4. Inference và đánh giá
python inference_yolo.py \
    --model runs/yolo_mvtec/yolo11n_20241114_120000/weights/best.pt \
    --dataset datasets/yolo_mvtec_bottle \
    --split val \
    --conf 0.25
```

---

## Lưu Ý

1. **Dataset riêng biệt**: Dataset YOLO được lưu trong thư mục riêng với prefix `yolo_` để không ảnh hưởng đến model khác
2. **Ground truth masks**: Script cần ground truth masks trong `ground_truth/` để tạo bounding boxes
3. **Good images**: Ảnh good không có bounding box (file label rỗng)
4. **Augmentation**: Augmentation sẽ tạo thêm ảnh trong cùng thư mục, không ghi đè ảnh gốc
5. **Model size**: Model nhỏ hơn (n) train nhanh hơn nhưng độ chính xác thấp hơn. Model lớn hơn (x) chính xác hơn nhưng cần nhiều tài nguyên hơn

---

## Troubleshooting

### Lỗi: "No ground truth masks found"
- Kiểm tra thư mục `ground_truth/` có tồn tại không
- Đảm bảo tên mask file đúng format: `{image_name}_mask.png`

### Lỗi: "Model not found"
- Kiểm tra đường dẫn model
- Hoặc train lại model

### Lỗi: "Dataset YAML not found"
- Tạo file YAML thủ công hoặc để script tự tạo
- Format: xem `datasets/yolo_mvtec_bottle.yaml`

---

## Kết Quả

Sau khi hoàn thành pipeline, bạn sẽ có:
- Dataset YOLO format tại `datasets/yolo_mvtec_bottle/`
- Model đã train tại `runs/yolo_mvtec/yolo{size}_{timestamp}/weights/best.pt`
- Kết quả inference và metrics tại `runs/yolo_mvtec/inference_{split}/`

