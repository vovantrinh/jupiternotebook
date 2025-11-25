#!/usr/bin/env python3
"""
Augment train images inside datasets/mvtec_balanced/bottle/train

Các phép biến đổi:
    - Lật ngang, lật dọc
    - Xoay ±15° (giữ kích thước, fill=RGB median)
    - Gaussian blur nhẹ

Ảnh augment được lưu cùng thư mục với ảnh gốc, tên dạng
<stem>_<suffix>.<ext>. Nếu file đã tồn tại thì bỏ qua để tránh nhân đôi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def rotate(img: Image.Image, degrees: float) -> Image.Image:
    # rotate giữ kích thước, fill bằng màu trung bình ảnh
    median_color = tuple(int(x) for x in ImageStat.Stat(img).median)
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=median_color)


def augment_image(img_path: Path, output_dir: Path, suffix: str, transform) -> None:
    out_path = output_dir / f"{img_path.stem}_{suffix}{img_path.suffix}"
    if out_path.exists():
        return
    with Image.open(img_path) as img:
        aug_img = transform(img.convert("RGB"))
        aug_img.save(out_path)


def augment_class_folder(folder: Path) -> None:
    output_dir = folder
    images = list_images(folder)
    if not images:
        print(f"⚠️  Không có ảnh trong {folder}")
        return

    print(f"Augmenting {len(images)} images in {folder}")

    transforms = [
        ("flip_h", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("flip_v", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
        ("rot_p15", lambda img: rotate(img, 15)),
        ("rot_n15", lambda img: rotate(img, -15)),
        ("blur", lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.5))),
    ]

    for img_path in images:
        for suffix, fn in transforms:
            augment_image(img_path, output_dir, suffix, fn)


def main():
    parser = argparse.ArgumentParser(description="Augment balanced train dataset in place")
    parser.add_argument(
        "--data_root",
        default="datasets/mvtec_balanced/bottle",
        help="Đường dẫn tới dataset đã cân bằng",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Các split cần augment (mặc định chỉ train)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["good", "defect"],
        help="Các lớp cần augment",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"❌ Không tìm thấy {data_root}")

    for split in args.splits:
        for cls in args.classes:
            folder = data_root / split / cls
            if not folder.exists():
                print(f"⚠️  Bỏ qua {folder}, không tồn tại")
                continue
            augment_class_folder(folder)

    print("✓ Hoàn tất augment")


if __name__ == "__main__":
    main()


