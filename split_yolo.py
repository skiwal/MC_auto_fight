from pathlib import Path
import random
import shutil


# =========================
# 1. 配置路径
# =========================

# 你的原始数据集目录
# 里面应该有 images/ 和 labels/
INPUT_ROOT = Path(r"D:\code\MC_auto_fight\video_data\data")

# 输出的 YOLO 标准数据集目录
OUTPUT_ROOT = Path(r"D:\code\MC_auto_fight\yolo_dataset")

IMAGE_DIR = INPUT_ROOT / "images"
LABEL_DIR = INPUT_ROOT / "labels"

# 数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子，保证每次划分结果一致
RANDOM_SEED = 42

# 是否复制文件
# True = 复制，不破坏原数据
# False = 移动，原文件会被移走
COPY_FILES = True

# 类别名称，按你的实际类别修改
# 例如只识别一种目标：
CLASS_NAMES = ["target"]

# 如果你有多个类别，比如：
# CLASS_NAMES = ["zombie", "skeleton", "creeper"]


# =========================
# 2. 工具函数
# =========================

def make_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path):
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def write_data_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"

    names_text = "\n".join([f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)])

    content = f"""path: {OUTPUT_ROOT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
{names_text}
"""

    yaml_path.write_text(content, encoding="utf-8")
    print(f"Created data.yaml: {yaml_path}")


# =========================
# 3. 主逻辑
# =========================

def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label folder not found: {LABEL_DIR}")

    make_dirs()

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    image_paths = []
    for ext in image_exts:
        image_paths.extend(IMAGE_DIR.glob(f"*{ext}"))
        image_paths.extend(IMAGE_DIR.glob(f"*{ext.upper()}"))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {IMAGE_DIR}")

    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)

    total = len(image_paths)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = image_paths[:train_count]
    val_images = image_paths[train_count:train_count + val_count]
    test_images = image_paths[train_count + val_count:]

    split_map = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    missing_label_count = 0

    for split, images in split_map.items():
        print(f"\nProcessing {split}: {len(images)} images")

        out_img_dir = OUTPUT_ROOT / "images" / split
        out_label_dir = OUTPUT_ROOT / "labels" / split

        for img_path in images:
            label_path = LABEL_DIR / f"{img_path.stem}.txt"

            out_img_path = out_img_dir / img_path.name
            out_label_path = out_label_dir / f"{img_path.stem}.txt"

            copy_or_move(img_path, out_img_path)

            if label_path.exists():
                copy_or_move(label_path, out_label_path)
            else:
                # 如果图片没有目标，可以创建空标签文件
                # YOLO 可以把这种图片当作背景负样本
                out_label_path.write_text("", encoding="utf-8")
                missing_label_count += 1

    write_data_yaml()

    print("\nDone.")
    print(f"Total images: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")
    print(f"Missing labels created as empty txt: {missing_label_count}")
    print(f"Output dataset: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
