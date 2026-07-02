"""
Corrected Dataset Pipeline - No Data Leakage
--------------------------------------------------
This fixes a serious issue from before, where augmentation was
done BEFORE splitting, causing near-duplicate images to leak
between train and test sets, giving a falsely high accuracy.

Correct order, used here:
1. Identify original images only (ignore filenames with "_aug").
2. Split ONLY original images into train, validation, and test.
3. Apply augmentation ONLY to the training folder, for each class,
   to balance class sizes. Validation and test stay 100% untouched,
   original, real images, never seen during training.

How to use:
Run this file using: py -3.11 fix_pipeline.py
"""

import os
import shutil
import random
from PIL import Image, ImageEnhance

# Step 1: Set your folder paths
SOURCE_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Final_Soil_Dataset"
OUTPUT_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Split_Dataset_Fixed"

TRAIN_FOLDER = os.path.join(OUTPUT_FOLDER, "train")
VAL_FOLDER = os.path.join(OUTPUT_FOLDER, "validation")
TEST_FOLDER = os.path.join(OUTPUT_FOLDER, "test")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_TRAIN_COUNT = 630   # roughly 70% of 900, the earlier per-class target

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_original_image(filename):
    """
    Returns True only for original images, filtering out any
    file that was created by our earlier augmentation script,
    since those all contain '_aug' in their filename.
    """
    return "_aug" not in filename.lower()


def augment_image(img):
    choice = random.choice(["flip", "rotate", "brightness", "zoom"])

    if choice == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif choice == "rotate":
        angle = random.choice([10, -10, 20, -20, 90])
        return img.rotate(angle)
    elif choice == "brightness":
        factor = random.uniform(0.6, 1.4)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    elif choice == "zoom":
        width, height = img.size
        crop_percent = random.uniform(0.8, 0.95)
        new_w, new_h = int(width * crop_percent), int(height * crop_percent)
        left = (width - new_w) // 2
        top = (height - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((width, height))


def process_class(class_name):
    source_path = os.path.join(SOURCE_FOLDER, class_name)
    all_files = [f for f in os.listdir(source_path)
                 if f.lower().endswith(VALID_EXTENSIONS)]

    original_files = [f for f in all_files if is_original_image(f)]
    random.shuffle(original_files)

    total_originals = len(original_files)
    train_end = int(total_originals * TRAIN_RATIO)
    val_end = train_end + int(total_originals * VAL_RATIO)

    train_files = original_files[:train_end]
    val_files = original_files[train_end:val_end]
    test_files = original_files[val_end:]

    # Step A: copy validation and test images, untouched, pure originals
    for split_name, files in [("validation", val_files), ("test", test_files)]:
        dest_folder = os.path.join(OUTPUT_FOLDER, split_name, class_name)
        os.makedirs(dest_folder, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(source_path, f), os.path.join(dest_folder, f))

    # Step B: copy original training images first
    train_dest_folder = os.path.join(TRAIN_FOLDER, class_name)
    os.makedirs(train_dest_folder, exist_ok=True)
    for f in train_files:
        shutil.copy2(os.path.join(source_path, f), os.path.join(train_dest_folder, f))

    # Step C: augment ONLY within the training folder, to reach target count
    current_train_count = len(train_files)
    if current_train_count < TARGET_TRAIN_COUNT and current_train_count > 0:
        needed = TARGET_TRAIN_COUNT - current_train_count
        created = 0
        idx = 0
        while created < needed:
            filename = train_files[idx % len(train_files)]
            img_path = os.path.join(train_dest_folder, filename)
            original = Image.open(img_path).convert("RGB")
            new_img = augment_image(original)
            name_only = os.path.splitext(filename)[0]
            new_filename = f"{name_only}_aug{created+1}.jpg"
            new_img.save(os.path.join(train_dest_folder, new_filename))
            created += 1
            idx += 1

    final_train_count = len(os.listdir(train_dest_folder))
    print(f"{class_name}: originals {total_originals} -> "
          f"train {final_train_count} (incl. augmentation), "
          f"validation {len(val_files)}, test {len(test_files)}")


def main():
    class_folders = [f for f in os.listdir(SOURCE_FOLDER)
                      if os.path.isdir(os.path.join(SOURCE_FOLDER, f))]

    print(f"Found {len(class_folders)} classes.")
    print("Splitting ORIGINAL images first, then augmenting only training data...")
    print("-" * 60)

    for class_name in class_folders:
        process_class(class_name)

    print("-" * 60)
    print(f"Done. Corrected, leak-free dataset created at: {OUTPUT_FOLDER}")
    print("Validation and test folders contain ONLY real, original, untouched images.")


if __name__ == "__main__":
    main()
