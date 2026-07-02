"""
Dataset Splitter - Train, Validation, Test
------------------------------------------------
This script takes your Final_Soil_Dataset folder, with 9 class
subfolders inside it, then creates a new folder structure split
into three parts, train, validation, and test.

This matters, since the test set stays completely unseen during
training, giving an honest, unbiased final accuracy number.

How to use:
1. Check SOURCE_FOLDER and OUTPUT_FOLDER paths below.
2. Run this file using: py -3.11 split_dataset.py
"""

import os
import shutil
import random

# Step 1: Set your folder paths here
SOURCE_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Final_Soil_Dataset"
OUTPUT_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Split_Dataset"

# Step 2: Set your split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def split_class_folder(class_name):
    source_path = os.path.join(SOURCE_FOLDER, class_name)
    images = [f for f in os.listdir(source_path)
              if f.lower().endswith(VALID_EXTENSIONS)]

    random.shuffle(images)  # mix up order, so split is not biased by filename

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():
        split_class_folder_path = os.path.join(OUTPUT_FOLDER, split_name, class_name)
        os.makedirs(split_class_folder_path, exist_ok=True)

        for filename in split_images:
            src = os.path.join(source_path, filename)
            dst = os.path.join(split_class_folder_path, filename)
            shutil.copy2(src, dst)

    print(f"{class_name}: total {total} -> train {len(train_images)}, "
          f"validation {len(val_images)}, test {len(test_images)}")


def main():
    class_folders = [f for f in os.listdir(SOURCE_FOLDER)
                      if os.path.isdir(os.path.join(SOURCE_FOLDER, f))]

    print(f"Found {len(class_folders)} classes to split.")
    print("-" * 50)

    for class_name in class_folders:
        split_class_folder(class_name)

    print("-" * 50)
    print(f"Done. Split dataset created at: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
