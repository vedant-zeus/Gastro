import os
import shutil
import random
from glob import glob

# Set these paths
SOURCE_DIR = "../Gastrovision"  # Path to your original Gastrovision dataset (27 class folders)
DEST_DIR = "dataset"  # Will create train/val/test inside this

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

random.seed(42)

os.makedirs(os.path.join(DEST_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "test"), exist_ok=True)

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:
    img_paths = glob(os.path.join(SOURCE_DIR, cls, "*"))
    random.shuffle(img_paths)
    n_total = len(img_paths)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_test = n_total - n_train - n_val

    splits = {
        "train": img_paths[:n_train],
        "val": img_paths[n_train:n_train+n_val],
        "test": img_paths[n_train+n_val:]
    }

    for split, paths in splits.items():
        split_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for img_path in paths:
            shutil.copy(img_path, split_dir)

print("Dataset split complete! Check the 'dataset' folder.")
