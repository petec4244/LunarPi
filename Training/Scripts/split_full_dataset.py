import os
import shutil
import random
from pathlib import Path

# Directory with all augmented files
base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
categories = ["anomaly", "empty_terrain", "horizon", "night", "rock_combined", "rover", "space", "sun"]

# Create train and test directories
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Split ratio
train_ratio = 0.8

# Process each category
for category in categories:
    input_folder = os.path.join(base_dir, category)
    if not os.path.exists(input_folder):
        print(f"Skipping {category} - folder not found")
        continue

    # Get all image files
    files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    total_files = len(files)
    if total_files == 0:
        print(f"No files found in {category}")
        continue

    # Shuffle files
    random.shuffle(files)

    # Split into train and test
    split_idx = int(train_ratio * total_files)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Move train files
    for f in train_files:
        src_path = os.path.join(input_folder, f)
        dst_path = os.path.join(train_dir, category, f)
        shutil.move(src_path, dst_path)

    # Move test files
    for f in test_files:
        src_path = os.path.join(input_folder, f)
        dst_path = os.path.join(test_dir, category, f)
        shutil.move(src_path, dst_path)

    print(f"{category}: {len(train_files)} train, {len(test_files)} test")

    # Remove empty category folder (optional)
    if not os.listdir(input_folder):
        os.rmdir(input_folder)

print("Dataset split complete!")