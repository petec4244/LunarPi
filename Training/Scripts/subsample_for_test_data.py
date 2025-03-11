# subsample_images.py
import os
import shutil
import random
from pathlib import Path

base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles"  
train_dir = r"P:\petes_code\github\LunarPi\Training\train"
test_dir = r"P:\petes_code\github\LunarPi\Training\test"
categories = ["anomaly", "close_rocks", "distant_rocks", "empty_terrain", "horizon", 
              "night", "rock", "rover", "space", "sun"]
target_total = 100000  # Desired total images

# Count total images
total_images = 0
for category in categories:
    folder = os.path.join(base_dir, category)
    total_images += len([f for f in os.listdir(folder) if f.endswith(".jpg")])
print(f"Total images: {total_images}")

# Calculate sampling ratio
sample_ratio = min(1.0, target_total / total_images)
train_ratio = 0.8

for category in categories:
    input_folder = os.path.join(base_dir, category)
    train_folder = os.path.join(train_dir, category)
    test_folder = os.path.join(test_dir, category)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    sample_size = int(sample_ratio * len(files))
    sampled_files = random.sample(files, sample_size)
    
    split_idx = int(train_ratio * len(sampled_files))
    train_files = sampled_files[:split_idx]
    test_files = sampled_files[split_idx:]
    
    for f in train_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(train_folder, f))
    for f in test_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(test_folder, f))
    print(f"{category}: {len(train_files)} train, {len(test_files)} test")

print(f"Subsampled to ~{target_total} images!")