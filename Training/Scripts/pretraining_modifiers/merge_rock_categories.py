import os
import shutil
from pathlib import Path

base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles"  
rock_categories = ["close_rocks", "distant_rocks", "rock"]
new_rock_dir = os.path.join(base_dir, "rock_combined")

# Create new rock_combined folder
os.makedirs(new_rock_dir, exist_ok=True)

# Move all rock images into rock_combined
for category in rock_categories:
    src_folder = os.path.join(base_dir, category)
    if os.path.exists(src_folder):
        for filename in os.listdir(src_folder):
            if filename.lower().endswith(".jpg"):
                src_path = os.path.join(src_folder, filename)
                dst_path = os.path.join(new_rock_dir, filename)
                shutil.move(src_path, dst_path)
        # Remove old folder if empty
        if not os.listdir(src_folder):
            os.rmdir(src_folder)
    print(f"Moved {category} to rock_combined")

# Update train and test splits
train_dir = r"P:\petes_code\github\LunarPi\Training\train"
test_dir = r"P:\petes_code\github\LunarPi\Training\test"
for split_dir in [train_dir, test_dir]:
    rock_combined_split = os.path.join(split_dir, "rock_combined")
    os.makedirs(rock_combined_split, exist_ok=True)
    for category in rock_categories:
        src_split_folder = os.path.join(split_dir, category)
        if os.path.exists(src_split_folder):
            for filename in os.listdir(src_split_folder):
                if filename.lower().endswith(".jpg"):
                    shutil.move(os.path.join(src_split_folder, filename), 
                               os.path.join(rock_combined_split, filename))
            if not os.listdir(src_split_folder):
                os.rmdir(src_split_folder)
    print(f"Updated {split_dir} with rock_combined")

print("Rock categories merged!")