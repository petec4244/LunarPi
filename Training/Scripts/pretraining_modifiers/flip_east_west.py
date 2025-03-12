import os
import cv2
from pathlib import Path

# Base directory with sorted folders
base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles"  
categories = ["anomaly", "close_rocks", "distant_rocks", "empty_terrain", "horizon", 
              "night", "rock", "rover", "space", "sun"] 

def flip_image(image_path, output_path):
    img = cv2.imread(image_path)
    flipped = cv2.flip(img, 1)  # 1 = horizontal flip
    cv2.imwrite(output_path, flipped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 100% quality

# Process each category
for category in categories:
    input_folder = os.path.join(base_dir, category)
    if not os.path.exists(input_folder):
        print(f"Skipping {category} - folder not found")
        continue
    
    print(f"Processing {category}...")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"): 
            input_path = os.path.join(input_folder, filename)
            output_filename = f"flip_{filename}"
            output_path = os.path.join(input_folder, output_filename)
            flip_image(input_path, output_path)
    print(f"Flipped {len(os.listdir(input_folder))} images in {category}")

print("East-West flipping complete!")