import os
import cv2
import numpy as np
from pathlib import Path

# Base directory with sorted folders
base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles\"  
categories = ["anomaly", "close_rocks", "distant_rocks", "empty_terrain", "horizon", 
              "night", "rock", "rover", "space", "sun"]

def adjust_brightness(image_path, output_path, factor):
    """Adjust image brightness by factor (e.g., 0.8 = -20%, 1.2 = +20%)"""
    img = cv2.imread(image_path)
    adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, adjusted, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 100% quality

# Process each category
for category in categories:
    input_folder = os.path.join(base_dir, category)
    if not os.path.exists(input_folder):
        print(f"Skipping {category} - folder not found")
        continue
    
    print(f"Processing {category}...")
    image_count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            # Generate images from -20% to +20%
            for percent in range(-20, 21):  # -20 to +20 inclusive
                factor = 1.0 + (percent / 100.0)  # Convert percent to factor (0.8 to 1.2)
                output_filename = f"bright_{percent}_{filename}"
                output_path = os.path.join(input_folder, output_filename)
                adjust_brightness(input_path, output_path, factor)
                image_count += 1
        print(f"Generated {image_count} brightness variations for {filename} in {category}")
    print(f"Total images in {category}: {len(os.listdir(input_folder))}")

print("Brightness variations complete! Total images: ~84,000")