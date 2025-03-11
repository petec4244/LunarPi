import os
import cv2
import numpy as np
from pathlib import Path

base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles"  
categories = ["anomaly", "close_rocks", "distant_rocks", "empty_terrain", "horizon", 
              "night", "rock", "rover", "space", "sun"]

def rotate_image(image_path, output_path, angle):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    cv2.imwrite(output_path, rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

for category in categories:
    input_folder = os.path.join(base_dir, category)
    if not os.path.exists(input_folder):
        print(f"Skipping {category} - folder not found")
        continue
    
    print(f"Processing {category}...")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            # Rotate from -5° to +5°
            for angle in range(-5, 6):  # -5 to +5 inclusive (11 steps)
                output_filename = f"rot_{angle}_{filename}"
                output_path = os.path.join(input_folder, output_filename)
                rotate_image(input_path, output_path, angle)
    print(f"Rotated {len(os.listdir(input_folder))} images in {category}")

print("Rotation complete!")