import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

base_dir = r"P:\petes_code\github\LunarPi\Training\images\small_dataset_tiles"  
categories = ["anomaly", "close_rocks", "distant_rocks", "empty_terrain", "horizon", 
              "night", "rock", "rover", "space", "sun"]

def adjust_brightness(args):
    image_path, output_path, factor = args
    img = cv2.imread(image_path)
    adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, adjusted, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def process_category(category):
    input_folder = os.path.join(base_dir, category)
    if not os.path.exists(input_folder):
        print(f"Skipping {category} - folder not found")
        return
    
    print(f"Processing {category}...")
    tasks = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            for percent in range(-20, 21):  # -20 to +20 inclusive
                factor = 1.0 + (percent / 100.0)
                output_filename = f"bright_{percent}_{filename}"
                output_path = os.path.join(input_folder, output_filename)
                tasks.append((input_path, output_path, factor))
    
    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        pool.map(adjust_brightness, tasks)
    print(f"Generated brightness variations for {len(os.listdir(input_folder))} images in {category}")

if __name__ == "__main__":
    print(f"Using {cpu_count()} CPU cores")
    for category in categories:
        process_category(category)
    print("Brightness variations complete! Total images: ~1,760,000")