import os
import shutil
import numpy as np
from PIL import Image

input_folder = '../categorized_chunks/empty_soil'
output_folder_black_borders = '../categorized_chunks/cutoff'
output_folder_clean = '../categorized_chunks/clean_soil'

os.makedirs(output_folder_black_borders, exist_ok=True)
os.makedirs(output_folder_clean, exist_ok=True)

def has_black_border(img, threshold=5):
    # Convert to grayscale numpy array
    img_array = np.array(img.convert('L'))
    
    # Check if any edge contains significant black pixels
    top_edge = img_array[0, :]
    bottom_edge = img_array[-1, :]
    left_edge = img_array[:, 0]
    right_edge = img_array[:, -1]

    # Threshold (adjustable, 10 = very dark pixels)
    threshold_value = 10

    edges = [top_edge, bottom_edge, left_edge, right_edge]
    for edge in edges:
        if np.mean(edge) < threshold_value:
            return True
    return False

os.makedirs(output_folder_black_borders, exist_ok=True)
os.makedirs(output_folder_clean, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        img_array = np.array(img.convert('L'))

        if (np.mean(img_array[:, -1]) < 10 or 
            np.mean(img_array[:, 0]) < 10 or 
            np.mean(img_array[0, :]) < 10 or
            np.mean(img_array[-1, :]) < 10):

            shutil.move(img_path, os.path.join(output_folder_black_borders, filename))
        else:
            shutil.copy(img_path, os.path.join(output_folder_clean, filename))
