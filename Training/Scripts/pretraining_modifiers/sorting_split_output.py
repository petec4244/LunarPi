import os
import shutil
import cv2
import numpy as np
from PIL import Image

input_folder = '../output_chunks'
output_folder = '../categorized_chunks'
os.makedirs(output_folder, exist_ok=True)

categories = ['sun', 'space', 'rocks_or_rover', 'horizon', 'empty_soil']
for cat in categories:
    os.makedirs(os.path.join(output_folder, cat), exist_ok=True)

def categorize_image(img_path):
    image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    avg_brightness = np.mean(image_gray)
    
    # Brightness threshold (tune if necessary)
    if avg_brightness > 180:
        return 'sun'
    elif avg_brightness < 30:
        return 'space'
    
    edges = cv2.Canny(image_gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Edge density thresholds (tune as needed)
    if edge_density > 0.15:
        return 'rocks_or_rover'
    elif edge_density > 0.05:
        return 'horizon'
    else:
        return 'empty_soil'

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(input_folder, filename)
        category = categorize_image(path)
        shutil.copy(path, os.path.join(output_folder, category, filename))

print("Automatic categorization completed.")
