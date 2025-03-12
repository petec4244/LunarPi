import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import shutil

# Paths
input_dir = "../images/small_dataset"  
output_dir = "../images/small_dataset_tiles"
os.makedirs(output_dir, exist_ok=True)

# Split function (skip if already split)
def split_image(image_path, rows=8, cols=11):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    tiles = []
    for i in range(rows):
        for j in range(cols):
            tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    return tiles

# Feature extraction
def extract_features(tile):
    # Mean RGB + basic texture (std dev)
    mean_rgb = np.mean(tile, axis=(0, 1))
    std_rgb = np.std(tile, axis=(0, 1))
    return np.concatenate([mean_rgb, std_rgb])

# Process images
all_tiles = []
tile_paths = []
if not os.path.exists(output_dir) or len(os.listdir(output_dir)) < 37000:
    print("Splitting images...")
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            tiles = split_image(img_path)
            for tile_idx, tile in enumerate(tiles):
                tile_filename = f"{output_dir}/tile_{idx}_{tile_idx}.jpg"
                cv2.imwrite(tile_filename, tile)
                all_tiles.append(extract_features(tile))
                tile_paths.append(tile_filename)
        print(f"Processed {idx+1} images")
else:
    print("Using existing split images...")
    for filename in os.listdir(output_dir):
        if filename.endswith(".jpg"):
            tile_path = os.path.join(output_dir, filename)
            tile = cv2.imread(tile_path)
            all_tiles.append(extract_features(tile))
            tile_paths.append(tile_path)

# Cluster into 7 groups (for varied content)
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(np.array(all_tiles))

# Organize into folders
class_dirs = {i: f"../small_dataset_tiles/cluster_{i}" for i in range(7)}
for d in class_dirs.values():
    os.makedirs(d, exist_ok=True)

for label, tile_path in zip(labels, tile_paths):
    shutil.move(tile_path, os.path.join(class_dirs[label], os.path.basename(tile_path)))

print("Clustering complete! Check ../small_dataset_tiles/cluster_* folders.")