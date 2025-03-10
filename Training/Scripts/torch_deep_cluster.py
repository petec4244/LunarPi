import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import shutil

# Paths
#input_dir = "../output_chunks"  # Where 37,840 small tiles are
#output_dir = "../images/torch_deep_cluster/clustered_images"

# Absolute paths (adjust if tiles are elsewhere)
input_dir = r"P:\petes_code\github\LunarPi\Training\output_chunks"  # Where 37,840 tiles are
output_dir = r"P:\petes_code\github\LunarPi\Training\images\torch_deep_cluster\clustered_images"
os.makedirs(output_dir, exist_ok=True)

# Load pre-trained ResNet
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Identity()
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_deep_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process tiles
all_features = []
tile_paths = []
print(f"Looking for tiles in: {input_dir}")
tile_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]  # Changed to .png
total_tiles = len(tile_files)
if total_tiles == 0:
    raise ValueError(f"No .png files found in {input_dir}! Verify path and contents.")
print(f"Found {total_tiles} tiles")

print("Extracting features...")
for idx, filename in enumerate(tile_files):
    tile_path = os.path.join(input_dir, filename)
    features = extract_deep_features(tile_path)
    if features is not None:
        all_features.append(features)
        tile_paths.append(tile_path)
    else:
        print(f"Skipped {tile_path}")
    if idx % 1000 == 0:
        print(f"Processed {idx}/{total_tiles} tiles")

if not all_features:
    raise ValueError(f"No features extracted from {total_tiles} tiles! Check images.")
print(f"Extracted features for {len(all_features)}/{total_tiles} tiles")

# Cluster with K-means
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(np.array(all_features))

# Organize into folders
class_dirs = {i: f"{output_dir}/cluster_{i}" for i in range(n_clusters)}
for d in class_dirs.values():
    os.makedirs(d, exist_ok=True)

for label, tile_path in zip(labels, tile_paths):
    shutil.move(tile_path, os.path.join(class_dirs[label], os.path.basename(tile_path)))

print("Deep clustering complete! Check clustered_images/cluster_* folders.")