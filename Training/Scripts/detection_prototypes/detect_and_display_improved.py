import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from collections import defaultdict

# Paths and settings
model_path = r"P:\petes_code\github\LunarPi\Training\scripts\rock_classifier.pth"
image_path = r"P:\petes_code\github\LunarPi\Training\images\rock_dataset\rock_0000.jpg"  # Full-size image
output_image_path = r"P:\petes_code\github\LunarPi\Training\detected_output.jpg"
tile_size = (684, 332)  # Training tile size
stride = 171  # Tighter overlap (quarter tile width)
confidence_threshold = 0.7  # Lowered for more detections

# Load trained model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Class names and colors
class_names = ["anomaly", "close_rocks", "distant_rocks", "empty_terrain", "horizon", 
               "night", "rock", "rover", "space", "sun"]
colors = {name: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
          for name in class_names}

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_tile(tile):
    tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    tile_tensor = transform(tile_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tile_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()

def merge_boxes(detections, iou_threshold=0.3):
    """Merge overlapping boxes into single bounding boxes"""
    merged = []
    detections.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
    while detections:
        label, conf, (x1, y1, x2, y2) = detections.pop(0)
        keep = True
        for i, (m_label, m_conf, (mx1, my1, mx2, my2)) in enumerate(merged):
            # Calculate IoU
            inter_x1, inter_y1 = max(x1, mx1), max(y1, my1)
            inter_x2, inter_y2 = min(x2, mx2), min(y2, my2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (mx2 - mx1) * (my2 - my1)
            iou = inter_area / (area1 + area2 - inter_area)
            
            if iou > iou_threshold and label == m_label:
                # Merge by expanding to encompass both
                merged[i] = (label, max(conf, m_conf), 
                             (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)))
                keep = False
                break
        if keep:
            merged.append((label, conf, (x1, y1, x2, y2)))
    return merged

def process_image(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    tile_h, tile_w = tile_size

    detections = []
    for y in range(0, h - tile_h + 1, stride):
        for x in range(0, w - tile_w + 1, stride):
            tile = img[y:y + tile_h, x:x + tile_w]
            label, confidence = classify_tile(tile)
            if confidence > confidence_threshold and label in ["rock", "close_rocks", "distant_rocks", "rover"]:
                detections.append((label, confidence, (x, y, x + tile_w, y + tile_h)))
            if confidence > confidence_threshold and label in ["rock", "close_rocks", "distant_rocks", "rover"]:
                label = "rock" if label in ["close_rocks", "distant_rocks"] else label

    # Merge overlapping detections
    merged_detections = merge_boxes(detections)

    # Draw bounding boxes
    for label, confidence, (x1, y1, x2, y2) in merged_detections:
        color = colors[label]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({confidence:.2f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_image_path, img)
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image(image_path)
    print(f"Output saved to {output_image_path}")