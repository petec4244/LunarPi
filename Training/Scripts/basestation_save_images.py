import zmq
import pickle
import os
from PIL import Image
import io

# ZMQ setup
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Directory for images
output_dir = "rock_dataset"
os.makedirs(output_dir, exist_ok=True)
image_count = 0

print("Waiting for images...")
while True:
    data = socket.recv()
    image_data = pickle.loads(data)
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Save image
    filename = f"{output_dir}/rock_{image_count:04d}.jpg"
    img.save(filename, "JPEG")
    image_count += 1
    print(f"Saved {filename}")

    socket.send(pickle.dumps("saved"))