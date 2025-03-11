from PIL import Image
import os

def split_image(image_path, output_folder, chunk_size=(512, 512)):
    image = Image.open(image_path)
    img_width, img_height = image.size

    basename = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for top in range(0, img_height, chunk_size[1]):
        for left in range(0, img_width, chunk_size[0]):
            box = (left, top, left + chunk_size[0], top + chunk_size[1])
            cropped_image = image.crop(box)

            # Only save chunks that match the chunk_size exactly
            if cropped_image.size == chunk_size:
                chunk_name = f"{basename}_chunk_{count}.png"
                cropped_image.save(os.path.join(output_folder, chunk_name))
                count += 1

    print(f"{basename}: {count} chunks created.")

# Usage
input_folder = "../images/small_dataset"
output_folder = "../images/small_data_png"
chunk_size = (512, 512)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        split_image(os.path.join(input_folder, filename), output_folder, chunk_size)
