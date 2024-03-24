import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from deskew import determine_skew

def deskew_and_save_image(image_path, output_path):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    angle = determine_skew(image_array)
    rotated_image = image.rotate(angle, expand=True)
    rotated_image.save(output_path)
    print(f"Processed and saved: {output_path}")

def process_image(file, root, input_folder, output_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(root, file)
        print(f"Processing file: {image_path}")
        relative_path = os.path.relpath(image_path, input_folder)
        output_image_path = os.path.join(output_folder, relative_path)
        output_image_dir = os.path.dirname(output_image_path)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
        deskew_and_save_image(image_path, output_image_path)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                executor.submit(process_image, file, root, input_folder, output_folder)

input_folder = '/home/soai/skewinput'
output_folder = '/home/soai/skewoutput'

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

process_folder(input_folder, output_folder)

