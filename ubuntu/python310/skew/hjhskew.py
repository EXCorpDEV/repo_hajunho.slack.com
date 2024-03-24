import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.color import rgb2gray
from io import BytesIO

def determine_skew(image):
    if image.ndim == 3:
        image = rgb2gray(image)
    edges = canny(image, sigma=3.0)
    h, theta, d = hough_line(edges)
    angles = [np.rad2deg(angle) for _, angle, _ in zip(*hough_line_peaks(h, theta, d))]
    if not angles:
        return 0
    median_angle = np.median(angles)
    return median_angle

def deskew_and_save_image(image_path, output_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_array = np.array(image)

    angle = determine_skew(image_array)
    rotated_image = image.rotate(angle, expand=True)
    rotated_image.save(output_path)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing file: {image_path}")
                relative_path = os.path.relpath(image_path, input_folder)
                output_image_path = os.path.join(output_folder, relative_path)
                output_image_dir = os.path.dirname(output_image_path)
                if not os.path.exists(output_image_dir):
                    os.makedirs(output_image_dir)
                deskew_and_save_image(image_path, output_image_path)

input_folder = '/home/soai/skewinput'
output_folder = '/home/soai/skewoutput'

process_folder(input_folder, output_folder)
