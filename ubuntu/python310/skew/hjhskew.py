import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.color import rgb2gray

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
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_array = np.array(image)

    angle = determine_skew(image_array)
    corrected_angle = angle
    rotated_image = image.rotate(-corrected_angle, expand=True)  # FastAPI 예제와 동일하게 적용
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
