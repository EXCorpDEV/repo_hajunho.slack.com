import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

def determine_skew(image):
    if len(image.shape) == 2:
        pass
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
        elif image.shape[2] == 4:
            image = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    edges = canny(image, sigma=3.0)
    h, theta, d = hough_line(edges)
    angles = []

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        angles.append(angle)

    if len(angles) == 0:
        return 0

    return np.rad2deg(np.median(angles))

def deskew_image(input_path, output_path):
    # 인풋 폴더가 없으면 생성
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # 아웃풋 폴더가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing file: {image_path}")
                image = io.imread(image_path)

                if len(image.shape) == 2:
                    pass
                elif len(image.shape) == 3:
                    if image.shape[2] == 3:
                        image = np.mean(image, axis=2).astype(np.uint8)
                    elif image.shape[2] == 4:
                        image = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
                    else:
                        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")

                angle = determine_skew(image)
                rotated_image = Image.fromarray(image).rotate(angle, expand=True)

                relative_path = os.path.relpath(root, input_path)
                output_subfolder = os.path.join(output_path, relative_path)

                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                output_image_path = os.path.join(output_subfolder, file)
                rotated_image.save(output_image_path)

        for dir in dirs:
            deskew_image(os.path.join(root, dir), os.path.join(output_path, dir))

input_folder = '/home/soai/skewinput'
output_folder = '/home/soai/skewoutput'

# 인풋 폴더와 아웃풋 폴더 생성
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

deskew_image(input_folder, output_folder)