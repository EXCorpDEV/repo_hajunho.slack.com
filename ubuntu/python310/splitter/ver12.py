import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import shutil

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# export CUDA_VISIBLE_DEVICES=""

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.saved_model.load("hjh_models/model10.savedmodel")

# Load the labels
class_names = open("hjh_models/10_labels.txt", "r").readlines()

# Create the array of the right shape to feed into the model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Set the paths
input_dir = "/mnt/splitter/datas/fortest"
output_dir = "/mnt/splitter/result"

# Iterate over all files in the input directory and its subdirectories
for root, dirs, files in os.walk(input_dir):
    for file in files:
        # Get the full path of the image file
        image_path = os.path.join(root, file)

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Run inference
        prediction = model(data)
        prediction = prediction.numpy()
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print(f"File: {image_path}")
        print(f"Class: {class_name}, Confidence Score: {confidence_score}")

        # Create the output directory for the predicted class if it doesn't exist
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Copy the image file to the corresponding class directory
        output_path = os.path.join(class_output_dir, file)
        shutil.copy(image_path, output_path)