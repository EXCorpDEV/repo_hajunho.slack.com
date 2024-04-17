import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('image_classification_model.h5')

# Set the data directory path
data_dir = '/mnt/splitter/datas'
class1_dir = os.path.join(data_dir, 'class1')
class2_dir = os.path.join(data_dir, 'class2')

# Set the image size
img_width, img_height = 224, 224

# List of class directories
class_dirs = [class1_dir, class2_dir]

# Perform predictions for each class
for class_idx, class_dir in enumerate(class_dirs):
    class_name = os.path.basename(class_dir)
    print(f"Predictions for {class_name}:")

    # Get the list of image files in the class directory
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Perform predictions for each image
    for image_file in image_files:
        image_path = os.path.join(class_dir, image_file)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Predict the image
        prediction = model.predict(img_array)
        predicted_class = 'class1' if prediction[0][0] < 0.5 else 'class2'

        print(f"Image: {image_file}, Actual Class: {class_name}, Predicted Class: {predicted_class}")

    print("---")