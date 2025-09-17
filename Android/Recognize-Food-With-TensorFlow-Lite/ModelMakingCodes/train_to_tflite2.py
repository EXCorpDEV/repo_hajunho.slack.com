import tensorflow as tf
import os
from tensorflow.keras import layers, models

# Dataset Parameters
img_height = 224
img_width = 224
batch_size = 32

dataset_dir = './'  # 현재 폴더

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Prepare Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=123
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f'클래스 목록: {class_names}')

# Define Model
base_model = tf.keras.applications.EfficientNetB3(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    data_augmentation,
    tf.keras.applications.efficientnet.preprocess_input,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train Model
model.fit(train_ds, epochs=50, callbacks=[early_stop])

# Save Model
model_save_path = 'my_food_classifier.keras'
model.save(model_save_path)
print(f'Model saved to {model_save_path}')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('my_food_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print('TFLite model saved as my_food_classifier.tflite')


