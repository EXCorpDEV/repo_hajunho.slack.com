import tensorflow as tf
import os

# 설정값
data_dir = './'
batch_size = 32
img_height = 224
img_width = 224
epochs = 10
model_save_path = 'my_food_classifier.keras'
tflite_save_path = 'my_food_classifier.tflite'
class_names_file = 'class_names.txt'

# 데이터셋 생성 (한글 경로 지원)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 클래스 이름 저장
class_names = train_ds.class_names
print("클래스 목록:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")

with open(class_names_file, 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(f"{name}\n")

# 모델 정의
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습
model.fit(train_ds, epochs=epochs)

# 모델 저장
model.save(model_save_path)

# TFLite 변환
converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
tflite_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite 모델 저장 완료: {tflite_save_path}")
print(f"클래스 목록 저장 완료: {class_names_file}")

