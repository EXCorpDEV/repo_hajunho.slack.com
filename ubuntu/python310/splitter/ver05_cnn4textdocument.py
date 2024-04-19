#export CUDA_VISIBLE_DEVICES=""
#tensorboard --logdir=logs --bind_all

import os, json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import threading

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 데이터 경로 설정
data_dir = '/mnt/splitter/datas'
train_dir = os.path.join(data_dir, 'data1')
test_dir = os.path.join(data_dir, 'data2')

# 데이터 경로 확인
if not os.path.exists(train_dir):
    raise ValueError(f"Training data directory does not exist: {train_dir}")
if not os.path.exists(test_dir):
    raise ValueError(f"Testing data directory does not exist: {test_dir}")

# 이미지 크기 설정
img_width, img_height = 128, 128
input_shape = (img_width, img_height, 1)

# 데이터 제너레이터 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='binary',
    color_mode='grayscale'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='binary',
    color_mode='grayscale'
)

# 모델 구성
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# 모델 컴파일
def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

def train_model(model):
    epochs = 30
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size
    )

    class_indices = train_generator.class_indices
    print(f'Class indices: {class_indices}')
    with open('class_indices.json', 'w') as file:
        json.dump(class_indices, file)


# 모델 평가
def evaluate_model(model):
    _, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy:.2f}')

# 모델 저장
def save_model(model):
    model.save('document_classification_model.h5')

# 쓰레드 생성 및 실행
def run_threads():
    model = build_model()
    compile_model(model)

    train_thread = threading.Thread(target=train_model, args=(model,))
    train_thread.start()
    train_thread.join()

    evaluate_thread = threading.Thread(target=evaluate_model, args=(model,))
    save_thread = threading.Thread(target=save_model, args=(model,))

    evaluate_thread.start()
    save_thread.start()

    evaluate_thread.join()
    save_thread.join()

# 메인 함수
if __name__ == '__main__':
    run_threads()