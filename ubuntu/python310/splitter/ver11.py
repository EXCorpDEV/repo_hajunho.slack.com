#tensorboard --logdir=logs --bind_all
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from scipy import ndimage
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
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)

# 데이터 제너레이터 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='binary'
)

# 전이 학습을 위한 모델 구성
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# 모델 컴파일
def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 모델 학습
def train_model(model):
    epochs = 50
    log_dir = 'logs'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        callbacks=[tensorboard_callback]
    )

# 모델 평가
def evaluate_model(model):
    _, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy:.2f}')

# 모델 저장
def save_model(model):
    model.save('image_classification_model_transfer_learning.h5')

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