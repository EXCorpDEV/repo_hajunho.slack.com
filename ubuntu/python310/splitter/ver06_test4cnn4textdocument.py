import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from concurrent.futures import ThreadPoolExecutor

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# export CUDA_VISIBLE_DEVICES=""

# 모델 로드
model = load_model('document_classification_model.keras')

# 데이터 경로 설정
data_dir = '/mnt/splitter/alldatas'
result_dir = '/mnt/splitter/result'

# 결과 디렉토리 생성 (없는 경우)
os.makedirs(result_dir, exist_ok=True)

# 이미지 크기 설정
img_width, img_height = 128, 128


# 이미지 분류 함수
def classify_image(image_path):
    # 이미지 로드 및 전처리
    img = image.load_img(image_path, target_size=(img_width, img_height), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # 이미지 예측
    prediction = model.predict(img_array)
    predicted_class = 'data1/class_1' if prediction[0][0] < 0.5 else 'data2/class_2'

    return predicted_class


# 쓰레드 풀 생성
with ThreadPoolExecutor() as executor:
    futures = []

    # 데이터 디렉토리 내의 이미지 파일 리스트 가져오기
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                future = executor.submit(classify_image, image_path)
                futures.append((future, image_path))

    # 모든 분류 작업이 완료될 때까지 대기
    for future, image_path in futures:
        predicted_class = future.result()

        if predicted_class == 'data1/class_1':
            # 파일을 결과 디렉토리로 복사
            shutil.copy(image_path, result_dir)
            print(f"Image: {image_path} copied to {result_dir}")
        else:
            print(f"Image: {image_path} not copied")