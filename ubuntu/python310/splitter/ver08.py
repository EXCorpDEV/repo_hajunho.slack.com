import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from concurrent.futures import ThreadPoolExecutor

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#export CUDA_VISIBLE_DEVICES=""

# 모델 로드
model = load_model('image_classification_model_transfer_learning.h5')

# 데이터 경로 설정
data_dir = '/mnt/splitter/datas/fortest'
data1_dir = os.path.join(data_dir, 'data1')
data2_dir = os.path.join(data_dir, 'data2')
class1_dir = os.path.join(data1_dir, 'class1')
class2_dir = os.path.join(data2_dir, 'class2')

# 이미지 크기 설정
img_width, img_height = 224, 224

# 클래스 디렉토리 리스트
class_dirs = [class1_dir, class2_dir]

# 정확도 계산을 위한 변수 초기화
total_images = 0
correct_predictions = 0

# 이미지 분류 함수
def classify_image(image_path, class_name, data_name):
    global total_images, correct_predictions

    # 이미지 로드 및 전처리
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # 이미지 예측
    prediction = model.predict(img_array)
    predicted_class = 'class1' if prediction[0][0] < 0.5 else 'class2'

    # 정확도 업데이트
    total_images += 1
    if predicted_class == class_name:
        correct_predictions += 1

    return f"Image: {os.path.basename(image_path)}, Actual Class: {data_name}/{class_name}, Predicted Class: {predicted_class}"

# 쓰레드 풀 생성
with ThreadPoolExecutor() as executor:
    futures = []

    # 각 클래스에 대한 예측 수행
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        data_name = os.path.basename(os.path.dirname(class_dir))
        print(f"Predictions for {data_name}/{class_name}:")

        # 클래스 디렉토리 내의 이미지 파일 리스트 가져오기
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 각 이미지에 대한 분류 작업을 쓰레드 풀에 제출
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            future = executor.submit(classify_image, image_path, class_name, data_name)
            futures.append(future)

        print("---")

    # 모든 분류 작업이 완료될 때까지 대기
    for future in futures:
        result = future.result()
        print(result)

# 정확도 계산 및 출력
accuracy = correct_predictions / total_images * 100
print(f"Accuracy: {accuracy:.2f}%")