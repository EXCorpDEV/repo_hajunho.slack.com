import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import os
from PIL import Image
import numpy as np

# 데이터셋 검증 및 정리 함수
def validate_and_clean_dataset(data_dir):
    """손상된 이미지 파일 검사 및 정리"""
    corrupted_files = []
    valid_files = []
    total_files = 0
    
    print("데이터셋 검증 중...")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                total_files += 1
                file_path = os.path.join(root, file)
                
                try:
                    # 파일 크기 확인
                    if os.path.getsize(file_path) == 0:
                        print(f"빈 파일 발견: {file_path}")
                        corrupted_files.append(file_path)
                        continue
                    
                    # PIL로 이미지 무결성 확인
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    # 실제 로딩 확인
                    with Image.open(file_path) as img:
                        img.load()
                        # RGB 변환 가능 여부 확인
                        if img.mode not in ['RGB', 'RGBA', 'L']:
                            img.convert('RGB')
                    
                    # TensorFlow로 디코딩 확인
                    try:
                        image_string = tf.io.read_file(file_path)
                        tf.io.decode_image(image_string, channels=3)
                        valid_files.append(file_path)
                    except:
                        print(f"TensorFlow 디코딩 실패: {file_path}")
                        corrupted_files.append(file_path)
                        
                except Exception as e:
                    print(f"손상된 파일 발견: {file_path} - {e}")
                    corrupted_files.append(file_path)
    
    print(f"총 파일 수: {total_files}")
    print(f"유효한 파일 수: {len(valid_files)}")
    print(f"손상된 파일 수: {len(corrupted_files)}")
    
    return valid_files, corrupted_files

# 손상된 파일 처리 함수
def handle_corrupted_files(corrupted_files, action='move'):
    """손상된 파일 처리 (이동 또는 삭제)"""
    if not corrupted_files:
        return
        
    if action == 'move':
        corrupted_dir = "./corrupted_files"
        os.makedirs(corrupted_dir, exist_ok=True)
        
        for file_path in corrupted_files:
            try:
                filename = os.path.basename(file_path)
                new_path = os.path.join(corrupted_dir, filename)
                # 파일명 중복 방지
                counter = 1
                while os.path.exists(new_path):
                    name, ext = os.path.splitext(filename)
                    new_path = os.path.join(corrupted_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                os.rename(file_path, new_path)
                print(f"파일 이동: {file_path} -> {new_path}")
            except Exception as e:
                print(f"파일 이동 실패 {file_path}: {e}")
                
    elif action == 'delete':
        for file_path in corrupted_files:
            try:
                os.remove(file_path)
                print(f"파일 삭제: {file_path}")
            except Exception as e:
                print(f"파일 삭제 실패 {file_path}: {e}")

# 안전한 데이터셋 생성 함수
def create_safe_dataset(data_dir, validation_split=0.2, subset="training", seed=123, 
                       image_size=(300, 300), batch_size=32):
    """안전한 데이터셋 생성"""
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )
        return dataset
    except Exception as e:
        print(f"데이터셋 생성 실패: {e}")
        return None

# 메인 코드
def main():
    # 설정
    data_dir = "./"
    batch_size = 32
    img_height = 300
    img_width = 300
    epochs = 50

    # 1. 데이터셋 검증 및 정리
    valid_files, corrupted_files = validate_and_clean_dataset(data_dir)
    
    if corrupted_files:
        print(f"\n손상된 파일들:")
        for file in corrupted_files[:10]:  # 처음 10개만 표시
            print(f"  - {file}")
        if len(corrupted_files) > 10:
            print(f"  ... 그 외 {len(corrupted_files) - 10}개")
        
        # 손상된 파일 처리 (이동 또는 삭제)
        user_choice = input("\n손상된 파일을 어떻게 처리하시겠습니까? (move/delete/skip): ").lower()
        
        if user_choice == 'move':
            handle_corrupted_files(corrupted_files, 'move')
        elif user_choice == 'delete':
            handle_corrupted_files(corrupted_files, 'delete')
        elif user_choice == 'skip':
            print("손상된 파일을 그대로 둡니다. 훈련 중 에러가 발생할 수 있습니다.")
        else:
            print("잘못된 선택입니다. 파일을 이동합니다.")
            handle_corrupted_files(corrupted_files, 'move')

    # 2. 안전한 데이터셋 생성
    print("\n데이터셋 생성 중...")
    
    train_ds = create_safe_dataset(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    val_ds = create_safe_dataset(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    if train_ds is None or val_ds is None:
        print("데이터셋 생성에 실패했습니다.")
        return
    
    # 3. 클래스 정보 출력
    class_names = train_ds.class_names
    print(f"클래스 목록: {class_names}")
    print(f"클래스 수: {len(class_names)}")
    
    # 4. 데이터셋 최적화
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    # 5. 데이터 증강
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # 6. 모델 생성
    print("\n모델 생성 중...")
    base_model = applications.EfficientNetB3(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    preprocess_input = applications.efficientnet.preprocess_input
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    # 7. 모델 컴파일
    model.compile(
        optimizer=optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 8. 콜백 설정
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # 9. 훈련 시작
    print("\n훈련 시작...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )
        
        # 10. 모델 저장
        print("\n모델 저장 중...")
        model_save_path = "my_food_classifier.keras"
        model.save(model_save_path)
        print(f"Keras 모델 저장됨: {model_save_path}")
        
        # 11. TFLite 변환
        print("TFLite 변환 중...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open("my_food_classifier.tflite", "wb") as f:
            f.write(tflite_model)
        print("TFLite 모델 저장됨: my_food_classifier.tflite")
        
        print("\n훈련 완료!")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        print("데이터셋을 다시 확인하고 손상된 파일을 모두 제거한 후 다시 시도해주세요.")

if __name__ == "__main__":
    main()

