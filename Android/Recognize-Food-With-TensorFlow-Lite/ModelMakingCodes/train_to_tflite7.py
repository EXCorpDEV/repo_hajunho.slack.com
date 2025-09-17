import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import os

# 빠른 데이터셋 필터링 함수
def filter_corrupted_images(ds):
    """런타임에 손상된 이미지 필터링"""
    def is_valid_image(image, label):
        try:
            # 이미지 텐서가 유효한지 확인
            tf.debugging.assert_greater(tf.size(image), 0)
            tf.debugging.assert_finite(image)
            return True
        except:
            return False
    
    # 손상된 이미지 필터링
    filtered_ds = ds.filter(lambda x, y: is_valid_image(x, y))
    return filtered_ds

# 안전한 데이터셋 생성 함수
def create_safe_dataset_with_error_handling(data_dir, validation_split=0.2, subset="training", 
                                          seed=123, image_size=(300, 300), batch_size=32):
    """에러 처리가 강화된 안전한 데이터셋 생성"""
    try:
        # 데이터셋 생성
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            crop_to_aspect_ratio=True,  # 이미지 크롭하여 비율 맞추기
            interpolation='bilinear'
        )
        
        # 에러가 발생하는 이미지들을 건너뛰도록 설정
        dataset = dataset.ignore_errors()  # 업데이트된 방식 사용
        
        return dataset
        
    except Exception as e:
        print(f"데이터셋 생성 실패: {e}")
        return None

def main():
    print("=== 비상용 훈련 스크립트 ===")
    print("이 스크립트는 손상된 이미지를 자동으로 건너뛰고 훈련을 진행합니다.")
    
    # 설정
    data_dir = "./"
    batch_size = 32
    img_height = 300
    img_width = 300
    epochs = 50

    # 데이터셋 생성 (에러 처리 포함)
    print("\n데이터셋 생성 중...")
    
    # 먼저 클래스 정보를 얻기 위해 기본 데이터셋 생성
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    # 클래스 정보 저장
    class_names = temp_ds.class_names
    print(f"클래스 목록: {class_names}")
    print(f"클래스 수: {len(class_names)}")
    
    # 이제 에러 처리가 포함된 데이터셋 생성
    train_ds = create_safe_dataset_with_error_handling(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    val_ds = create_safe_dataset_with_error_handling(
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
    
    # 데이터셋 최적화
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # 데이터 증강
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # 모델 생성
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
    
    # 모델 컴파일
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # 훈련 시작
    print("\n훈련 시작...")
    print("※ 손상된 이미지는 자동으로 건너뛰어집니다.")
    
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 모델 저장
        print("\n모델 저장 중...")
        model_save_path = "my_food_classifier.keras"
        model.save(model_save_path)
        print(f"Keras 모델 저장됨: {model_save_path}")
        
        # TFLite 변환
        print("TFLite 변환 중...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("my_food_classifier.tflite", "wb") as f:
            f.write(tflite_model)
        print("TFLite 모델 저장됨: my_food_classifier.tflite")
        
        # 훈련 결과 출력
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\n최종 훈련 정확도: {final_train_acc:.4f}")
        print(f"최종 검증 정확도: {final_val_acc:.4f}")
        
        print("\n훈련 완료!")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        print("다시 시도해보세요.")

if __name__ == "__main__":
    main()

