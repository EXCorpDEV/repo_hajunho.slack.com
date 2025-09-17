import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import gc

# 메모리 최적화 설정
def configure_memory():
    """메모리 최적화 설정"""
    # GPU 메모리 증가 허용
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 메모리 설정 오류: {e}")
    
    # 혼합 정밀도 사용 (메모리 절약)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision 활성화됨")

def create_optimized_dataset(data_dir, validation_split=0.2, subset="training", 
                           seed=123, image_size=(224, 224), batch_size=8):
    """메모리 최적화된 데이터셋 생성"""
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            crop_to_aspect_ratio=True,
            interpolation='bilinear'
        )
        
        # 에러 무시
        dataset = dataset.ignore_errors()
        
        # 메모리 최적화
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float16), y),  # float16 사용
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
        
    except Exception as e:
        print(f"데이터셋 생성 실패: {e}")
        return None

def create_lightweight_model(input_shape, num_classes):
    """경량화된 모델 생성"""
    # EfficientNetB0 사용 (B3보다 가벼움)
    base_model = applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # 간단한 데이터 증강
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),  # 회전 각도 줄임
    ])
    
    # 경량화된 모델 구조
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        data_augmentation,
        layers.Lambda(lambda x: applications.efficientnet.preprocess_input(x)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float32')  # 출력은 float32
    ])
    
    return model

def main():
    print("=== 메모리 최적화 훈련 스크립트 ===")
    
    # 메모리 최적화 설정
    configure_memory()
    
    # 설정 (메모리 절약)
    data_dir = "./"
    batch_size = 8  # 작은 배치 사이즈
    img_height = 224  # 작은 이미지 크기
    img_width = 224
    epochs = 50
    
    print(f"배치 사이즈: {batch_size}")
    print(f"이미지 크기: {img_height}x{img_width}")
    
    # 클래스 정보 얻기
    print("\n클래스 정보 확인 중...")
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    class_names = temp_ds.class_names
    print(f"클래스 수: {len(class_names)}")
    print("클래스 샘플:", class_names[:10])
    
    # 메모리 정리
    del temp_ds
    gc.collect()
    
    # 최적화된 데이터셋 생성
    print("\n최적화된 데이터셋 생성 중...")
    
    train_ds = create_optimized_dataset(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    val_ds = create_optimized_dataset(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    
    if train_ds is None or val_ds is None:
        print("데이터셋 생성 실패")
        return
    
    # 데이터셋 최적화
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    # 경량화된 모델 생성
    print("\n경량화된 모델 생성 중...")
    model = create_lightweight_model(
        input_shape=(img_height, img_width, 3),
        num_classes=len(class_names)
    )
    
    # 모델 컴파일 (mixed precision 고려)
    optimizer = optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # 더 빠른 조기 종료
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        # 메모리 사용량 모니터링
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=False,
            write_images=False
        )
    ]
    
    # 훈련 시작
    print("\n훈련 시작...")
    print("메모리 사용량을 모니터링하며 훈련합니다.")
    
    try:
        # 몇 개 배치로 테스트
        print("첫 배치 테스트 중...")
        for batch in train_ds.take(1):
            test_pred = model(batch[0])
            print(f"첫 배치 테스트 성공: {test_pred.shape}")
            break
        
        # 실제 훈련
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 모델 저장
        print("\n모델 저장 중...")
        model_save_path = "food_classifier_optimized.keras"
        model.save(model_save_path)
        print(f"모델 저장됨: {model_save_path}")
        
        # TFLite 변환
        print("TFLite 변환 중...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        with open("food_classifier_optimized.tflite", "wb") as f:
            f.write(tflite_model)
        print("TFLite 모델 저장됨: food_classifier_optimized.tflite")
        
        print("\n훈련 완료!")
        
    except Exception as e:
        print(f"훈련 중 오류: {e}")
        print("\n메모리 부족일 가능성이 높습니다.")
        print("배치 사이즈를 더 줄이거나 이미지 크기를 줄여보세요.")
        
        # 메모리 정리
        gc.collect()

if __name__ == "__main__":
    main()

