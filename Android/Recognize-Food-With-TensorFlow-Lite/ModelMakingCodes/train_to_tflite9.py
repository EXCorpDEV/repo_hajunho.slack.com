import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import gc

# Lambda layer 대신 사용할 커스텀 레이어
class PreprocessingLayer(tf.keras.layers.Layer):
    """전처리 레이어 (Lambda 대신)"""
    def __init__(self, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return applications.efficientnet.preprocess_input(inputs)
    
    def get_config(self):
        return super(PreprocessingLayer, self).get_config()

# 메모리 최적화 설정
def configure_memory():
    """메모리 최적화 설정"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 메모리 설정 오류: {e}")
    
    # 혼합 정밀도 사용
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision 활성화됨")

def create_optimized_dataset(data_dir, validation_split=0.2, subset="training", 
                           seed=123, image_size=(224, 224), batch_size=16):
    """메모리 최적화된 데이터셋 생성"""
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset=subset,
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            crop_to_aspect_ratio=True,
            interpolation='bilinear'
        )
        
        # 에러 무시
        dataset = dataset.ignore_errors()
        
        # float16으로 변환
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float16), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
        
    except Exception as e:
        print(f"데이터셋 생성 실패: {e}")
        return None

def create_fixed_model(input_shape, num_classes):
    """Lambda layer 없는 안전한 모델"""
    
    # EfficientNetB0 사용 (가벼움)
    base_model = applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # 데이터 증강
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")
    
    # Lambda 없는 모델 구조
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # 데이터 증강
    x = data_augmentation(inputs)
    
    # 전처리 (Lambda 대신 커스텀 레이어)
    x = PreprocessingLayer(name="preprocessing")(x)
    
    # 백본 모델
    x = base_model(x, training=False)
    
    # 분류 헤드
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', 
                          dtype='float32', name="predictions")(x)
    
    model = models.Model(inputs, outputs, name="food_classifier")
    
    return model

def main():
    print("=== Lambda 문제 해결된 훈련 스크립트 ===")
    
    # 메모리 최적화
    configure_memory()
    
    # 설정
    data_dir = "./"
    batch_size = 16  # 적당한 크기
    img_height = 224
    img_width = 224
    epochs = 30
    
    print(f"배치 사이즈: {batch_size}")
    print(f"이미지 크기: {img_height}x{img_width}")
    
    # 클래스 정보 확인
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
    
    # 메모리 정리
    del temp_ds
    gc.collect()
    
    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    
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
    
    # 개선된 모델 생성
    print("\nLambda 없는 모델 생성 중...")
    model = create_fixed_model(
        input_shape=(img_height, img_width, 3),
        num_classes=len(class_names)
    )
    
    print(f"모델 파라미터 수: {model.count_params():,}")
    
    # 모델 컴파일
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
            patience=7,
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
        # 체크포인트 저장
        tf.keras.callbacks.ModelCheckpoint(
            filepath='food_classifier_checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 첫 배치 테스트
    print("\n첫 배치 테스트 중...")
    try:
        for batch in train_ds.take(1):
            print(f"배치 모양: {batch[0].shape}")
            test_pred = model(batch[0])
            print(f"예측 모양: {test_pred.shape}")
            print("첫 배치 테스트 성공!")
            break
    except Exception as e:
        print(f"첫 배치 테스트 실패: {e}")
        return
    
    # 훈련 시작
    print("\n훈련 시작...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 모델 저장
        print("\n최종 모델 저장 중...")
        model_save_path = "food_classifier_fixed.keras"
        model.save(model_save_path)
        print(f"모델 저장됨: {model_save_path}")
        
        # 모델 크기 확인
        size_mb = os.path.getsize(model_save_path) / (1024*1024)
        print(f"모델 크기: {size_mb:.2f} MB")
        
        if size_mb < 100:
            print("⚠️ 모델 크기가 너무 작습니다. 훈련이 제대로 완료되지 않았을 수 있습니다.")
        else:
            print("✅ 정상적인 모델 크기입니다.")
        
        # TFLite 변환 시도
        print("\nTFLite 변환 시도...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            with open("food_classifier_fixed.tflite", "wb") as f:
                f.write(tflite_model)
            
            tflite_size_mb = len(tflite_model) / (1024*1024)
            print(f"✅ TFLite 변환 성공: {tflite_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ TFLite 변환 실패: {e}")
        
        print("\n훈련 완료!")
        
    except Exception as e:
        print(f"훈련 중 오류: {e}")
        print("메모리 부족일 가능성이 높습니다.")

if __name__ == "__main__":
    main()

