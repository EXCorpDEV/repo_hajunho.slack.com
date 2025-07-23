import os
import warnings
import logging

# TensorFlow 로그 레벨 설정 (에러 메시지 줄이기)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import tensorflow as tf
import shutil
from pathlib import Path
import gc

# 학습할 클래스 목록
TARGET_CLASSES = list(set([
    "닭갈비", "훈제오리", "콩나물", "찹쌀떡", "쌀국수", "비빔밥", "연어초밥", "장어초밥",
    "순대볶음", "참치샌드위치", "김치찌개", "부대찌개", "순대", "초콜릿", "순살치킨",
    "에그타르트", "치킨너겟", "고등어구이", "등심스테이크", "삼겹살구이", "조미김",
    "미역국", "삼계탕", "어묵국", "배추김치", "오이김치", "숙주나물", "군만두", "떡국",
    "라면", "만두", "비빔냉면", "짜장면", "짬뽕", "김밥", "비빔밥", "알밥", "오므라이스",
    "육회비빔밥", "떡볶이", "잡채", "닭가슴살샐러드", "육회", "마늘장아찌", "떡갈비",
    "스크램블드에그", "채소죽", "된장찌개", "부대찌개", "순두부찌개", "달걀찜", "닭찜",
    "돼지갈비찜", "순대", "아귀찜", "족발", "감자튀김", "오징어튀김", "치킨너겟",
    "후라이드치킨", "까르보나라", "츄러스", "초코아이스크림"
]))

def filter_classes(source_dir, target_dir):
    """지정된 클래스만 필터링"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 타겟 디렉토리 생성
    target_path.mkdir(exist_ok=True)
    
    copied_classes = []
    skipped_classes = []
    
    print(f"\n=== 클래스 필터링 시작 ===")
    print(f"원본 디렉토리: {source_dir}")
    print(f"필터링된 디렉토리: {target_dir}")
    print(f"타겟 클래스 수: {len(TARGET_CLASSES)}")
    
    # 원본 디렉토리의 모든 폴더 확인
    for folder in source_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            
            if folder_name in TARGET_CLASSES:
                # 타겟 클래스에 포함되면 복사
                target_class_dir = target_path / folder_name
                
                if target_class_dir.exists():
                    shutil.rmtree(target_class_dir)
                
                shutil.copytree(folder, target_class_dir)
                
                # 이미지 개수 세기
                image_count = len([f for f in target_class_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                
                copied_classes.append((folder_name, image_count))
                print(f"✅ {folder_name}: {image_count}개 이미지")
                
            else:
                skipped_classes.append(folder_name)
    
    print(f"\n복사된 클래스: {len(copied_classes)}개")
    print(f"건너뛴 클래스: {len(skipped_classes)}개")
    
    if skipped_classes:
        print(f"건너뛴 클래스들: {', '.join(skipped_classes[:10])}{'...' if len(skipped_classes) > 10 else ''}")
    
    # 누락된 타겟 클래스 확인
    copied_class_names = [name for name, _ in copied_classes]
    missing_classes = [cls for cls in TARGET_CLASSES if cls not in copied_class_names]
    
    if missing_classes:
        print(f"\n⚠️ 누락된 타겟 클래스: {len(missing_classes)}개")
        print(f"누락된 클래스들: {', '.join(missing_classes[:10])}{'...' if len(missing_classes) > 10 else ''}")
    
    return target_dir, len(copied_classes)

def main():
    print("=== 극단적 메모리 절약 훈련 ===")
    print(f"타겟 클래스 수: {len(TARGET_CLASSES)}")
    
    # 극도로 메모리 절약하는 설정
    original_data_dir = './'
    filtered_data_dir = './filtered_classes'
    batch_size = 8  # 32 → 8로 대폭 감소
    img_height = 128  # 224 → 128로 감소
    img_width = 128
    epochs = 5  # 10 → 5로 감소
    model_save_path = 'ultra_small_food_classifier.keras'
    tflite_save_path = 'ultra_small_food_classifier.tflite'
    class_names_file = 'ultra_small_class_names.txt'
    
    print(f"\n⚠️ 극단적 메모리 절약 설정:")
    print(f"- 배치 사이즈: {batch_size} (원본 32에서 감소)")
    print(f"- 이미지 크기: {img_height}x{img_width} (원본 224x224에서 감소)")
    print(f"- 에폭 수: {epochs} (원본 10에서 감소)")
    
    # 클래스 필터링
    filtered_dir, actual_class_count = filter_classes(original_data_dir, filtered_data_dir)
    
    if actual_class_count == 0:
        print("❌ 필터링된 클래스가 없습니다.")
        return
    
    print(f"\n실제 사용할 클래스 수: {actual_class_count}")
    
    # 데이터셋 생성 (작은 크기로)
    print("\n작은 크기 데이터셋 생성 중...")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        filtered_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_height, img_width),  # 작은 크기
        batch_size=batch_size  # 작은 배치
    )
    
    # 클래스 이름 확인
    class_names = train_ds.class_names
    print(f"\n실제 로드된 클래스 수: {len(class_names)}")
    
    # 클래스 이름 저장
    with open(class_names_file, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"클래스 목록 저장: {class_names_file}")
    
    # 극도로 작은 모델 생성
    print("\n극도로 작은 MobileNetV2 모델 생성 중...")
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # 모델 크기를 35%로 축소
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dropout 추가로 오버피팅 방지
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"모델 파라미터 수: {model.count_params():,}")
    
    # 메모리 정리
    gc.collect()
    
    # 안전한 첫 배치 테스트 (더 작은 배치로)
    print("\n안전한 첫 배치 테스트 중...")
    try:
        for batch in train_ds.take(1):
            print(f"배치 모양: {batch[0].shape}")
            
            # 더 작은 서브배치로 테스트
            mini_batch = batch[0][:4]  # 8개 중에서 4개만
            test_pred = model(mini_batch)
            print(f"예측 모양: {test_pred.shape}")
            print("✅ 첫 배치 테스트 성공!")
            
            # 메모리 정리
            del mini_batch, test_pred
            gc.collect()
            break
            
    except Exception as e:
        print(f"❌ 첫 배치 테스트 실패: {e}")
        print("메모리가 여전히 부족합니다. 시스템 메모리를 확인하세요.")
        return
    
    # 학습 (조기 종료 콜백 추가)
    print("\n안전한 학습 시작...")
    
    # 메모리 부족을 대비한 콜백
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=2,
            restore_best_weights=True
        )
    ]
    
    try:
        # 메모리 정리
        gc.collect()
        
        history = model.fit(
            train_ds, 
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 모델 저장
        print(f"\n모델 저장 중: {model_save_path}")
        model.save(model_save_path)
        
        # 모델 크기 확인
        size_mb = os.path.getsize(model_save_path) / (1024*1024)
        print(f"모델 크기: {size_mb:.2f} MB")
        
        # TFLite 변환
        print(f"\nTFLite 변환 중: {tflite_save_path}")
        converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size_mb = len(tflite_model) / (1024*1024)
        print(f"✅ TFLite 변환 성공: {tflite_size_mb:.2f} MB")
        
        print(f"\n🎉 극단적 메모리 절약 훈련 완료!")
        print(f"- 클래스 수: {len(class_names)}")
        print(f"- 모델 파라미터: {model.count_params():,}")
        print(f"- 모델 파일: {model_save_path}")
        print(f"- TFLite 파일: {tflite_save_path}")
        print(f"- 클래스 목록: {class_names_file}")
        
        print(f"\n💡 성공하면 다음 단계에서 점진적으로 설정을 늘려갈 수 있습니다:")
        print(f"   1. 배치 사이즈: 8 → 16 → 32")
        print(f"   2. 이미지 크기: 128 → 160 → 224")
        print(f"   3. 에폭 수: 5 → 10")
        
    except Exception as e:
        print(f"❌ 학습 중 오류: {e}")
        print("시스템 메모리가 심각하게 부족합니다.")
        print("다른 프로세스를 종료하거나 더 많은 메모리가 있는 시스템에서 실행하세요.")

if __name__ == "__main__":
    main()
