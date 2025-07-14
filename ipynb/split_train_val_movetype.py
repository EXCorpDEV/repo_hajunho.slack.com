import os
import shutil
import random
import sys
from datetime import datetime

# 프로그레스바 함수
def show_progress_bar(current, total, prefix="진행률", length=50):
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    sys.stdout.write(f'\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})')
    sys.stdout.flush()

print("🚀 Train/Validation 데이터 분할 시작!")
print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

base_dir = './'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
yolo_label_dir = os.path.join(base_dir, 'labels_yolo')

print("📁 출력 디렉토리 생성 중...")
# 출력 디렉토리
for split in ['train', 'val']:
    train_img_dir = os.path.join(images_dir, split)
    train_lbl_dir = os.path.join(labels_dir, split)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    print(f"   ✅ {split} 디렉토리 생성: {train_img_dir}, {train_lbl_dir}")

print()

print("🔍 이미지 폴더 검색 중...")
# 폴더 순회하며 이미지 & 라벨 분리
image_subdirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) 
                and not d.endswith('json') 
                and d not in ['images', 'labels', 'labels_yolo']]

print(f"📂 발견된 이미지 폴더: {len(image_subdirs)}개")
for folder in image_subdirs:
    print(f"   - {folder}")
print()

print("🔗 이미지-라벨 쌍 수집 중...")
image_label_pairs = []
total_folders = len(image_subdirs)
missing_labels = 0

# 초기 프로그레스바
show_progress_bar(0, total_folders, "📊 폴더 스캔")
print()

for folder_idx, folder in enumerate(image_subdirs, 1):
    image_folder_path = os.path.join(base_dir, folder)
    folder_pairs = 0
    folder_missing = 0
    
    print(f"\n🔄 [{folder_idx}/{total_folders}] 처리 중: {folder}")
    
    # 폴더 내 jpg 파일 목록
    jpg_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]
    print(f"   📸 이미지 파일: {len(jpg_files)}개")
    
    for file in jpg_files:
        image_path = os.path.join(image_folder_path, file)
        label_name = os.path.splitext(file)[0] + '.txt'
        label_path = os.path.join(yolo_label_dir, label_name)
        
        if os.path.exists(label_path):
            image_label_pairs.append((image_path, label_path))
            folder_pairs += 1
        else:
            folder_missing += 1
            missing_labels += 1
    
    print(f"   ✅ 매칭된 쌍: {folder_pairs}개")
    if folder_missing > 0:
        print(f"   ⚠️ 라벨 없는 이미지: {folder_missing}개")
    
    # 전체 진행률 업데이트
    show_progress_bar(folder_idx, total_folders, "📊 폴더 스캔")

print(f"\n\n📋 수집 완료!")
print(f"   - 총 이미지-라벨 쌍: {len(image_label_pairs)}개")
if missing_labels > 0:
    print(f"   - 라벨 없는 이미지: {missing_labels}개")
print()

print("🔀 데이터 셔플 및 분할 중...")
# 셔플 후 분할
random.shuffle(image_label_pairs)
split_idx = int(len(image_label_pairs) * 0.8)
train_set = image_label_pairs[:split_idx]
val_set = image_label_pairs[split_idx:]

print(f"   📊 Train: {len(train_set)}개 ({len(train_set)/len(image_label_pairs)*100:.1f}%)")
print(f"   📊 Val: {len(val_set)}개 ({len(val_set)/len(image_label_pairs)*100:.1f}%)")
print()

# 이동 함수 (진행률 포함)
def move_pairs_with_progress(pairs, img_out, label_out, set_name):
    print(f"📁 {set_name} 데이터 이동 중...")
    
    # 초기 프로그레스바
    show_progress_bar(0, len(pairs), f"🔄 {set_name} 이동")
    print()
    
    moved_images = 0
    moved_labels = 0
    move_errors = 0
    
    for idx, (img, lbl) in enumerate(pairs, 1):
        try:
            # 이미지 이동
            shutil.move(img, os.path.join(img_out, os.path.basename(img)))
            moved_images += 1
            
            # 라벨 이동
            shutil.move(lbl, os.path.join(label_out, os.path.basename(lbl)))
            moved_labels += 1
            
        except Exception as e:
            print(f"\n   ❌ 이동 실패: {os.path.basename(img)}, 오류: {e}")
            move_errors += 1
        
        # 프로그레스바 업데이트
        show_progress_bar(idx, len(pairs), f"🔄 {set_name} 이동")
    
    print(f"\n   ✅ {set_name} 완료 - 이미지: {moved_images}개, 라벨: {moved_labels}개")
    if move_errors > 0:
        print(f"   ⚠️ 이동 오류: {move_errors}개")
    print()

# 이동 실행
move_pairs_with_progress(train_set, 
                        os.path.join(images_dir, 'train'), 
                        os.path.join(labels_dir, 'train'), 
                        "Train")

move_pairs_with_progress(val_set, 
                        os.path.join(images_dir, 'val'), 
                        os.path.join(labels_dir, 'val'), 
                        "Validation")

print("=" * 60)
print("🎉 데이터 분할 완료!")
print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("📊 최종 통계:")
print(f"   - 총 데이터: {len(image_label_pairs)}개")
print(f"   - Train 세트: {len(train_set)}개 ({len(train_set)/len(image_label_pairs)*100:.1f}%)")
print(f"   - Validation 세트: {len(val_set)}개 ({len(val_set)/len(image_label_pairs)*100:.1f}%)")
if missing_labels > 0:
    print(f"   - 제외된 데이터: {missing_labels}개 (라벨 없음)")
print()
print(f"📁 출력 위치:")
print(f"   - Train 이미지: {os.path.join(images_dir, 'train')}")
print(f"   - Train 라벨: {os.path.join(labels_dir, 'train')}")
print(f"   - Val 이미지: {os.path.join(images_dir, 'val')}")
print(f"   - Val 라벨: {os.path.join(labels_dir, 'val')}")
print("🚀 YOLO 학습 데이터 준비 완료!")
