import os
import json
import sys
from datetime import datetime

# 프로그레스바 함수
def show_progress_bar(current, total, prefix="진행률", length=50):
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    sys.stdout.write(f'\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})')
    sys.stdout.flush()

print("🚀 YOLO 라벨 변환 시작!")
print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 전체 데이터 루트 디렉토리
root_dir = './'

# class_name → class_id 매핑 (폴더명 기준)
print("📂 클래스 폴더 검색 중...")
class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.endswith('json')])
class_map = {name: idx for idx, name in enumerate(class_names)}

print(f"✅ 총 {len(class_names)}개 클래스 발견:")
for name, idx in class_map.items():
    print(f"   {idx:2d}: {name}")
print()

# 출력 디렉토리
output_label_dir = os.path.join(root_dir, 'labels_yolo')
os.makedirs(output_label_dir, exist_ok=True)
print(f"📁 출력 디렉토리: {output_label_dir}")
print()

# JSON 폴더 찾기
json_folders = [folder for folder in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, folder)) and folder.endswith('json')]

print(f"📋 처리할 JSON 폴더: {len(json_folders)}개")
for folder in json_folders:
    print(f"   - {folder}")
print()

# 통계 변수
total_files = 0
processed_files = 0
total_objects = 0
error_files = 0
current_file_count = 0

# 전체 파일 수 미리 계산
print("📊 전체 파일 수 계산 중...")
for folder in json_folders:
    folder_path = os.path.join(root_dir, folder)
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    total_files += len(json_files)

print(f"📋 전체 처리할 파일: {total_files}개")
print()

# 초기 프로그레스바 표시
show_progress_bar(0, total_files, "🔄 전체 진행률")
print()  # 줄바꿈

# JSON 폴더 순회
for folder_idx, folder in enumerate(json_folders, 1):
    folder_path = os.path.join(root_dir, folder)
    
    print(f"🔄 [{folder_idx}/{len(json_folders)}] 처리 중: {folder}")
    
    # 대응하는 class_id 찾기 (ex: '애호박 json' → '애호박')
    class_base = folder.replace(' json', '')
    class_id = class_map.get(class_base)
    
    if class_id is None:
        print(f"   ❌ class_id 찾을 수 없음: {folder}")
        continue
    
    # 폴더 내 JSON 파일 목록
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    print(f"   📌 클래스: {class_base} (ID: {class_id})")
    print(f"   📄 JSON 파일 수: {len(json_files)}개")
    
    folder_objects = 0
    folder_errors = 0
    
    for file_idx, file in enumerate(json_files, 1):
        current_file_count += 1
        
        json_path = os.path.join(folder_path, file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"\n      ❌ JSON 파싱 실패: {file}, 오류: {e}")
            error_files += 1
            folder_errors += 1
            show_progress_bar(current_file_count, total_files, "🔄 전체 진행률")
            continue
        
        out_name = os.path.splitext(file)[0] + '.txt'
        out_path = os.path.join(output_label_dir, out_name)
        
        file_objects = 0
        try:
            with open(out_path, 'w') as out_file:
                for obj in data:
                    try:
                        x, y = map(float, obj['Point(x,y)'].split(','))
                        w = float(obj['W'])
                        h = float(obj['H'])
                        out_file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        file_objects += 1
                    except Exception as e:
                        print(f"\n      ⚠️ 객체 변환 오류 (파일: {file}): {e}")
                        show_progress_bar(current_file_count, total_files, "🔄 전체 진행률")
                        continue
            
            processed_files += 1
            folder_objects += file_objects
            total_objects += file_objects
            
        except Exception as e:
            print(f"\n      ❌ 파일 쓰기 실패: {file}, 오류: {e}")
            error_files += 1
            folder_errors += 1
        
        # 전체 프로그레스바 업데이트
        show_progress_bar(current_file_count, total_files, "🔄 전체 진행률")
    
    
    print(f"\n   ✅ 완료 - 객체 수: {folder_objects}개, 오류: {folder_errors}개")

print("\n" + "=" * 60)
print("🎉 라벨 변환 완료!")
print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("📊 최종 통계:")
print(f"   - 전체 JSON 파일: {total_files}개")
print(f"   - 성공적으로 처리: {processed_files}개")
print(f"   - 실패한 파일: {error_files}개")
print(f"   - 변환된 객체 수: {total_objects}개")
print(f"   - 성공률: {processed_files/total_files*100 if total_files > 0 else 0:.1f}%")
print()
print(f"📁 출력 위치: {output_label_dir}")
print("🚀 YOLO 학습 준비 완료!")
