from paddleocr import PaddleOCR
import re
import os
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor

# PaddleOCR 로그 비활성화
logger = logging.getLogger('ppocr')
logger.setLevel(logging.ERROR)

# PaddleOCR 객체 생성
ocr = PaddleOCR(lang="korean")

def check_business_license(file_path):
    print(f"Processing file: {file_path}")
    
    result = ocr.ocr(file_path, cls=False)
    ocr_result = result[0]

    if ocr_result is None:
        print(f"Skipping {file_path} due to OCR failure.")
        return

    korean_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣\s]+')

    korean_text = ""
    for res in ocr_result:
        text = res[1][0]
        korean_words = korean_pattern.findall(text)
        korean_text += ''.join(korean_words)

    if "사업자등록증" not in korean_text:
        print(f"{file_path} 파일은 '사업자등록증'이라는 글자가 없습니다.")
        move_to_folder(file_path, "1nonbusinesslicense")
    else:
        print(f"{file_path} 파일은 '사업자등록증'이라는 글자가 있습니다.")
        move_to_folder(file_path, "0businesslicense")
def move_to_folder(file_path, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(folder_name, file_name)

    if not os.path.exists(destination_path):
        shutil.move(file_path, destination_path)
        print(f"Moved {file_path} to {destination_path}")

def traverse_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        if root.startswith("./0businesslicense") or root.startswith("./1nonbusinesslicense"):
            continue
        for file in files:
            if file.lower().endswith(".jpg"):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

# 현재 디렉토리 경로
current_directory = os.getcwd()

# 현재 디렉토리와 하위 디렉토리의 모든 jpg 파일 경로 가져오기
file_list = traverse_directory(current_directory)

# 쓰레드 풀의 크기 설정
max_workers = 40  # 적절한 값으로 조정

# 쓰레드 풀 생성
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 각 파일에 대해 check_business_license 함수 실행
    executor.map(check_business_license, file_list)
    
