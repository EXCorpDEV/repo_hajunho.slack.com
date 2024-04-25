from paddleocr import PaddleOCR
import re
import os
import logging
import shutil
from multiprocessing import Pool

# PaddleOCR 로그 비활성화
logger = logging.getLogger('ppocr')
logger.setLevel(logging.ERROR)

# 분류 기준이 되는 텍스트 리스트
classification_texts = ["사업자등록증", "면세사업자"]

def check_business_license(file_path):
    try:
        ocr = PaddleOCR(lang="korean")
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

        if any(text in korean_text for text in classification_texts):
            print(f"{file_path} 파일에는 분류 기준 텍스트 중 하나가 있습니다.")
            move_to_folder(file_path, "0businesslicense")
        else:
            print(f"{file_path} 파일에는 분류 기준 텍스트가 없습니다.")
            move_to_folder(file_path, "1nonbusinesslicense")
    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(f"Error message: {str(e)}")

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

# 프로세스 수 지정
#num_processes = 4

# 프로세스 풀 생성
#with Pool(processes=num_processes) as pool:
with Pool() as pool:
    # 각 파일에 대해 check_business_license 함수 실행
    pool.map(check_business_license, file_list)
