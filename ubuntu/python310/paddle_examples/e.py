from paddleocr import PaddleOCR
import re
import os

def check_business_license(file_path):
    ocr = PaddleOCR(lang="korean")
    result = ocr.ocr(file_path, cls=False)
    ocr_result = result[0]

    korean_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣\s]+')

    korean_text = ""
    for res in ocr_result:
        text = res[1][0]
        korean_words = korean_pattern.findall(text)
        korean_text += ''.join(korean_words)

    if "사업자등록증" not in korean_text:
        print(f"{file_path} 파일은 '사업자등록증'이라는 글자가 없습니다.")

def traverse_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                file_path = os.path.join(root, file)
                check_business_license(file_path)

# 현재 디렉토리 경로
current_directory = os.getcwd()

# 현재 디렉토리와 하위 디렉토리의 모든 jpg 파일 검사
traverse_directory(current_directory)

