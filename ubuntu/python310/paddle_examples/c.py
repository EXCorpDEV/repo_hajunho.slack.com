from paddleocr import PaddleOCR
import re
import sys

ocr = PaddleOCR(lang="korean")

img_path = "a.jpg"

result = ocr.ocr(img_path, cls=False)

ocr_result = result[0]

korean_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]+')

korean_text = ""
for res in ocr_result:
    text = res[1][0]
    korean_words = korean_pattern.findall(text)
    korean_text += ' '.join(korean_words) + ' '

korean_text = korean_text.strip()

if "사업자등록증" in korean_text:
    print("사업자등록증 글자가 있습니다.")
    print("Extracted Text:")
    print(korean_text)
    sys.exit(0)  # 상태 코드 0으로 프로그램 종료
else:
    print("사업자등록증 글자가 없습니다.")
    sys.exit(1)  # 상태 코드 1로 프로그램 종료

