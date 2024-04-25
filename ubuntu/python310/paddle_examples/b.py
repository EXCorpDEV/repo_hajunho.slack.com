from paddleocr import PaddleOCR
import re

ocr = PaddleOCR(lang="korean")

img_path = "a.jpg"

result = ocr.ocr(img_path, cls=False)

ocr_result = result[0]

# 한글만 추출하기 위한 정규식 패턴
korean_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]+')

korean_text = ""
for res in ocr_result:
    text = res[1][0]
    korean_words = korean_pattern.findall(text)
    korean_text += ' '.join(korean_words) + ' '

print(korean_text.strip())
