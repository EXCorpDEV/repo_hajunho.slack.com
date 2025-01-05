from fpdf import FPDF
from PyPDF2 import PdfWriter, PdfReader

# 파일 경로
txt_file = "t.txt"
pdf_file = "e.pdf"
watermark_pdf = "watermark.pdf"
final_pdf = "HJH_restricted.pdf"

# 텍스트를 안전하게 변환하는 함수
def safe_text(text):
    try:
        return text.encode('latin-1', 'replace').decode('latin-1')  # 변환 불가한 문자는 '?'로 대체
    except UnicodeEncodeError:
        return "Encoding Error"

# PDF 생성 클래스
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'CONFIDENTIAL - HJH', 0, 1, 'C')  # 워터마크 텍스트

pdf = PDF()
pdf.add_page()
with open(txt_file, 'r', encoding='utf-8') as file:
    for line in file:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=safe_text(line.strip()), ln=True)  # 안전한 텍스트 처리
pdf.output(pdf_file)

# 워터마크 PDF 생성
watermark_pdf_creator = PDF()
watermark_pdf_creator.add_page()
watermark_pdf_creator.set_font('Arial', 'B', 50)
watermark_pdf_creator.set_text_color(200, 200, 200)  # 연한 회색

# 페이지 크기 가져오기
page_width = watermark_pdf_creator.w  # 페이지 너비
page_height = watermark_pdf_creator.h  # 페이지 높이
x_center = page_width / 2
y_center = page_height / 2

watermark_pdf_creator.set_xy(x_center, y_center)  # 페이지 중앙으로 이동
watermark_pdf_creator.rotate(45)  # 45도 회전
watermark_pdf_creator.cell(0, 0, 'CONFIDENTIAL - HJH', align='C')  # 중앙 정렬 텍스트
watermark_pdf_creator.output(watermark_pdf)

# 워터마크 추가
reader = PdfReader(pdf_file)
watermark_reader = PdfReader(watermark_pdf)
writer = PdfWriter()

for page in reader.pages:
    watermark_page = watermark_reader.pages[0]  # 워터마크 페이지
    page.merge_page(watermark_page)
    writer.add_page(page)

# 암호화 적용 (user_password와 owner_password 설정)
writer.encrypt(user_password="", owner_password="owner_password")  # 사용자 암호는 빈 문자열, 소유자 암호는 설정

with open(final_pdf, "wb") as f:
    writer.write(f)

print(f"Restricted PDF saved: {final_pdf}")
