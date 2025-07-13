# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
import io
import uvicorn

app = FastAPI(title="PDF 텍스트 추출기", description="PDF 파일에서 텍스트를 추출하는 웹 애플리케이션")

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF 텍스트 추출기</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            margin-bottom: 30px;
            padding: 20px;
            border: 2px dashed #ddd;
            border-radius: 8px;
            text-align: center;
        }
        .file-input {
            margin: 10px 0;
        }
        input[type="file"] {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result-section {
            margin-top: 30px;
        }
        .result-text {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        .success {
            color: #155724;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        .info {
            color: #0c5460;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 PDF 텍스트 추출기</h1>

        <div class="upload-section">
            <h3>PDF 파일 업로드</h3>
            <div class="file-input">
                <input type="file" id="pdfFile" accept=".pdf" />
            </div>
            <button onclick="extractText()" id="extractBtn">텍스트 추출</button>
        </div>

        <div id="messages"></div>

        <div class="result-section" id="resultSection" style="display: none;">
            <h3>추출된 텍스트:</h3>
            <div class="result-text" id="resultText"></div>
            <button onclick="copyText()" style="margin-top: 10px;">📋 클립보드에 복사</button>
            <button onclick="downloadText()" style="margin-top: 10px;">💾 텍스트 파일로 다운로드</button>
        </div>
    </div>

    <script>
        let extractedText = '';

        function showMessage(message, type = 'info') {
            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML = `<div class="${type}">${message}</div>`;
            setTimeout(() => {
                messagesDiv.innerHTML = '';
            }, 5000);
        }

        async function extractText() {
            const fileInput = document.getElementById('pdfFile');
            const extractBtn = document.getElementById('extractBtn');
            const resultSection = document.getElementById('resultSection');
            const resultText = document.getElementById('resultText');

            if (!fileInput.files[0]) {
                showMessage('PDF 파일을 선택해주세요.', 'error');
                return;
            }

            const file = fileInput.files[0];

            // 파일 크기 체크 (10MB 제한)
            if (file.size > 10 * 1024 * 1024) {
                showMessage('파일 크기가 너무 큽니다. 10MB 이하의 파일을 선택해주세요.', 'error');
                return;
            }

            extractBtn.disabled = true;
            extractBtn.textContent = '추출 중...';
            showMessage('PDF 텍스트 추출 중입니다...', 'info');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/extract-text', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    extractedText = data.text;
                    resultText.textContent = extractedText;
                    resultSection.style.display = 'block';
                    showMessage(`텍스트 추출 완료! (${data.pages}페이지, ${extractedText.length}자)`, 'success');
                } else {
                    const error = await response.json();
                    showMessage(`오류: ${error.detail}`, 'error');
                    resultSection.style.display = 'none';
                }
            } catch (error) {
                showMessage('서버 연결 오류가 발생했습니다.', 'error');
                resultSection.style.display = 'none';
            } finally {
                extractBtn.disabled = false;
                extractBtn.textContent = '텍스트 추출';
            }
        }

        function copyText() {
            navigator.clipboard.writeText(extractedText).then(() => {
                showMessage('클립보드에 복사되었습니다!', 'success');
            }).catch(() => {
                showMessage('클립보드 복사에 실패했습니다.', 'error');
            });
        }

        function downloadText() {
            if (!extractedText) return;

            const blob = new Blob([extractedText], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'extracted_text.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showMessage('텍스트 파일이 다운로드되었습니다!', 'success');
        }

        // 파일 선택 시 파일명 표시
        document.getElementById('pdfFile').addEventListener('change', function(e) {
            if (e.target.files[0]) {
                showMessage(`선택된 파일: ${e.target.files[0].name}`, 'info');
            }
        });

        // 드래그 앤 드롭 기능
        const uploadSection = document.querySelector('.upload-section');

        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.backgroundColor = '#e3f2fd';
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.style.backgroundColor = '';
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.backgroundColor = '';

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                document.getElementById('pdfFile').files = files;
                showMessage(`드롭된 파일: ${files[0].name}`, 'info');
            } else {
                showMessage('PDF 파일만 업로드 가능합니다.', 'error');
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """메인 페이지"""
    return HTML_TEMPLATE

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """PDF 파일에서 텍스트 추출"""

    # 파일 타입 검증
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 파일 크기 검증 (10MB 제한)
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="파일 크기가 너무 큽니다. 10MB 이하의 파일을 업로드해주세요.")

    try:
        # PDF 텍스트 추출
        pdf_stream = io.BytesIO(content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        text = ""
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n"

        doc.close()

        # 텍스트가 비어있는 경우
        if not text.strip():
            raise HTTPException(status_code=400, detail="텍스트를 추출할 수 없습니다. 이미지 기반 PDF이거나 보안이 설정된 파일일 수 있습니다.")

        return {
            "text": text,
            "pages": page_count,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 추출 중 오류가 발생했습니다: {str(e)}")

# 서버 실행용 코드
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
