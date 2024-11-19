from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import io

app = FastAPI()

# Haar Cascade 파일 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Blur App</title>
    </head>
    <body>
        <h1>(주)EX 얼굴 블러 처리 앱</h1>
        <form id="upload-form">
            <input type="file" id="file-input" accept="image/*">
            <button type="submit">업로드</button>
        </form>
        <div id="result">
            <h2>결과:</h2>
            <img id="result-image" src="" alt="처리된 이미지" style="max-width:100%;">
        </div>
        <script>
            const form = document.getElementById('upload-form');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/blur_faces/', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const img = document.getElementById('result-image');
                    img.src = url;
                } else {
                    alert('이미지 처리 중 오류가 발생했습니다.');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/blur_faces/")
async def blur_faces(file: UploadFile = File(...)):
    try:
        # 업로드된 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 이미지가 유효한지 확인
        if img is None:
            return {"error": "유효하지 않은 이미지입니다."}

        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴이 검출되지 않은 경우 원본 이미지 반환
        if len(faces) == 0:
            # 원본 이미지 인코딩
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

        # 얼굴 영역 블러 처리
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi = cv2.GaussianBlur(roi, (99, 99), 30)
            img[y:y+h, x:x+w] = roi

        # 처리된 이미지 인코딩
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}
