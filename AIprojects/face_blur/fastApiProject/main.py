from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# SDXL 파이프라인 임포트
from diffusers import StableDiffusionXLPipeline
import torch
import time
import os

# FastAPI 앱 생성
app = FastAPI()

# "static" 폴더를 "/static" 경로로 서빙 (이미지·CSS·JS 등 정적 파일 제공)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 템플릿 설정 (템플릿 폴더 이름: "templates")
templates = Jinja2Templates(directory="templates")

# 사용하려는 SDXL 모델 ID (Turbo 모델)
model_id = "stabilityai/stable-diffusion-xl-turbo"

# SDXL 파이프라인 로드
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"  # Turbo 모델용 추가 설정
).to("cuda")

# 기본 라우트 (GET) - 입력 폼을 보여주는 페이지
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 이미지 생성 라우트 (POST) - 프롬프트를 받아 SDXL로 이미지 생성
@app.post("/generate")
async def generate(request: Request):
    # HTML form 데이터 받기
    form = await request.form()
    prompt = form.get("prompt")

    # SDXL로 이미지 생성
    image = pipe(prompt).images[0]

    # 파일 이름에 타임스탬프를 붙여 중복 방지
    timestamp = int(time.time())
    image_path = f"static/generated_image_{timestamp}.png"

    # 이미지 파일 저장
    # static 폴더가 없으면 미리 만들어 주세요 (os.makedirs("static", exist_ok=True))
    try:
        image.save(image_path)
        print(f"Image saved to {image_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    # 결과 페이지 렌더링, 생성된 이미지 경로 전달
    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": image_path
    })
