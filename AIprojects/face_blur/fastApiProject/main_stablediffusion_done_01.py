from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from diffusers import StableDiffusionPipeline
from fastapi.staticfiles import StaticFiles
import torch

app = FastAPI()
# "static" 폴더를 "/static" 경로로 서빙한다.
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# CUDA에서 Stable Diffusion Pipeline 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/generate")
# async def generate(request: Request):
#     form = await request.form()
#     prompt = form.get("prompt")
#
#     image = pipe(prompt).images[0]
#     image.save("generated_image.png")
#
#     return templates.TemplateResponse("result.html", {"request": request, "image_path": "generated_image.png"})

import time


@app.post("/generate")
async def generate(request: Request):
    form = await request.form()
    prompt = form.get("prompt")

    image = pipe(prompt).images[0]

    # 현재 시간을 타임스탬프로 변환
    timestamp = int(time.time())

    # 타임스탬프를 포함한 파일 이름 생성
    image_path = f"static/generated_image_{timestamp}.png"

    # 이미지 저장
    # image.save(image_path)
    try:
        image.save(image_path)
    except Exception as e:
        print(f"Error saving file: {e}")

    return templates.TemplateResponse("result.html", {"request": request, "image_path": image_path})