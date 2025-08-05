import torch
import time
import os
import random
import shutil
import sys
import subprocess
import importlib.util
import math
from PIL import Image
from diffusers import StableDiffusion3Pipeline

# 출력 폴더 설정
OUTPUT_DIR = "/Users/junhoha/ex_shutterstock"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 패키지 설치 로직
required_packages = ["diffusers", "accelerate", "protobuf", "sentencepiece"]

def check_install_package(package_name):
    """필요한 패키지가 설치되어 있는지 확인하고, 없으면 설치합니다."""
    if importlib.util.find_spec(package_name) is None:
        print(f"{package_name}을(를) 찾을 수 없습니다. 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} 설치 완료.")
    else:
        print(f"{package_name} 이미 설치되어 있음.")

# 필요한 패키지 확인 및 설치
for package in required_packages:
    check_install_package(package)

# Midjourney 기반 프롬프트 생성을 위한 테마 및 주제 목록
THEMES = [
    "nature", "technology", "city life", "abstract", "wildlife", "food", "architecture",
    "space", "underwater", "fantasy", "urban", "industrial", "traditional", "futuristic"
]

SUBJECTS = [
    "landscape", "portrait", "still life", "macro", "aerial", "minimalist",
    "vintage", "futuristic", "historical", "product", "architectural", "documentary"
]

# 실사 스타일 강조
STYLES = [
    "photorealistic", "ultra realistic", "hyperrealistic", "cinematic photography",
    "professional photography", "DSLR photo", "4K", "8K", "high definition",
    "detailed photography", "documentary style", "editorial photography",
    "commercial photography", "professional studio lighting", "product photography"
]

# 촬영 기법
PHOTOGRAPHY_TECHNIQUES = [
    "rule of thirds", "depth of field", "bokeh effect", "golden hour lighting",
    "blue hour", "natural lighting", "studio lighting", "volumetric lighting",
    "Rembrandt lighting", "split lighting", "dramatic lighting", "soft lighting",
    "backlighting", "silhouette", "macro photography", "wide angle", "telephoto lens"
]

# 카메라 키워드
CAMERA_KEYWORDS = [
    "shot on Canon EOS", "shot on Nikon D850", "shot on Sony A7R", "shot on Hasselblad",
    "shot on Leica", "50mm lens", "85mm lens", "24-70mm lens", "70-200mm lens",
    "wide-angle", "shallow depth of field", "perfect focus", "studio setting"
]

# 실사 효과 강화 키워드
REALISTIC_ENHANCERS = [
    "photorealistic", "realistic", "real life", "true to life", "high definition",
    "detailed", "lifelike", "natural", "professional photography", "DSLR quality",
    "studio quality", "8K resolution", "high resolution", "Ultra HD", "sharp focus",
    "perfect lighting", "high detail", "crystal clear", "perfect composition"
]

# 인물 사진용 키워드
PORTRAIT_ENHANCERS = [
    "professional portrait", "studio portrait", "perfect facial features",
    "detailed skin texture", "sharp facial features", "professional retouching",
    "magazine quality portrait", "commercial portrait", "editorial portrait",
    "fashion photography", "beauty photography", "lifestyle photography"
]

# 분위기 및 감성 키워드
MOOD_KEYWORDS = [
    "dramatic", "moody", "atmospheric", "serene", "peaceful", "energetic",
    "vibrant", "epic", "breathtaking", "stylish", "creative", "perfect",
    "gorgeous", "beautiful", "pretty"
]

# 기술적 요소
TECHNICAL_ELEMENTS = [
    "high dynamic range", "perfect composition", "color grading", "color correction",
    "professional post-processing", "detailed shadows", "perfect exposure",
    "balanced lighting", "optimal white balance", "perfect contrast"
]

def get_disk_space():
    """남은 디스크 용량을 바이트 단위로 반환"""
    total, used, free = shutil.disk_usage(OUTPUT_DIR)
    return free

def is_disk_full():
    """디스크가 거의 찼는지 확인 (5GB 미만 남음)"""
    free_space = get_disk_space()
    return free_space < 5 * 1024 * 1024 * 1024

def generate_random_prompt():
    """실사 강조를 포함한 랜덤 프롬프트 생성"""
    theme = random.choice(THEMES)
    subject = random.choice(SUBJECTS)
    style = random.choice(STYLES)
    technique = random.choice(PHOTOGRAPHY_TECHNIQUES)
    camera = random.choice(CAMERA_KEYWORDS)
    enhancer = random.choice(REALISTIC_ENHANCERS)
    mood = random.choice(MOOD_KEYWORDS)
    technical = random.choice(TECHNICAL_ELEMENTS)

    # "실사" 키워드 강제 포함
    if "photo" not in style.lower() and "real" not in style.lower():
        style = f"photorealistic {style}"

    # 인물 사진일 경우 추가 키워드
    portrait_details = ""
    if "portrait" in subject.lower():
        portrait_details = f", {random.choice(PORTRAIT_ENHANCERS)}"

    # 다양한 프롬프트 구조
    variations = [
        f"{mood} {subject} of {theme}, {style}, {enhancer}, {technique}, {camera}, {technical}{portrait_details}",
        f"{subject} of {theme} with {technique}, {style}, {enhancer}, {camera}, {technical}, {mood}{portrait_details}",
        f"{style} {subject} featuring {theme}, {enhancer}, {technique}, {camera}, {mood}, {technical}{portrait_details}",
        f"{mood} {style} {subject} of {theme}, {enhancer}, {camera}, {technical}, {technique}{portrait_details}",
        f"A {mood} {subject} showcasing {theme}, {style}, {enhancer}, {technique}, {camera}, {technical}{portrait_details}"
    ]

    # 디테일 강조 변형
    detailed_variations = [
        f"Highly detailed {style} {subject} of {theme}, {enhancer}, shot with {camera}, {technique}, {mood} atmosphere, {technical}{portrait_details}",
        f"Professional {style} {subject} featuring {theme}, extreme detail, {enhancer}, {technique}, {camera}, {mood} feel, {technical}{portrait_details}",
        f"Ultra-realistic {subject} of {theme}, {enhancer}, perfect {technique}, captured with {camera}, {mood}, {technical}, studio quality{portrait_details}"
    ]

    all_variations = variations + detailed_variations
    prompt = random.choice(all_variations)

    # 셔터스톡 스타일 메타데이터 태그 추가
    metadata_tags = f", shutterstock, stock photo, commercial use, high quality, professional"
    return prompt + metadata_tags

def resize_to_square_4k(image):
    """
    4K 해상도(3840x2160)의 픽셀 수와 동일한 정사각형 이미지로 변환
    총 픽셀 수 = 3840 * 2160 = 8,294,400, 따라서 약 2880x2880
    """
    total_pixels = 3840 * 2160
    side_length = int(math.sqrt(total_pixels))
    square_image = image.resize((side_length, side_length), Image.LANCZOS)
    return square_image

def read_first_prompts(file_path, num_prompts=10):
    """firstprompt.txt 파일에서 지정된 수의 프롬프트 읽기"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < num_prompts:
                    prompts.append(line.strip())
                else:
                    break
    except FileNotFoundError:
        print(f"{file_path} 파일을 찾을 수 없습니다. 랜덤 프롬프트를 사용합니다.")
    return prompts

def main():
    print("Stable Diffusion 3.5 모델 초기화 중...")
    model_id = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        variant="fp16"
    )
    pipe.enable_attention_slicing()  # 메모리 최적화
    pipe = pipe.to("mps")  # Mac에서 MPS 사용

    # firstprompt.txt 파일 경로
    first_prompt_file = os.path.join(os.path.dirname(__file__), "firstprompt.txt")
    first_prompts = read_first_prompts(first_prompt_file)

    count = 0
    print(f"{OUTPUT_DIR}에서 실사 이미지 생성 시작")
    print("Ctrl+C로 프로세스를 중단할 수 있습니다.")

    try:
        while not is_disk_full():
            try:
                if count < len(first_prompts):
                    prompt = first_prompts[count]
                    print(f"이미지 {count + 1} 생성 중 (지정된 프롬프트): '{prompt}'")
                else:
                    prompt = generate_random_prompt()
                    print(f"이미지 {count + 1} 생성 중 (랜덤 프롬프트): '{prompt}'")

                # 기본 이미지 생성
                image = pipe(
                    prompt,
                    num_inference_steps=40,
                    guidance_scale=7.5,
                    width=1024,
                    height=1024
                ).images[0]

                # 기본 이미지 저장
                base_timestamp = int(time.time())
                base_image_path = os.path.join(OUTPUT_DIR, f"generated_image_{base_timestamp}_base.png")
                image.save(base_image_path)
                print(f"기본 이미지 저장됨: {base_image_path}")

                # 4K 정사각형으로 업스케일링
                try:
                    image_square_4k = resize_to_square_4k(image)
                    timestamp = base_timestamp + 1
                    image_path = os.path.join(OUTPUT_DIR, f"generated_image_{timestamp}_square4k.png")
                    image_square_4k.save(image_path, format="PNG")
                    print(f"정방형 4K 이미지 저장됨: {image_path}")
                    count += 1
                except Exception as e:
                    print(f"업스케일링 중 오류: {e}")
                    print(f"기본 이미지는 저장됨: {base_image_path}")
                    count += 1

                # 메타데이터 저장
                prompt_file_path = os.path.join(OUTPUT_DIR, f"generated_image_{base_timestamp}_prompt.txt")
                with open(prompt_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_timestamp))}\n")
                    f.write(f"Model: stabilityai/stable-diffusion-3.5-large\n")
                    f.write(f"Steps: 40, Guidance Scale: 7.5\n")
                print(f"프롬프트 정보 저장됨: {prompt_file_path}")

                time.sleep(1)  # 시스템 과부하 방지
                free_space_gb = get_disk_space() / (1024 * 1024 * 1024)
                print(f"남은 공간: {free_space_gb:.2f} GB")
            except Exception as e:
                print(f"이미지 생성 중 오류: {e}")
                print("다른 프롬프트로 재시도 중...")
                time.sleep(5)

        print(f"{OUTPUT_DIR} 폴더가 가득 찼습니다. 생성 중단.")
        print(f"총 생성된 이미지 수: {count}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로세스 중단됨")
        print(f"총 생성된 이미지 수: {count}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()