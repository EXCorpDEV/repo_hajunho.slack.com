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

#Realistic 과 Photorealistic은 다르다.
#🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷🩷
#🩷prompt = "A round metal sign with a scene of the Archangel Michael in relief, highly integrated with the base plate, with the word Michael written on it, presenting the theme in a realistic style, adding some details to enrich the theme, with a relief effect, slight rust, with a delicate border, high resolution and high definition"
#🩷prompt = "Topic: emoji, 3d rendering, expression sheet of chibi Maruko-chan, slapped, dizzy, happy, angry, crying, sad, cute, looking forward to, laughing, disappointed and sha, sleepy, Eating, Dizzy, Love, Pixar Style"
# prompt = "A man far away in the rain, with his back"
# prompt = "Photorealistic, A group picture of 6 women with ages 20-60 years and different skin colors with curly hair, afro's and african braids. Also they should wear different clothing They all stand together in a group photograph with plain gray background like a cover photo for a magazine" #first_prompts[count]
prompt = "A beautiful girl, Imagine a style characterized by enchanting, hand-drawn charm with soft, luminous colors and a dreamy, painterly quality. The characters feature large, expressive eyes that convey deep emotion and wonder, while their fluid, natural movements exude both vitality and graceful subtlety. Backgrounds burst with intricately detailed, organic landscapes—lush, misty forests, tranquil villages, and sun-dappled meadows rendered in delicate brushstrokes and gentle gradients. This aesthetic blends whimsical fantasy with heartfelt storytelling, evoking a nostalgic warmth and magical atmosphere that transforms everyday scenes into captivating, otherworldly visions." #first_prompts[count]#🩷
# prompt = "realistic, A beautiful girl" #🩷
# prompt = "Photorealistic, UltraReal, 8K, Cute kitten minuet, ears drooping, being held by a pretty girl, smiling, pale and dreamy, shining, pastel colors" #
# prompt = "An ultra-luxurious cinematic animation featuring a massive number 8 sculpted from crystal and diamonds, radiating soft pink and white light. A group of miniature elegant gentlemen in white tuxedos carefully decorates the structure with precise, graceful movements. One gentleman stands on a silver ladder, meticulously setting a sparkling gemstone into place. Another polishes the surface with a delicate cloth, creating a radiant shine. A third adjusts the alignment of tiny embedded lights, making the diamonds glisten with dazzling reflections. Two more gentlemen at the base carefully examine the composition, one holding a magnifying glass while the other inspects a butterfly that gracefully lands on the diamond surface. Delicate iridescent butterflies flutter around, catching the light in soft slow motion. The camera smoothly glides through the scene, capturing detailed reflections and subtle hand movements. Gentle bokeh effects, soft cinematic lighting, dreamlike slow-motion elements, high-fashion runway atmosphere, ethereal elegance, ultra-high-definition quality."
#🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵🩵
#􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲􀀲
# prompt = "extreme close up, lace and a beautiful woman’s entire face, LED twinkle lights, ginger hair, golden hour, big ocean eyes, monarch butterflies, blue eyes, dramatic, orange, turquoise"
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

def get_disk_space():
    """남은 디스크 용량을 바이트 단위로 반환"""
    total, used, free = shutil.disk_usage(OUTPUT_DIR)
    return free

def is_disk_full():
    """디스크가 거의 찼는지 확인 (5GB 미만 남음)"""
    free_space = get_disk_space()
    return free_space < 5 * 1024 * 1024 * 1024


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
                print(f"이미지 {count + 1} 생성 중 (지정된 프롬프트): '{prompt}'")

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