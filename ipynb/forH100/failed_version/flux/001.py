import torch
import time
import os
from datetime import datetime
from diffusers import FluxPipeline
import gc
import psutil
import GPUtil

class FluxInfiniteGenerator:
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", output_dir="generated_images"):
        """
        Flux 모델을 사용한 무한 이미지 생성기
        
        Args:
            model_name: 사용할 Flux 모델 이름
            output_dir: 이미지를 저장할 디렉토리
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.pipe = None
        self.generation_count = 0
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"출력 디렉토리: {os.path.abspath(output_dir)}")
        
    def setup_model(self):
        """모델 로드 및 설정"""
        print("Flux 모델을 로드하는 중...")
        print(f"사용 중인 GPU: {torch.cuda.get_device_name()}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print("토큰 인증 확인 중...")
            
            # Flux 파이프라인 로드 - device_map 제거
            self.pipe = FluxPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            
            # GPU로 이동
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipe = self.pipe.to(device)
            
            # 메모리 최적화
            if torch.cuda.is_available():
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers 최적화를 사용할 수 없습니다.")
            
            print(f"모델이 {device}에 로드되었습니다!")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("다음을 확인해주세요:")
            print("1. Hugging Face 로그인: huggingface-cli login")
            print("2. 모델 액세스 권한 확인")
            print("3. 또는 FLUX.1-schnell 모델 사용")
            raise
    
    def generate_image(self, prompt, negative_prompt="", num_inference_steps=4):
        """단일 이미지 생성"""
        try:
            # 메모리 정리
            if self.generation_count > 0 and self.generation_count % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # 이미지 생성
            with torch.autocast("cuda"):
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,  # schnell은 4스텝이 최적
                    guidance_scale=0.0,  # schnell은 guidance_scale=0.0 사용
                    width=1024,
                    height=1024,
                    generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 1000000, (1,)).item())
                ).images[0]
            
            return image
            
        except Exception as e:
            print(f"이미지 생성 실패: {e}")
            return None
    
    def save_image(self, image, prompt):
        """이미지 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flux_gen_{self.generation_count:06d}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        image.save(filepath)
        print(f"저장됨: {filename}")
        
        # 메타데이터 저장 (선택사항)
        meta_filepath = filepath.replace('.png', '_meta.txt')
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generation: {self.generation_count}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model_name}\n")
    
    def print_system_status(self):
        """시스템 상태 출력"""
        if self.generation_count % 5 == 0:  # 5개마다 상태 체크
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    print(f"GPU 메모리: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
                
                ram = psutil.virtual_memory()
                print(f"RAM 사용량: {ram.percent:.1f}%")
                print("-" * 50)
            except:
                pass  # 상태 체크 실패시 무시
    
    def infinite_generate(self, base_prompt="beautiful woman", variations=True):
        """무한 생성 시작"""
        print("무한 이미지 생성을 시작합니다...")
        print("중단하려면 Ctrl+C를 누르세요.")
        print("=" * 60)
        
        # 프롬프트 변형을 위한 추가 키워드들
        style_variations = [
            "", "portrait", "realistic", "artistic", "elegant", "graceful",
            "professional photo", "studio lighting", "natural lighting",
            "soft lighting", "cinematic", "detailed", "high quality",
            "beautiful face", "stunning", "gorgeous", "photorealistic"
        ]
        
        try:
            while True:
                start_time = time.time()
                
                # 프롬프트 생성
                if variations and len(style_variations) > 0:
                    import random
                    variation = random.choice(style_variations)
                    if variation:
                        current_prompt = f"{base_prompt}, {variation}"
                    else:
                        current_prompt = base_prompt
                else:
                    current_prompt = base_prompt
                
                print(f"생성 #{self.generation_count + 1}: '{current_prompt}'")
                
                # 이미지 생성
                image = self.generate_image(current_prompt)
                
                if image is not None:
                    # 이미지 저장
                    self.save_image(image, current_prompt)
                    self.generation_count += 1
                    
                    generation_time = time.time() - start_time
                    print(f"생성 시간: {generation_time:.2f}초")
                    
                    # 시스템 상태 출력
                    self.print_system_status()
                    
                else:
                    print("이미지 생성 실패, 다시 시도합니다...")
                    time.sleep(2)
                
        except KeyboardInterrupt:
            print(f"\n생성이 중단되었습니다. 총 {self.generation_count}개의 이미지가 생성되었습니다.")
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            
    def run(self, prompt="beautiful woman"):
        """전체 프로세스 실행"""
        try:
            # 모델 설정
            self.setup_model()
            
            # 무한 생성 시작
            self.infinite_generate(prompt)
            
        except Exception as e:
            print(f"프로그램 실행 오류: {e}")

if __name__ == "__main__":
    # 설정 - FLUX.1-schnell이 더 빠르고 토큰 없이도 사용 가능
    MODEL_NAME = "black-forest-labs/FLUX.1-schnell"  # 더 빠른 모델, 4스텝
    # MODEL_NAME = "black-forest-labs/FLUX.1-dev"    # 더 고품질이지만 28스텝 필요
    
    OUTPUT_DIR = "flux_generated_women"
    PROMPT = "beautiful woman"
    
    print("Flux H100 무한 이미지 생성기")
    print(f"모델: {MODEL_NAME}")
    print(f"프롬프트: {PROMPT}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 생성기 실행
    generator = FluxInfiniteGenerator(MODEL_NAME, OUTPUT_DIR)
    generator.run(PROMPT)
