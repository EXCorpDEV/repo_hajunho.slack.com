from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 모델 디렉토리 경로
model_directory = "downloaded_models/LG-AI-Research/SOLAR-10.7B-v1.0"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(model_directory)

# 입력 텍스트를 토큰화
input_text = "남한과 북한의 관계에 대해 설명해줘."
inputs = tokenizer(input_text, return_tensors="pt")

# 모델을 사용하여 텍스트 생성
outputs = model.generate(**inputs, max_length=50)

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
