from transformers import pipeline

# 파이프라인을 사용하여 모델 로드
pipe = pipeline("text-generation", model="upstage/SOLAR-10.7B-v1.0")

# 텍스트 생성
generated_text = pipe("Hello, how are you?", max_length=50)
print(generated_text[0]['generated_text'])
