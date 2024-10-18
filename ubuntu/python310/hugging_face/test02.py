from transformers import AutoTokenizer, AutoModelForCausalLM

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-v1.0")

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("upstage/SOLAR-10.7B-v1.0")

# 입력 텍스트를 토큰화
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# 모델을 사용하여 텍스트 생성
outputs = model.generate(**inputs, max_length=50)

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
