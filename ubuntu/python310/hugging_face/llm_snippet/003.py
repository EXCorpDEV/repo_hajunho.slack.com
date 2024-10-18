from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 모델과 토크나이저 로드
model = AutoModelForQuestionAnswering.from_pretrained('cooler8/klue-mrc-bert-hjh-test001')
tokenizer = AutoTokenizer.from_pretrained('cooler8/klue-mrc-bert-hjh-test001')

# 지문과 질문 예시
context = """
대한민국의 국보 제1호는 숭례문이다. 숭례문은 조선시대에 한양도성의 정문으로 사용되었으며, 
현재는 서울특별시 중구 세종대로에 있다. 숭례문은 임진왜란 때 불타 없어졌으나, 1979년에 
복원되었다. 숭례문은 조선 태조 이성계가 새로 수도를 정할 때 건립되었다.
"""

questions = [
    "대한민국의 국보 제1호는 무엇인가요?",
    "숭례문은 언제 건립되었나요?",
    "숭례문은 과거에 어떤 용도로 사용되었나요?",
    "숭례문은 임진왜란 때 어떻게 되었나요?",
    "숭례문은 현재 어디에 위치해 있나요?"
]

for question in questions:
    # 입력 인코딩
    inputs = tokenizer(question, context, return_tensors='pt')

    # 모델 추론
    outputs = model(**inputs)

    # 답변 디코딩
    start_index = outputs.start_logits.argmax()
    end_index = outputs.end_logits.argmax()
    answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index+1])

    print("질문:", question)
    print("답변:", answer)
    print()

