import torch
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
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # 점수가 가장 높은 시작 및 끝 토큰 인덱스 찾기
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    # 답변 추출
    input_ids = inputs["input_ids"].tolist()[0]
    answer_ids = input_ids[answer_start:answer_end]
    answer = tokenizer.decode(answer_ids)

    # [CLS] 토큰 제거
    answer = answer.replace("[CLS] ", "")

    print("질문:", question)
    print("답변:", answer)
    print()
