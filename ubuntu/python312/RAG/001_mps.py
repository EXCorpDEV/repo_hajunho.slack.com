from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import torch

# CUDA 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wiki_dpr 데이터셋을 다운로드 (구성 이름 추가)
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact")

# RAG 모델과 토크나이저 로드
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq").to(device)

# 질문 입력
question = "What is the capital of France?"

# 질문을 토큰화
inputs = tokenizer(question, return_tensors="pt").to(device)

# 관련 문서 검색
retrieved_docs = retriever.retrieve(question)

# RAG 모델을 통해 답변 생성
generated_ids = model.generate(**inputs, retriever=retrieved_docs)

# 생성된 답변 디코딩
answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Answer:", answer[0])
