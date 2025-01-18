from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import torch

# CUDA 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wiki_dpr 데이터셋을 다운로드 (trust_remote_code=True 추가)
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", trust_remote_code=True)

# RAG 모델과 토크나이저 로드
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Faiss 없이 간단히 검색(retrieval) 대체
class SimpleRetriever:
    def __init__(self, dataset):
        self.dataset = dataset

    def retrieve(self, question, top_k=5):
        # 데이터셋의 문서에서 질문과 관련된 단순 검색 (여기서는 첫 N개 반환)
        return [self.dataset["train"][i]["text"] for i in range(top_k)]

retriever = SimpleRetriever(dataset)

# RAG 모델 로드
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq").to(device)
`
# 질문 입력
question = "What is the capital of France?"

# 질문을 토큰화
inputs = tokenizer(question, return_tensors="pt").to(device)

# 관련 문서 검색 (Faiss 대체)
retrieved_docs = retriever.retrieve(question)

# 검색된 문서를 모델 입력에 추가
retrieval_inputs = tokenizer.batch_encode_plus(
    retrieved_docs, return_tensors="pt", truncation=True, padding=True
).to(device)

# 모델 입력 결합
inputs["context_input_ids"] = retrieval_inputs["input_ids"]
inputs["context_attention_mask"] = retrieval_inputs["attention_mask"]

# RAG 모델을 통해 답변 생성
generated_ids = model.generate(**inputs)

# 생성된 답변 디코딩
answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Answer:", answer[0])
