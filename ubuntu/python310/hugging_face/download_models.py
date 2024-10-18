import os
from transformers import AutoModel, AutoTokenizer

# 인기 있는 모델 리스트
popular_models = [
    "bert-base-uncased",
    "gpt2",
    "xlnet-base-cased",
    "roberta-base",
    "distilbert-base-uncased",
    "albert-base-v2",
    "t5-small",
    "facebook/bart-base",
    "microsoft/DialoGPT-medium",
    "google/electra-small-discriminator",
]

# 모델 다운로드 및 저장
for model_name in popular_models:
    print(f"Downloading {model_name}...")

    # 모델 다운로드
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 모델 저장
    save_directory = f"downloaded_models/{model_name}"
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"Model {model_name} downloaded and saved.\n")
