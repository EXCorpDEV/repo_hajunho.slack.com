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
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "openai/whisper-base",
    "LG-AI-Research/SOLAR-10.7B-v1.0",
    "microsoft/deberta-base",
    "google/flan-t5-base",
    "stanford-crfm/pubmed-gpt",
    "allenai/longformer-base-4096",
    "bigscience/bloom-560m",
    "facebook/blenderbot-400M-distill",
]

# 모델 다운로드 및 저장
for model_name in popular_models:
    print(f"Downloading {model_name}...")

    try:
        # 모델 다운로드
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 모델 저장
        save_directory = f"downloaded_models/{model_name}"
        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        print(f"Model {model_name} downloaded and saved.\n")
    except Exception as e:
        print(f"Error downloading model {model_name}: {str(e)}\n")
