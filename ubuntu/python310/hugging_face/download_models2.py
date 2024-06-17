import os
from transformers import AutoModel, AutoTokenizer

# 인기 있는 모델 리스트
popular_models = [
    # "bert-base-uncased",
    # "gpt2",
    # "xlnet-base-cased",
    # "roberta-base",
    # "distilbert-base-uncased",
    # "albert-base-v2",
    # "t5-small",
    # "facebook/bart-base",
    # "microsoft/DialoGPT-medium",
    # "google/electra-small-discriminator",
    # "EleutherAI/gpt-neo-2.7B",
    # "EleutherAI/gpt-j-6B",
    # "openai/whisper-base",
    # "LG-AI-Research/SOLAR-10.7B-v1.0",
    # "microsoft/deberta-base",
    # "google/flan-t5-base",
    # "stanford-crfm/pubmed-gpt",
    # "allenai/longformer-base-4096",
    # "bigscience/bloom-560m",
    # "facebook/blenderbot-400M-distill",
    # "gpt2-large",
    # "gpt2-xl",
    # "microsoft/DialoGPT-large",
    # "facebook/bart-large",
    # "t5-base",
    # "t5-large",
    # "google/pegasus-xsum",
    # "sshleifer/distilbart-cnn-12-6",
    # "EleutherAI/gpt-neo-1.3B",
    # "EleutherAI/gpt-neox-20b",
    "stabilityai/stable-diffusion-2-base",
    "openai/whisper-medium",
    "openai/whisper-large",
    "bigscience/T0pp",
    "bigscience/mt0-large",
    "deepmind/Flamingo-9B",
    "anthropic/claude-v1",
    "databricks/dolly-v2-3b",
    "facebook/llama-7b",
    "facebook/llama-13b",
    "google/flan-ul2",
    "google/ul2",
    "nlpcloud/llama-13b-hf",
    "nlpcloud/llama-7b-hf",
    "stabilityai/stablelm-base-alpha-3b",
    "stabilityai/stablelm-tuned-alpha-3b",
    "togethercomputer/GPT-NeoXT-Chat-Base-20B",
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
