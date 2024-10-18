from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset('klue', 'mrc')

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# 토크나이징 함수 정의
def tokenize_function(examples):
    tokenized_examples = tokenizer(
        examples['context'],
        examples['question'],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # 정답이 없는 경우 start_positions와 end_positions을 -1로 설정
    tokenized_examples['start_positions'] = [-1] * len(examples['context'])
    tokenized_examples['end_positions'] = [-1] * len(examples['context'])

    for i, answer in enumerate(examples['answers']):
        if len(answer['text']) > 0:
            tokenized_examples['start_positions'][i] = answer['answer_start'][0]
            tokenized_examples['end_positions'][i] = answer['answer_start'][0] + len(answer['text'][0])

    return tokenized_examples

# 데이터셋 토크나이징
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 모델 로드
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-multilingual-cased')

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 트레이너 설정 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()

# 모델 저장
model.save_pretrained('./testmodel')
tokenizer.save_pretrained('./testmodel')