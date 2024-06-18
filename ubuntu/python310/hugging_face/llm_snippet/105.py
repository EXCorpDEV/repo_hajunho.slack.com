from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import numpy as np
from datasets import load_dataset
from evaluate import load

# 데이터셋 로드
dataset = load_dataset('klue', 'mrc')

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

# 토크나이징 함수 정의
def tokenize_function(examples):
    tokenized_examples = tokenizer(
        examples['context'],
        examples['question'],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []
    
    for i in range(len(examples['answers'])):
        start_position = -1
        end_position = -1
        for answer in examples['answers'][i]:
            if isinstance(answer, dict):
                if isinstance(answer['answer_start'], list):
                    start_position = answer['answer_start'][0]
                else:
                    start_position = answer['answer_start']
                
                end_position = start_position + len(answer['text'])
                break
        
        tokenized_examples['start_positions'].append(start_position)
        tokenized_examples['end_positions'].append(end_position)
    
    return tokenized_examples

# 데이터셋 토크나이징
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)

# 모델 로드
model = AutoModelForQuestionAnswering.from_pretrained('klue/bert-base')

# 평가 지표 설정
metric = load("squad")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

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
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 모델 저장
model.save_pretrained('./testmodel')
tokenizer.save_pretrained('./testmodel')
