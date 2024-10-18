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

        # Handle negative start_position
        if start_position < 0:
            token_start_position = 0
        else:
            token_start_position = tokenized_examples.char_to_token(i, start_position)

        # Handle negative end_position
        if end_position < 0:
            token_end_position = 0
        else:
            token_end_position = tokenized_examples.char_to_token(i, end_position - 1)

        tokenized_examples['start_positions'].append(token_start_position)
        tokenized_examples['end_positions'].append(token_end_position)

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

    # 예측값과 레이블을 적절한 형식으로 변환
    true_predictions = []
    true_labels = []

    for i in range(len(predictions)):
        pred_start = predictions[i][0]
        pred_end = predictions[i][1] + 1
        label_start = labels[i][0]
        label_end = labels[i][1] + 1

        pred_text = tokenizer.decode(tokenized_datasets['validation'][i]['input_ids'][pred_start:pred_end])
        label_text = tokenizer.decode(tokenized_datasets['validation'][i]['input_ids'][label_start:label_end])

        true_predictions.append({"id": str(i), "prediction_text": pred_text})
        true_labels.append({"id": str(i), "answers": {"text": [label_text], "answer_start": [label_start]}})

    return metric.compute(predictions=true_predictions, references=true_labels)

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