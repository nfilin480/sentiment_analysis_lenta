import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollator
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


MODEL = 'xlm-roberta-base'
DATA = './dataset.csv'
BATCH_SIZE = 8

device = torch.device("cuda:0")

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

dataset = load_dataset('csv', data_files=DATA)

train_val = dataset["train"].train_test_split(
    test_size=0.1, shuffle=False, seed=42
)

def tokenize_and_align_labels(examples):

    text = examples['normalized']
    encoded_input = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(device)

    l = {'neutral': 0, 'positive': 1, 'negative': 2}

    labels = []

    targets = examples['sentiment']

    for item in targets:
      #a = [0] * len(l.keys())
      #a[l[item]] = 1
      labels.append(l[item])

    encoded_input["labels"] = labels
    return encoded_input


train_data = (
    train_val["train"].shuffle().map(tokenize_and_align_labels, batched=True, remove_columns=['Unnamed: 0', 'text', 'sentiment', 'normalized']) #batched=True,
)
train_data.set_format("torch")
val_data = (
    train_val["test"].shuffle().map(tokenize_and_align_labels, batched=True, remove_columns=['Unnamed: 0', 'text', 'sentiment', 'normalized'])
)
val_data.set_format("torch")

args = TrainingArguments(
    MODEL,
    evaluation_strategy = "epoch",
    logging_strategy = 'epoch',
    save_total_limit=1,
    save_strategy='epoch',
    warmup_steps=150,
    learning_rate=2e-05,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=10,
    lr_scheduler_type='linear',
    metric_for_best_model='f1_m',
    seed=42
)

def compute_metrics(p):

    predictions, labels = p
    #print(f"predictions: {predictions}")
    #print(f"labels: {labels}")
    predictions = np.argmax(predictions, axis=-1)

    # Remove ignored index (special tokens) and flatten the output
    true_predictions = [p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100 ]
    true_labels = [l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=true_labels, y_pred=true_predictions, average=None)
    f1_macro = f1_score(y_true=true_labels, y_pred=true_predictions, average="macro")
    return {
        'precision': list(precision),
        'recall': list(recall),
        'f1': list(f1),
        'f1 macro': f1_macro
    }


trainer = Trainer(
    model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    #data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()