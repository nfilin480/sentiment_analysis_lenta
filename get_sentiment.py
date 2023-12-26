import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollator
from datasets import load_dataset
import ast
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

MODEL = '/content/drive/MyDrive/data_bert/xlm-roberta-base/checkpoint-900/'
DATA = './data.csv'
BATCH_SIZE = 8

device = torch.device("cuda:0")

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

dataset = load_dataset('csv', data_files=DATA)

test_val= dataset["train"]

def tokenize_and_align_labels(examples):

    text = examples['normalized_text']
    text = ''. join(text)
    encoded_input = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(device)

    return encoded_input

test_data = (
    test_val.shuffle().map(tokenize_and_align_labels, batched=True, remove_columns=['Unnamed: 0.1', 'Unnamed: 0', 'url','title','text','tags','normalized_text','topic']) #batched=True,
)


labels = []
for example in tqdm(test_val):
  input = tokenize_and_align_labels(example)
  outputs = model(**input)
  logits = outputs.logits[0]

  label = logits.argmax().item()
  labels.append(label)

l = {'neutral' : 0, 'positive' : 1, 'negative' : 2}

sentiment = [list(l.keys())[i] for i in labels]

df = pd.read_csv('./data.csv')
df['sentiment'] = sentiment

df.to_csv('./data_with_sentiment.csv')