from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch, numpy as np, os, pandas as pd, datasets, evaluate, sys, pickle, json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoModelForMultipleChoice
from scipy.special import softmax
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate, random, os
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from utils.selection_utils import log_training_dynamics


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"
checkpoint = "roberta-base"
num_labels=2
model_type="roberta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


output_dir="cartography/outputs"

ds_train = load_dataset('csv', data_files='data/train_imdb.csv')
ds_val = load_dataset('csv', data_files='data/val_imdb.csv')
ds_test = load_dataset('csv', data_files='data/test_imdb.csv')

#print(ds)
#sys.exit()
data = DatasetDict({
    'train': ds_train['train'],
    'valid': ds_val['train'],
    'test': ds_test['train']}  
)

print(data)
#sys.exit()
tokenized_dataset = data.map(preprocess_function, batched=True, num_proc=12)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
# tokenized_dataset.set_format("torch")
tokenized_dataset.set_format("torch",columns=["global_index","input_ids", "attention_mask", "labels"])
print(tokenized_dataset["train"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_dataloader = DataLoader(
    tokenized_dataset["test"], shuffle=True, batch_size=32, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 6
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

metric = evaluate.load("f1")


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
  train_ids = None
  train_golds = None
  train_logits = None
  train_losses = None
  
  for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    inputs = {"input_ids": batch['input_ids'], "attention_mask": batch['attention_mask'], "labels": batch['labels']}
    if model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch['token_type_ids'] if model_type in ["bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    outputs = model(**inputs)
    loss = outputs.loss  # model outputs are always tuple in transformers (see doc)
    if train_logits is None:  # Keep track of training dynamics.
        train_ids = batch['global_index'].detach().cpu().numpy()
        train_logits = outputs[1].detach().cpu().numpy()
        train_golds = inputs["labels"].detach().cpu().numpy()
        train_losses = loss.detach().cpu().numpy()
    else:
        train_ids = np.append(train_ids, batch['global_index'].detach().cpu().numpy())
        train_logits = np.append(train_logits, outputs[1].detach().cpu().numpy(), axis=0)
        train_golds = np.append(train_golds, inputs["labels"].detach().cpu().numpy())
        train_losses = np.append(train_losses, loss.detach().cpu().numpy())

    # # loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    log_training_dynamics(output_dir=output_dir,
                              epoch=epoch,
                              train_ids=list(train_ids),
                              train_logits=list(train_logits),
                              train_golds=list(train_golds))
    #break
    progress_bar.update(1)

