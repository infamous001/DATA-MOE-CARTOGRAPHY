from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch, os, evaluate
import torch.nn as nn
import pandas as pd

import dataclasses
import torch.nn.functional as F

from typing import List, Optional

from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import DataLoader
from transformers import AdamW,get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm
from utils.helper import CustomModel, ModelArgs, MoeArgs



def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512

args=ModelArgs(dim=768,hidden_dim=3072,num_labels=2,moe=MoeArgs(3,1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,args=args).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')

ds = load_dataset("imdb",split="train")
ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")
#print(ds)


data = DatasetDict({
    'train': ds['train'],
    'valid': ds['test'],
    'test': load_dataset("imdb",split="test")}  
)

print(data)

tokenized_dataset = data.map(preprocess_function, batched=True, num_proc=12)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
print(tokenized_dataset["train"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=32, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["valid"], batch_size=32, collate_fn=data_collator
)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

metric = evaluate.load("f1")


progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))


for epoch in range(num_epochs):
  model.train()
  for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs, s = model(**batch)
      print(f"train: {s}")
      loss = outputs.loss
      loss.backward()

      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar_train.update(1)

  model.eval()
  for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs, s = model(**batch)

    logits = outputs.logits
    print(f"val: {s}")
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar_eval.update(1)

  print(metric.compute())


model.eval()
test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=32, collate_fn=data_collator
)

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs, s = model(**batch)

    logits = outputs.logits
    print(f"test: {s}")
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())

