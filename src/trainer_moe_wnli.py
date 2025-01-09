import torch, os, evaluate
import torch.nn as nn
import pandas as pd

import dataclasses, sys, pickle, json
import torch.nn.functional as F
from typing import List, Optional
import matplotlib.pyplot as plt, random, numpy as np
from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import DataLoader
from transformers import AdamW,get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm
from utils.helper_1 import CustomModel, ModelArgs, MoeArgs
from utils.helper_1 import ConfiguredMetric
from torch.nn import DataParallel

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# def preprocess_function(examples):
#     return tokenizer(examples["sentence"], truncation=True, max_length=128, padding=True)

def preprocess_function(examples):
    return tokenizer(examples['text1'], examples['text2'], truncation=True,max_length=128)

os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


with open("results/best_hyperparameters_tweet_eval.json", "r") as jfile:
    hp_params=json.load(jfile)

lr=hp_params['learning_rate']
batch_size=hp_params['per_device_train_batch_size']
num_epochs=hp_params['num_train_epochs']
num_experts=3
num_experts_per_tok=1
mode=f"{num_experts}_{num_experts_per_tok}_cbz_f" #ce/cbz/cb/cz/15_3_cbz
dt="wnli"


args=ModelArgs(dim=768,hidden_dim=3072,num_labels=2,moe=MoeArgs(num_experts,num_experts_per_tok))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,args=args).to(device)

# model.to(device)
# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     model = DataParallel(model)
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')


ds_train = load_dataset('csv', data_files=f'data/{dt}/train.csv')
ds_val = load_dataset('csv', data_files=f'data/{dt}/val.csv')
ds_test = load_dataset('csv', data_files=f'data/{dt}/test.csv')

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
# tokenized_dataset = tokenized_dataset.remove_columns(["text", "split"])
tokenized_dataset = tokenized_dataset.remove_columns(["text1","text2","label_text", "idx"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
#tokenized_dataset.set_format("torch")
tokenized_dataset.set_format("torch",columns=["global_idx","input_ids", "attention_mask", "labels"])
print(tokenized_dataset["train"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["valid"], batch_size=batch_size, collate_fn=data_collator
)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

num_epochs = num_epochs
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

metric = evaluate.combine([
                evaluate.load('accuracy'), 
                ConfiguredMetric(evaluate.load('f1'), average='macro'),
                ConfiguredMetric(evaluate.load('precision'), average='macro'),
                ConfiguredMetric(evaluate.load('recall'), average='macro'),
            ])


progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

train_losses = []  
val_losses = []   

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in train_dataloader:
        e = 'global_idx'
        batch = {k: v.to(device) for k, v in batch.items() if k != e}
        outputs, selected_experts = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()  # Accumulate the loss
        num_batches += 1
        
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

    
    avg_train_loss = total_loss / num_batches
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}, Training loss: {avg_train_loss}")

    # Evaluation step
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    for batch in eval_dataloader:
        e = 'global_idx'
        batch = {k: v.to(device) for k, v in batch.items() if k != e}
        with torch.no_grad():
            outputs, sel = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
        # Calculate validation loss
        val_loss = outputs.loss.item()  # Assuming outputs has a loss attribute
        total_val_loss += val_loss
        num_val_batches += 1
        progress_bar_eval.update(1)

    # Calculate the average validation loss for this epoch and store it
    avg_val_loss = total_val_loss / num_val_batches
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}, Validation loss: {avg_val_loss}")
    print(metric.compute())

# Plotting the training and validation loss
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.savefig(f'plots/Loss_{mode}_{dt}.png')
plt.show()

metric_test = evaluate.combine([
                evaluate.load('accuracy'), 
                ConfiguredMetric(evaluate.load('f1'), average='macro'),
                ConfiguredMetric(evaluate.load('precision'), average='macro'),
                ConfiguredMetric(evaluate.load('recall'), average='macro'),
            ])
test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
)
expert_list=[]
model.eval()
for batch in test_dataloader:
    key_to_exclude = 'global_idx'
    glo_batch = {k: v for k, v in batch.items() if k == key_to_exclude}
    batch = {k: v.to(device) for k, v in batch.items() if k != key_to_exclude}
    with torch.no_grad():
        outputs, s = model(**batch)
        for k in s.keys():
            for i in range(len(s[k])):
                index=s[k][i]
                s[k][i]=glo_batch['global_idx'][index].item()
        expert_list.append(s)

    logits = outputs.logits
    #print(f"test: {s}")
    predictions = torch.argmax(logits, dim=-1)
    metric_test.add_batch(predictions=predictions, references=batch["labels"])


# with open("results/expert_dist.pkl", "wb") as pfile:
#     pickle.dump(expert_list, pfile)
results = metric_test.compute()
print(results)

with open(f"results/{dt}_{mode}.json", "w") as jfile:
    json.dump(results, jfile)


# result1 = {0: 0, 1: 0, 2: 0}
result1 = {key:0 for key in range(num_experts)}
for d in expert_list:
    for key in result1.keys():
        result1[key] += len(d[key])

# result2 = {0: [], 1: [], 2: []}
result2 = {key:[] for key in range(num_experts)}
for d in expert_list:
    for key in result2.keys():
        result2[key]=result2[key]+d[key]


with open(f"results/expert_dist_{mode}_{dt}.json", "w") as jfile:
    json.dump(result1, jfile)
print('json_dist saved')




with open(f"results/expert_mapped_{mode}_{dt}.json", "w") as jfile:
    json.dump(result2, jfile)
print('json_mapped saved')

