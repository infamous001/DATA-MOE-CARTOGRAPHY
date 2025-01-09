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
from utils.helper import CustomModel, ModelArgs, MoeArgs
from utils.helper import ConfiguredMetric
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding=True)

os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


with open("results/best_hyperparameters_tweet_eval.json", "r") as jfile:
    hp_params=json.load(jfile)

lr=hp_params['learning_rate']
batch_size=hp_params['per_device_train_batch_size']
num_epochs=1
num_experts=3
num_experts_per_tok=1
mode=f"{num_experts}_{num_experts_per_tok}_cbz_AL" #ce/cbz/cb/cz/15_3_cbz


args=ModelArgs(dim=768,hidden_dim=3072,num_labels=3,moe=MoeArgs(num_experts,num_experts_per_tok))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,args=args).to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')


ds_train = load_dataset('csv', data_files='data/tweet_eval/train.csv')
ds_val = load_dataset('csv', data_files='data/tweet_eval/val.csv')
ds_test = load_dataset('csv', data_files='data/tweet_eval/test.csv')

data = DatasetDict({
    'train': ds_train['train'].select(range(1280)),
    'valid': ds_val['train'],
    'test': ds_test['train']}  
)

print(data)

tokenized_dataset = data.map(preprocess_function, batched=True, num_proc=12)
tokenized_dataset = tokenized_dataset.remove_columns(["text", "split"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

tokenized_dataset.set_format("torch",columns=["global_index","input_ids", "attention_mask", "labels"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.combine([
                evaluate.load('accuracy'), 
                ConfiguredMetric(evaluate.load('f1'), average='macro'),
                ConfiguredMetric(evaluate.load('precision'), average='macro'),
                ConfiguredMetric(evaluate.load('recall'), average='macro'),
            ])

#intial 10 percent instances to train
incides_to_include=[]
l=len(data['train'])//10
for i in range(l):
    incides_to_include.append(data['train']['global_index'][i])

iter=0

while iter<10-1:
    print(f'this is {iter} iteration')
    print('training.....')
    matching_indices = [i for i, val in enumerate(tokenized_dataset["train"]["global_index"]) if val in incides_to_include]
    subset_dataset = Subset(tokenized_dataset["train"], matching_indices)
    train_dataloader = DataLoader(subset_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,)
    print(num_training_steps)

    progress_bar_train = tqdm(range(num_training_steps))
    model.train()

    for batch in train_dataloader:
        e = 'global_index'
        batch = {k: v.to(device) for k, v in batch.items() if k != e}
        outputs, selected_experts = model(**batch)
        loss = outputs.loss
        
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

 
    matching_indices = [i for i, val in enumerate(tokenized_dataset["train"]["global_index"]) if val not in incides_to_include]
    subset_dataset = Subset(tokenized_dataset["train"], matching_indices)
    eval_dataloader = DataLoader(subset_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    progress_bar_eval = tqdm(range(len(eval_dataloader)))

    # Evaluation step
    model.eval()
    dict_val={}
    print('validating....')
    for batch in eval_dataloader:
        e = 'global_index'
        glo_batch = {k: v for k, v in batch.items() if k == e}
        batch = {k: v.to(device) for k, v in batch.items() if k != e}
        with torch.no_grad():
            outputs, sel = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])


        logits_temp=logits
        softmax_values = F.softmax(logits_temp, dim=1)
        softmax_values=torch.max(softmax_values,dim=-1)

        for i in range(len(softmax_values.values)):
            dict_val[glo_batch['global_index'][i]]=softmax_values.values[i]
        
        sorted_dict_desc = dict(sorted(dict_val.items(), key=lambda item: item[1], reverse=True))

        if(len(sorted_dict_desc.keys())>l):
            keys = list(sorted_dict_desc.keys())[:l]
        else:
            key=list(sorted_dict_desc.keys())
        
        incides_to_include=incides_to_include+key

        # Calculate validation loss
        val_loss = outputs.loss.item()  # Assuming outputs has a loss attribute
        progress_bar_eval.update(1)
    
    iter=iter+1


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
print('testing....')
for batch in test_dataloader:
    key_to_exclude = 'global_index'
    glo_batch = {k: v for k, v in batch.items() if k == key_to_exclude}
    batch = {k: v.to(device) for k, v in batch.items() if k != key_to_exclude}
    with torch.no_grad():
        outputs, s = model(**batch)
        for k in s.keys():
            for i in range(len(s[k])):
                index=s[k][i]
                s[k][i]=glo_batch['global_index'][index].item()
        expert_list.append(s)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric_test.add_batch(predictions=predictions, references=batch["labels"])

results = metric_test.compute()
print(results)




with open(f"results/scores_tweet_eval_{mode}.json", "w") as jfile:
    json.dump(results, jfile)


result1 = {key:0 for key in range(num_experts)}
for d in expert_list:
    for key in result1.keys():
        result1[key] += len(d[key])

result2 = {key:[] for key in range(num_experts)}
for d in expert_list:
    for key in result2.keys():
        result2[key]=result2[key]+d[key]


# with open(f"results/expert_dist_{mode}_tweet_eval.json", "w") as jfile:
#     json.dump(result1, jfile)
# print('json_dist saved')


# with open(f"results/expert_mapped_{mode}_tweet_eval.json", "w") as jfile:
#     json.dump(result2, jfile)
# print('json_mapped saved')
