#import the libraries
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
from utils.helper import ConfiguredMetric
from torch.nn import DataParallel

#setting up seed
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def preprocess_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,max_length=128)


#making custom model with same ffn and classifier head as MOE
class ModelArgs:
    def __init__(self, dim: int, hidden_dim: int, num_labels: int):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class CustomModel(nn.Module):
  def __init__(self,checkpoint,args: ModelArgs):
    super(CustomModel,self).__init__()
    self.args=args
    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    for param in self.model.parameters():
        param.requires_grad = False
    self.ffn=FeedForward(args=args) 
    self.classifier = nn.Linear(768,args.num_labels)

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    ffn_outputs = self.ffn(outputs[0][:,0,:].view(-1,768))
    logits=self.classifier(ffn_outputs)

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.args.num_labels), labels.view(-1))

    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)



os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
checkpoint = "roberta-base"
mode="mnli_no_moe_f" #ce/cbz/cb/cz/no_moe
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


with open("results/best_hyperparameters_tweet_eval.json", "r") as jfile:
    hp_params=json.load(jfile)

lr=hp_params['learning_rate']
batch_size=hp_params['per_device_train_batch_size']
# batch_size=32
num_epochs=hp_params['num_train_epochs']
num_labels=3

args=ModelArgs(dim=768,hidden_dim=3072,num_labels=num_labels)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,args=args)
model.to(device)

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     model = DataParallel(model)
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')


data=load_dataset('nyu-mll/multi_nli')

print(data)
#sys.exit()
tokenized_dataset = data.map(preprocess_function, batched=True, num_proc=12)
# tokenized_dataset = tokenized_dataset.remove_columns(["text", "split"])
tokenized_dataset = tokenized_dataset.remove_columns(["premise","hypothesis", "promptID","pairID","premise_binary_parse","premise_parse","hypothesis_binary_parse","hypothesis_parse","genre"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
#tokenized_dataset.set_format("torch")
tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "labels"])
print(tokenized_dataset["train"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["validation_matched"], batch_size=batch_size, collate_fn=data_collator
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

metric = evaluate.load("accuracy")


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
        outputs = model(**batch)
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
            outputs = model(**batch)
        
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
plt.savefig(f'plots/Loss_{mode}_imdb.png')
plt.show()

metric_test = evaluate.load("accuracy")
test_dataloader = DataLoader(
    tokenized_dataset["validation_mismatched"], batch_size=batch_size, collate_fn=data_collator
)
# expert_list=[]
model.eval()
for batch in test_dataloader:
    key_to_exclude = 'global_idx'
    #glo_batch = {k: v for k, v in batch.items() if k == key_to_exclude}
    batch = {k: v.to(device) for k, v in batch.items() if k != key_to_exclude}
    with torch.no_grad():
        outputs = model(**batch)
        # for k in s.keys():
        #     for i in range(len(s[k])):
        #         index=s[k][i]
        #         s[k][i]=glo_batch['global_index'][index].item()
        # expert_list.append(s)

    logits = outputs.logits
    #print(f"test: {s}")
    predictions = torch.argmax(logits, dim=-1)
    metric_test.add_batch(predictions=predictions, references=batch["labels"])


# with open("results/expert_dist.pkl", "wb") as pfile:
#     pickle.dump(expert_list, pfile)
results = metric_test.compute()
print(results)

with open(f"results/scores_{mode}.json", "w") as jfile:
    json.dump(results, jfile)


# result1 = {0: 0, 1: 0, 2: 0}
# for d in expert_list:
#     for key in result1.keys():
#         result1[key] += len(d[key])

# result2 = {0: [], 1: [], 2: []}
# for d in expert_list:
#     for key in result2.keys():
#         result2[key]=result2[key]+d[key]

# df_dist = pd.DataFrame(result1.items(), columns=['expert', 'num_of_mapped_instances'])
# df_dist.to_csv(f'results/expert_dist_{mode}_tweet_eval.csv', index=False)
# print('df_dist saved')

# df_dist = pd.DataFrame(result2.items(), columns=['expert', 'mapped_instances'])
# df_dist.to_csv(f'results/expert_mapped_{mode}_tweet_eval.csv', index=False)
# print('df_mapped saved')