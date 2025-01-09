from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch, os
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


@dataclasses.dataclass
class MoeArgs:
  def __init__(self, num_experts: int,num_experts_per_tok):
    self.num_experts=num_experts
    self.num_experts_per_tok=num_experts_per_tok

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for current_expert_index, current_expert in enumerate(self.experts):
            token_index, token_expert_index = torch.where(selected_experts == current_expert_index)
            results[token_index] += weights[token_index, token_expert_index, None] * current_expert(
                inputs[token_index]
            )
        return results


class ModelArgs:
    def __init__(self, dim: int, hidden_dim: int, num_labels: int, moe: MoeArgs):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.moe = moe


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class CustomModel(nn.Module):
  def __init__(self,checkpoint,args:ModelArgs):
    super(CustomModel,self).__init__()

    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    # for param in self.model.parameters():
    #     param.requires_grad = False
        
    self.moe_layer=MoeLayer(experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),moe_args=args.moe,)
    self.classifier = nn.Linear(768,args.num_labels)

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    moe_outputs = self.moe_layer(outputs[0][:,0,:].view(-1,768))
    logits=self.classifier(moe_outputs)

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))

    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512

args=ModelArgs(dim=768,hidden_dim=3072,num_labels=2,moe=MoeArgs(20,2))
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

metric = load_metric("f1")


progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))


for epoch in range(num_epochs):
  model.train()
  for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
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
        outputs = model(**batch)

    logits = outputs.logits
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
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())

