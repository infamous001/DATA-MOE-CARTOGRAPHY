import torch, os
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from transformers import AutoModel, AutoConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from utils.moe import MoeArgs, MoeLayer
from transformers import DataCollatorWithPadding
from datasets import load_dataset


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"

# Load the pre-trained model and freeze the desired layers
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Freeze the desired layers
for name, param in model.named_parameters():
    if "bert" in name:
        param.requires_grad = False

# Define the new trainable layers with MoE
class NewModel(nn.Module):
    def __init__(self, hidden_size, num_labels, moe_args):
        super(NewModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.moe_layer = MoeLayer(
            experts=[nn.Linear(hidden_size, num_labels) for _ in range(moe_args.num_experts)],
            gate=nn.Linear(hidden_size, moe_args.num_experts),
            moe_args=moe_args
        )

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        outputs = self.moe_layer(hidden_states)
        return outputs

# Instantiate the new model with MoE
num_labels = 2
moe_args = MoeArgs(num_experts=4, num_experts_per_tok=2)
new_model = NewModel(config.hidden_size, num_labels, moe_args)

# Combine the models
class CombinedModel(nn.Module):
    def __init__(self, base_model, new_model):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.new_model = new_model

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        print(outputs)
        hidden_states = outputs.hidden_states
        outputs = self.new_model(hidden_states)
        return outputs
    

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Instantiate the combined model with MoE
combined_model = CombinedModel(model, new_model)

# Define your training dataset and data collator
imdb = load_dataset("imdb")
tokenized_imdb = imdb.map(preprocess_function, batched=True, num_proc=12)
tokenized_imdb = tokenized_imdb.remove_columns(["text"])
tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
tokenized_imdb.set_format("torch")
print(tokenized_imdb["train"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Your data collator

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(combined_model.new_model.parameters(), lr=1e-5)
#loss_fn = nn.CrossEntropyLoss()

# Define the training arguments
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)

# Instantiate the trainer
trainer = Trainer(
    model=combined_model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    #optimizers=optimizer,
    tokenizer=tokenizer,
    #loss_fn=loss_fn,
    data_collator=data_collator
)

# Train the new layers with MoE
trainer.train()