import torch, os, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from transformers import AutoModel, AutoConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from utils.helper import MoeArgs, MoeLayer, NewModel, ModelArgs
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from datasets import load_metric
from transformers.modeling_outputs import TokenClassifierOutput




#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model_name = "bert-base-uncased"
batch_size = 32


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# class CustomModel(nn.Module):
#   def __init__(self,checkpoint,args:ModelArgs):
#     super(CustomModel,self).__init__()

#     self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
#     self.moe_layer=MoeLayer(experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),moe_args=args.moe,)
#     self.classifier = nn.Linear(768,args.num_labels)

#   def forward(self, input_ids=None, attention_mask=None,labels=None):
#     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#     moe_outputs = self.moe_layer(outputs[0][:,0,:].view(-1,768))
#     logits=self.classifier(moe_outputs)

#     loss = None
#     if labels is not None:
#       loss_fct = nn.CrossEntropyLoss()
#       loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))

#     return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


# class CombinedModel(nn.Module):
#     def __init__(self, base_model, new_model):
#         super(CombinedModel, self).__init__()
#         self.base_model = base_model
#         self.new_model = new_model

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         #print("outputs:", outputs.hidden_states)
#         hidden_states = outputs.hidden_states[-1]
#         pooled_output = torch.mean(hidden_states, axis=1)
#         print(hidden_states.shape)
#         print(pooled_output.shape)
#         #print("hidden_states:", hidden_states)
#         logits = self.new_model(pooled_output)
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))

#         return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


class CombinedModel(nn.Module):
    def __init__(self, base_model, new_model):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.new_model = new_model

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.new_model(outputs, labels=labels)
        return logits

config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, num_labels=2, id2label=id2label, label2id=label2id)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_labels = 2
moe_args = MoeArgs(num_experts=4, num_experts_per_tok=2)
new_model = NewModel(num_labels, moe_args)

for name, param in model.named_parameters():
    if "bert" in name:
        param.requires_grad = False

combined_model = CombinedModel(model, new_model)

imdb = load_dataset("imdb")
encoded_dataset = imdb.map(preprocess_function, batched=True, num_proc=12)
encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch")
print(encoded_dataset["train"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


accuracy_score = load_metric("accuracy")
model_name = model_name.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-moe-r1",
    report_to="none",
    #logging_dir='./logs',
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    logging_steps=1,
    seed=42
    #push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)

trainer = Trainer(
    combined_model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.evaluate(encoded_dataset['test'])
#trainer.save_model("smnli-roberta-pretrained-model/")
print("_______________Training Done________________")

