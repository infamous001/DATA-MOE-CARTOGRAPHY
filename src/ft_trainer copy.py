from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction
import torch, numpy as np, os, pandas as pd, datasets, evaluate, sys, pickle, json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
#from scipy.special import softmax
from evaluate import load
from utils import data_utils as du
from custom_models import CustomRobertaForSequenceClassification, CustomAlbertForSequenceClassification, \
    CustomBertForSequenceClassification, CustomDistilBertForSequenceClassification, CustomXLNetForSequenceClassification
from datasets import concatenate_datasets, load_dataset


task = "csnli"
model_checkpoint = "roberta-base"
batch_size = 32

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"


task_to_keys = {
    "csnli": ("premise", "hypothesis"),
    "cmnli": ("premise", "hypothesis"),
    "canli": ("obs1", "obs2", "hyp1", "hyp2"),
    
}

def preprocess_function_smnli(examples):
        return tokenizer(
            examples['premise'], examples['hypothesis'], max_length=128, truncation=True,
        )



fname="/home/pritam.k/research/hlv/models/ecart/datasets_diff_test/csnli/100-annotator-sets/m100s1.tsv"
metric = evaluate.load('glue', 'mnli')
#dataset_test = du.extractData(task, fname)
df_test=pd.read_csv(fname, sep='\t')
path="/home/pritam.k/research/hlv/datasets/glue/"
df_train = pd.read_csv(path+"df-snli-mnli-train.csv")
df_valid = pd.read_csv(path+"df-snli-test.csv")

def maplabels(val):
    if val=='entailment':
        return 0
    elif val=='neutral':
        return 1
    elif val=='contradiction':
        return 2

df_train = df_train.dropna()
df_valid = df_valid.dropna()
df_train['label'] = df_train['label'].apply(maplabels)
df_train = df_train.dropna()
df_valid['label'] = df_valid['label'].apply(maplabels)
df_valid = df_valid.dropna()
df_test['label'] = df_test['label'].apply(maplabels)
df_train['label'] = df_train['label'].astype('int')
df_valid['label'] = df_valid['label'].astype('int')
df_test['label'] = df_test['label'].astype('int')
df_test['idx'] = [i for i in range(len(df_test))]
df_train = df_train.sample(frac=1, random_state=42)
train = Dataset.from_pandas(df_train)
dev = Dataset.from_pandas(df_valid)
test = Dataset.from_pandas(df_test)
columns_to_remove = ['split', 'label_dist', 'entropy', 'continuous_disagreement_score']
test = test.remove_columns(columns_to_remove)
train = train.remove_columns('__index_level_0__')
dev = dev.remove_columns('__index_level_0__')

dataset = DatasetDict()

dataset['train'] = train
dataset['validation'] = dev
dataset['test'] = test
print(dataset)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)



encoded_dataset = dataset.map(preprocess_function_smnli, batched=True, num_proc=12)
#encoded_dataset_test = combined_dataset.map(preprocess_function_smnli, batched=True, num_proc=12)
encoded_dataset
#encoded_dataset = datasets.load_from_disk("../data/encoded_data/anli")

num_labels = 2 if task.startswith("canli") else 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{task}-{model_name}-pretrained",
    report_to="none",
    #logging_dir='./logs',
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    seed=42
    #push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.evaluate(encoded_dataset['test'])
#trainer.save_model("smnli-roberta-pretrained-model/")
print("_______________Training Done________________")

