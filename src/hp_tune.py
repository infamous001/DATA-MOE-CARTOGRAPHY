from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction
import torch, numpy as np, os, pandas as pd, datasets, evaluate, sys, pickle, json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
#from scipy.special import softmax
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset

# from utils import data_utils as du

#os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

checkpoint = "roberta-base"
batch_size = 32
num_labels=2
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def preprocess_function(examples):
    return tokenizer(
        examples['text'], max_length=512, padding=True, truncation=True,
    )

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
    }

def model_init():
  return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

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
encoded_dataset = data.map(preprocess_function, batched=True, num_proc=12)
encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch",columns=["global_index","input_ids", "attention_mask", "labels"])
print(encoded_dataset["train"])




model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

model_name = checkpoint.split("/")[-1]

args = TrainingArguments(
    f"models/{model_name}-tuned",
    report_to="none",
    #logging_dir='./logs',
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    seed=42
    #push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset['validation'],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()

trainer = Trainer(
    model_init = model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['valid'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    # compute_objective=compute_objective,
)

best_hyperparameters = best_run.hyperparameters
with open("results/best_hyperparameters.json", "w") as f:
    json.dump(best_hyperparameters, f)
    
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)
trainer.train()

trainer.evaluate()
logits = trainer.predict(encoded_dataset['test'])

with open('results/scores.json', "w") as jfile:
    json.dump(logits.metrics, jfile)

print("_______________Hyperparameter Tuning Done________________")
