"""
This example is uses the official
huggingface transformers `hyperparameter_search` API.
"""
import os

import ray, evaluate, numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from datasets import load_dataset, load_metric
from datasets import Dataset, DatasetDict
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)


os.environ["HF_HOME"]="/home/pritam.k/research/huggingface"

def tune_transformer(num_samples=8, gpus_per_trial=0, smoke_test=False):
    # data_dir_name = "./data" if not smoke_test else "./test_data"
    # data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir, 0o755)

    # Change these as needed.
    model_name = (
        "roberta-base" if not smoke_test else "sshleifer/tiny-distilroberta-base"
    )
    # task_name = "rte"

    # task_data_dir = os.path.join(data_dir, task_name.upper())

    num_labels = 2

    # config = AutoConfig.from_pretrained(
    #     model_name, num_labels=num_labels, finetuning_task=task_name
    # )

    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Triggers tokenizer download to cache
    print("Downloading and caching pre-trained model")
    AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # # Download data.
    # download_data(task_name, data_dir)

    # data_args = GlueDataTrainingArguments(task_name=task_name, data_dir=task_data_dir)

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

    metric = evaluate.load('f1')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=".",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
    )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['valid'],
        compute_metrics=compute_metrics,
    )

    tune_config = {
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [16, 32, 64],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
    )

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        default=True,
        action="store_true",
        help="Finish quickly for testing",
    )
    args, _ = parser.parse_known_args()

    ray.init()

    if args.smoke_test:
        tune_transformer(num_samples=1, gpus_per_trial=0, smoke_test=True)
    else:
        # You can change the number of GPUs here:
        tune_transformer(num_samples=8, gpus_per_trial=1)