import argparse
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

"""
train.py

Fine-tunes a HuggingFace Transformer model for text classification tasks.

This script supports:
    • Arbitrary pretrained model selection
    • Automatic dataset loading from the HuggingFace Hub
    • Tokenization and preprocessing
    • Configurable hyperparameters via command-line flags
    • Evaluation during training
    • Best-model selection using a chosen metric
    • Checkpoint pruning and run reproducibility
    • Optional push to HuggingFace Hub

Example:
    python scripts/train.py \
        --model_name distilbert-base-uncased \
        --dataset_name imdb \
        --output_dir models/runs/imdb-distilbert-v1

The script is intended to be called from the project root.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer model for text classification."
    )

    # Data / model
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base pretrained model checkpoint.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Hugging Face dataset name (e.g. 'imdb', 'ag_news').",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Text column name in the dataset.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Label column name in the dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0: set total number of training steps to perform (overrides num_train_epochs).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory.",
    )

    # Checkpointing / logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Checkpoint saving strategy.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps if save_strategy='steps'.",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps if evaluation_strategy='steps'.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit the total amount of checkpoints. Deletes the older checkpoints.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X updates.",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Whether to load the best model at the end of training.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="accuracy",
        help="Metric to use for best model selection.",
    )

    # Hub integration
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the model to the Hugging Face Hub at the end of training.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Repository name on the Hub (e.g. 'username/imdb-sentiment').",
    )
    parser.add_argument(
        "--hub_private_repo",
        action="store_true",
        help="Create a private repo on the Hub.",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Where to report logs. Use 'tensorboard' or 'wandb' if configured.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Dataset ----
    raw_datasets = load_dataset(args.dataset_name)

    if "validation" not in raw_datasets:
        if "train" in raw_datasets:
            split = raw_datasets["train"].train_test_split(test_size=0.1, seed=args.seed)
            raw_datasets = {
                "train": split["train"],
                "validation": split["test"],
            }
        else:
            raise ValueError("Dataset must have a 'train' split if no 'validation' is present.")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized_datasets = {}
    for split in raw_datasets:
        tokenized_datasets[split] = raw_datasets[split].map(
            preprocess_function,
            batched=True,
            remove_columns=[
                col
                for col in raw_datasets[split].column_names
                if col not in [args.label_column]
            ],
        )

    # ---- Labels ----
    train_features = tokenized_datasets["train"].features
    label_feature = train_features[args.label_column]

    if hasattr(label_feature, "names") and label_feature.names is not None:
        num_labels = len(label_feature.names)
        id2label = {i: name for i, name in enumerate(label_feature.names)}
        label2id = {name: i for i, name in enumerate(label_feature.names)}
    else:
        # fallback if labels are ints without a ClassLabel
        all_labels = sorted(set(tokenized_datasets["train"][args.label_column]))
        num_labels = len(all_labels)
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- Metrics ----
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    train_result = trainer.train()

    # ---- Save final/best model ----
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optionally push to hub
    if args.push_to_hub:
        trainer.push_to_hub()

    print("Training complete.")
    print("Metrics:", train_result.metrics)


if __name__ == "__main__":
    main()
