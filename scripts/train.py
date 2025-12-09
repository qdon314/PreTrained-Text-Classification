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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for text classification.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model checkpoint to start from.",
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
        help="Name of the text column in the dataset.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="Where to store checkpoints and final model.",
    )
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
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the final model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Repository name to push to (e.g. 'username/my-seq-classifier').",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset_name}")
    raw_datasets = load_dataset(args.dataset_name)

    # Make sure we have validation data
    if "validation" not in raw_datasets:
        # Simple split: take 10% of train as validation if not provided
        if "train" in raw_datasets:
            raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
            raw_datasets = {
                "train": raw_datasets["train"],
                "validation": raw_datasets["test"],
            }
        else:
            raise ValueError("Dataset has no 'validation' or 'train' split to derive one from.")

    # --- Tokenizer ---
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    print("Tokenizing dataset...")
    tokenized_datasets = {}
    for split in raw_datasets:
        tokenized_datasets[split] = raw_datasets[split].map(
            preprocess_function,
            batched=True,
            remove_columns=[c for c in raw_datasets[split].column_names if c != args.label_column],
        )

    # --- Labels ---
    # If labels are already integer-encoded, this will just pull num_classes.
    train_features = tokenized_datasets["train"].features
    label_feature = train_features[args.label_column]
    if hasattr(label_feature, "names"):
        num_labels = len(label_feature.names)
        id2label = {i: name for i, name in enumerate(label_feature.names)}
        label2id = {name: i for i, name in enumerate(label_feature.names)}
    else:
        # Fallback if labels are just ints without a ClassLabel feature
        all_labels = set(tokenized_datasets["train"][args.label_column])
        num_labels = len(all_labels)
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}

    print(f"Detected {num_labels} labels.")

    # --- Model ---
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # --- Metrics ---
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        report_to="none",  # change to "tensorboard" or "wandb" if you want logging
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save final model locally ---
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Push to hub if requested ---
    if args.push_to_hub:
        print("Pushing model to the Hugging Face Hub...")
        trainer.push_to_hub()

    print("Done.")


if __name__ == "__main__":
    main()
