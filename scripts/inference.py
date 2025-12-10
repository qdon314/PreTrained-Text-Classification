import argparse
from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
inference.py

Provides programmatic and command-line interfaces for running inference using a
fine-tuned text classification model.

Functions:
    • load_model_and_tokenizer(): Load a model from disk or HuggingFace Hub
    • classify_texts(): Run batched inference and return predicted labels/scores

CLI Usage:
    python scripts/inference.py \
        --model_path models/runs/imdb-distilbert-v1 \
        --text "This movie was great!" "Terrible acting."

Notes:
    This script automatically selects MPS, CUDA, or CPU depending on availability.
"""


def load_model_and_tokenizer(model_path_or_hub_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_hub_id)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def classify_texts(
    texts: Union[str, List[str]],
    model_path_or_hub_id: str,
    max_length: int = 256,
):
    if isinstance(texts, str):
        texts = [texts]

    model, tokenizer, device = load_model_and_tokenizer(model_path_or_hub_id)

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # Don't need gradients for inference - no backpropagation
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)

    pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
    probs = probs.cpu().tolist()

    id2label = model.config.id2label

    results = []
    for text, pred_id, prob_vec in zip(texts, pred_ids, probs):
        label = id2label[int(pred_id)]
        score = float(prob_vec[int(pred_id)])
        results.append({"text": text, "label": label, "score": score})

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned text classification model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory or a Hub repo ID.",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        help="Text(s) to classify (space-separated).",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Optional: path to a text file, one example per line.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    inputs = []
    if args.text:
        inputs.extend(args.text)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            inputs.extend([line.strip() for line in f if line.strip()])

    if not inputs:
        raise SystemExit("Please provide --text or --file.")

    results = classify_texts(
        texts=inputs,
        model_path_or_hub_id=args.model_path,
        max_length=args.max_length,
    )

    for r in results:
        print(f"TEXT: {r['text']}")
        print(f"  -> LABEL: {r['label']} (score={r['score']:.4f})")
        print("-" * 60)


if __name__ == "__main__":
    main()
