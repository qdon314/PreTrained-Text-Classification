from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_dir = "models/notebook-checkpoints"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Apple Silicon / MPS
import torch
if torch.backends.mps.is_available():
    device = "mps"
    model.to(device)
else:
    device = "cpu"

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1  # -1 = use model's device - already moved to mps/cpu
)

texts = [
    "This movie was absolutely fantastic!",
    "I hated every minute of this film."
]

preds = clf(texts)
for t, p in zip(texts, preds):
    print(f"{t!r} -> {p}")
