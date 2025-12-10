from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.path_utils import get_model_path


def test_simple_inference_pipeline():
    model_dir = get_model_path( "notebook-checkpoints")

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
        
    assert len(preds) == len(texts)
    assert all(isinstance(p, dict) for p in preds)
    assert all('label' in p for p in preds)
    assert all('score' in p for p in preds)
