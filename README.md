# **Sequence Classification with Hugging Face Transformers**

A full end-to-end project for building, training, evaluating, and deploying a state-of-the-art text classification model using the 🤗 Transformers and 🤗 Datasets libraries.

This project expands the official Hugging Face tutorial:
🔗 [Sequence Classication Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
and organizes it into a complete, production-ready workflow.

---

# **📌 Project Overview**

The goal of this project is to fine-tune a pretrained transformer model (e.g., BERT, RoBERTa, DistilBERT) on a text classification task such as sentiment analysis, topic classification, or spam detection.

This repository includes:

* Dataset loading and preprocessing
* Tokenization and input preparation
* Model fine-tuning with Trainer
* Metrics + error analysis
* Batch inference and deployment scripts
* Model saving, versioning, and Hub integration
* Experiment tracking
* Optional: ONNX export and optimization

---

# **📁 Repository Structure**

```
project-root/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_finetune_model.ipynb
│   └── 04_evaluation_and_error_analysis.ipynb
│
├── scripts/
│   ├── train.py             # training entrypoint
│   ├── evaluate.py          # evaluation entrypoint
│   ├── predict.py           # batch + single-text predictions
│   └── export_onnx.py       # optional ONNX conversion
│
├── data/
│   ├── raw/                 # untouched dataset
│   ├── processed/           # tokenized/imputed data
│   └── splits/              # reproducible train/val/test
│
├── models/
│   ├── checkpoints/         # intermediate checkpoints
│   ├── final/               # best model for production
│   └── hub/                 # ready-to-push Hugging Face repo
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── predictions.csv
│
├── utils/
│   ├── metrics.py           # accuracy, f1, etc.
│   ├── plotting.py          # confusion matrix, ROC, etc.
│   └── preprocessing.py     # dataset transforms
│
├── environment.yml or requirements.txt
└── README.md
```

---

# **⚙️ Setup**

### **1. Create & activate environment**

> [Install Miniconda if not already installed](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2)

```bash
conda env create -f environment.yml # If not already created
conda activate hf-seq-classification
```

__Authenticate with your HuggingFace token to push to the Hub__:
```bash
huggingface-cli login
```

---

# **🧠 Training Workflow**

## **1. Load dataset**

Example: IMDB sentiment classification

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

## **2. Tokenize**

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
```

## **3. Fine-tune the model**

```bash
python scripts/train.py \
  --model_name bert-base-uncased \
  --dataset_name imdb \
  --output_dir models/checkpoints \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5
```

Training includes:

* mixed precision (AMP)
* evaluation at end of each epoch
* optional logging via TensorBoard or Weights & Biases

---

# **📊 Evaluation & Analysis**

Run:

```bash
python scripts/evaluate.py \
  --model_dir models/final \
  --dataset_name imdb
```

Produces:

* Accuracy, Precision, Recall, F1
* Confusion matrix (`results/confusion_matrix.png`)
* Classification report
* Misclassified example dump

Error analysis details which examples the model struggles with (sarcasm, negation, domain shifts, etc.).

---

# **🔍 Inference**

### **Single text**

```bash
python scripts/predict.py --text "I loved this movie!"
```

### **Batch inference**

```bash
python scripts/predict.py --file path/to/file.csv
```

Outputs a CSV with predicted labels + probabilities.

---

# **☁️ Pushing to Hugging Face Hub**

1. Login:

```bash
huggingface-cli login
```

2. Push:

```python
trainer.push_to_hub()
```

or manually:

```python
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

The final model lives under `models/hub/`.

---

# **🚀 Deployment Options**

### Option A — Use Hugging Face Inference API

Instant hosting via:

```
https://huggingface.co/<username>/<model>/api
```

### Option B — Local FastAPI inference server

### Option C — Export to ONNX and serve with ONNX Runtime

```bash
python scripts/export_onnx.py
```

---

# **🧪 Experiment Tracking**

You can optionally integrate:

* **TensorBoard**
* **Weights & Biases**
* **MLflow**

Example Trainer config:

```python
TrainingArguments(
    ...,
    report_to="tensorboard",
)
```

---

# **🔁 Reproducibility**

This repo uses:

* fixed seeds
* stored environment files (requirements/conda env)
* deterministic train/val/test splits
* versioned checkpoints
* `config.json` for model metadata

---

# **🙌 Contributing**

Contributions welcome. Ideas include:

* adding new datasets
* experimenting with different architectures
* adding ONNX Runtime benchmarks
* exploring quantization
* improving evaluation visualizations

---

# **📎 References**

* Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
* Hugging Face Datasets: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
* Evaluation: [https://huggingface.co/docs/evaluate](https://huggingface.co/docs/evaluate)
* Accelerate: [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate)

---
