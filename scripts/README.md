# Script usage
**_All commands need to be run from the project root_**.
## Environment Prep
> [Install Miniconda if not already installed](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2)

```bash
conda env create -f environment.yml # If not already created
conda activate hf-seq-classification
```

Authenticate with your HuggingFace token to push to the Hub:
`huggingface-cli login`

___
## 1. Train.py

### Basic Execution
```bash
python scripts/train.py \
  --model_name distilbert-base-uncased \
  --dataset_name imdb \
  --output_dir models/checkpoints \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8
```

### If you want to push to the Hub:

``` bash
python scripts/train.py \
  --model_name distilbert-base-uncased \
  --dataset_name imdb \
  --output_dir models/checkpoints \
  --num_train_epochs 2 \
  --hub_model_id your-username/imdb-distilbert
```

### Lightweight run for M-series:
```bash
python scripts/train.py \
  --model_name distilbert-base-uncased \
  --dataset_name imdb \
  --output_dir models/imdb-distilbert \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_length 128 \
  --load_best_model_at_end \
  --save_total_limit 1
```

### With Hub flags:
```bash
python scripts/train.py \
  --model_name distilbert-base-uncased \
  --dataset_name imdb \
  --output_dir models/imdb-distilbert \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --save_total_limit 2 \
  --metric_for_best_model accuracy \
  --push_to_hub \
  --hub_model_id your-username/imdb-distilbert \
  --hub_private_repo
```
___
## Inference.py

### One line of text:
```bash
python scripts/inference.py \
  --model_path models/imdb-distilbert \
  --text "This movie was absolutely fantastic!" "I hated every minute of it."
```

### From Hub:
```bash
python scripts/inference.py \
  --model_path your-username/imdb-distilbert \
  --file sample_texts.txt
```
___