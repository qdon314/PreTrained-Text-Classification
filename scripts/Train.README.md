##Example usage

From the project root:

```bash
conda env create -f environment.yml # If not already created
conda activate hf-seq-classification
```

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
  --hub_model_id your-username/imdb-seq-classifier
```

### Lightweight run for M-series
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

