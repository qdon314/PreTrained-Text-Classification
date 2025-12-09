Example usage

From the project root:

conda activate hf-seq-classification

```bash
python scripts/train.py \
  --model_name distilbert-base-uncased \
  --dataset_name imdb \
  --output_dir models/checkpoints \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8
```


If you want to push to the Hub:

``` bash
python scripts/train.py \
  --model_name distilbert-base-uncased \
  --dataset_name imdb \
  --output_dir models/checkpoints \
  --num_train_epochs 2 \
  --hub_model_id your-username/imdb-seq-classifier
```


