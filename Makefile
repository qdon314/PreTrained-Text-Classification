# -------------------------------------------------------------------
# Project Makefile for Pretrained Text Classification
# -------------------------------------------------------------------

# Variables
ENV_NAME = hf-seq-classification
MODEL_RUN = models/runs/imdb-distilbert-v1
NOTEBOOK = notebooks/evaluate.ipynb

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

env: # Create a new conda environment
	conda env create -f environment.yml

update-env: # Update the existing conda environment
	conda env update -f environment.yml --prune

activate: # Activate the conda environment
	@echo "Run this manually: conda activate $(ENV_NAME)"

# -------------------------------------------------------------------
# Training & Inference
# -------------------------------------------------------------------

train: # Train the model with default settings
	python scripts/train.py \
		--model_name distilbert-base-uncased \
		--dataset_name imdb \
		--output_dir $(MODEL_RUN) \
		--num_train_epochs 2 \
		--per_device_train_batch_size 4 \
		--load_best_model_at_end \
		--save_total_limit 2

inference: # Run inference with a pre-trained model
	python scripts/inference.py \
		--model_path $(MODEL_RUN) \
		--text "This movie was great!" "Awful plot."

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

evaluate: # Evaluate the model using a Jupyter notebook
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --inplace

notebook: # Start JupyterLab for interactive development
	jupyter lab

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------

test: # Run unit tests using pytest
	python -m pytest -vv

# -------------------------------------------------------------------
# Sanity Checks
# -------------------------------------------------------------------

check-python:  # Verify correct Python executable and version
	@echo "Python executable:"
	@which python
	@echo "Python version:"
	@python -c "import sys; print(sys.version)"
	@echo "sys.executable:"
	@python -c "import sys; print(sys.executable)"
	@echo "OK: Python sanity check completed."

check-imports:  # Verify all required packages import cleanly
	python -c "import transformers, datasets, evaluate, accelerate; import utils.data_set_utils; print('All imports successful.')"

check-python:
	python -c "import sys; print(sys.version); print(sys.executable)"

check-kernel:
	python -c "import sys; print('sys.executable:', sys.executable)"

check-hf:
	python -c "from huggingface_hub import HfFolder; print('HF token found.' if HfFolder.get_token() else 'No HF token configured.')"

check-models:  # Verify models/ directory exists
	@if [ -d "models" ]; then \
		echo "models/ directory exists."; \
		ls -R models; \
	else \
		echo "ERROR: models/ directory not found."; \
		exit 1; \
	fi

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

clean-checkpoints:
	rm -rf models/runs/*/checkpoint-* 2>/dev/null || true

clean: # Clean up generated files and directories
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Default help target
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?#' Makefile | sed 's/:.*?#/ - /'

.PHONY: env update-env activate train inference evaluate notebook test clean clean-checkpoints
.PHONY:	check-python check-imports check-kernel check-models check-hf help