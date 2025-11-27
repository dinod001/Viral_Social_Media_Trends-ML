.PHONY: install data train train-regression train-classification train-clustering inference clean help

PYTHON = python
PIP = pip

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) pipelines/data_pipeline.py

train:
	$(PYTHON) pipelines/training_pipeline.py

train-regression:
	@echo "Running Regression Training..."
	@echo "regression" | $(PYTHON) pipelines/training_pipeline.py

train-classification:
	@echo "Running Classification Training..."
	@echo "classification" | $(PYTHON) pipelines/training_pipeline.py

train-clustering:
	@echo "Running Clustering Training (PCA)..."
	@echo "clustering\npca" | $(PYTHON) pipelines/training_pipeline.py

inference:
	$(PYTHON) pipelines/inference_pipeline.py

clean:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

help:
	@echo "Available commands:"
	@echo "  make install              - Install dependencies"
	@echo "  make data                 - Run the data pipeline"
	@echo "  make train                - Run the training pipeline (interactive)"
	@echo "  make train-regression     - Run regression training (non-interactive)"
	@echo "  make train-classification - Run classification training (non-interactive)"
	@echo "  make train-clustering     - Run clustering training (default: PCA)"
	@echo "  make inference            - Run the inference pipeline"
	@echo "  make clean                - Remove temporary files (__pycache__, .pyc)"
	@echo "  make help                 - Show this help message"
