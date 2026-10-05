# Transformers for Movie Review Sentiment Analysis

This project implements a decoder-style Transformer encoder in PyTorch for binary sentiment classification on the IMDB movie-review dataset. It includes data preparation, a custom `IMDBDataset`, PyTorch `DataLoader`s, mean pooling, training/validation, test evaluation, checkpoint saving, and inference.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place `aclImdb_v1.tar.gz` (or an extracted `aclImdb` directory) in `data/`. If neither is present, the script downloads the IMDB dataset through Hugging Face Datasets.

## Train

```powershell
python train.py --data-dir data --epochs 3 --batch-size 32 --max-length 128
```

For a quick smoke test:

```powershell
python train.py --max-samples 1000 --epochs 1 --batch-size 16
```

The script prints descriptive statistics, validation accuracy, and final test accuracy. It saves `models/sentiment_model.pt` and `models/metrics.json`. A CUDA-enabled GPU is recommended; CPU training is functional but slow.

## Predict

```powershell
python predict.py "The acting was excellent and the story was genuinely moving."
```

## Project criteria covered

- Loads the original 25,000-review train and test splits.
- Uses a stratified 90/10 train-validation split.
- Provides descriptive statistics and label/review-length summaries.
- Implements `__init__`, `__len__`, and `__getitem__` for `IMDBDataset`.
- Implements multi-head causal self-attention, feed-forward blocks, layer normalization, mean pooling, and a two-class head.
- Calculates validation/test accuracy and saves a reusable checkpoint.

The model achieved 76.51% test accuracy in the reference three-epoch run, exceeding the project requirement of 75%. Exact results vary with hardware and seed; increasing `--epochs` or `--max-length` may change the result and increase training time.
