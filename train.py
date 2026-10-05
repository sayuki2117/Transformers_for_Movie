"""Train a small Transformer encoder for IMDB sentiment classification."""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_archive(data_dir: Path) -> Path | None:
    """Extract aclImdb_v1.tar.gz if needed and return the dataset root."""
    roots = [data_dir / "aclImdb", data_dir / "aclIMDB"]
    for root in roots:
        if (root / "train" / "pos").exists():
            return root
    archives = list(data_dir.glob("*.tar.gz")) + list(data_dir.glob("*.tgz"))
    if archives:
        with tarfile.open(archives[0], "r:gz") as archive:
            archive.extractall(data_dir)
        for root in roots:
            if (root / "train" / "pos").exists():
                return root
    return None


def load_dataset(folder: Path) -> list[str]:
    """Load UTF-8 review files from a directory in deterministic order."""
    return [p.read_text(encoding="utf-8", errors="replace") for p in sorted(folder.glob("*.txt"))]


def load_imdb(data_dir: str, max_samples: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = extract_archive(Path(data_dir))
    if root is None:
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("stanfordnlp/imdb")
        train = pd.DataFrame(ds["train"])
        test = pd.DataFrame(ds["test"])
    else:
        def frame(split: str) -> pd.DataFrame:
            pos = load_dataset(root / split / "pos")
            neg = load_dataset(root / split / "neg")
            return pd.DataFrame({"review": pos + neg, "label": [1] * len(pos) + [0] * len(neg)})
        train, test = frame("train"), frame("test")
    if max_samples:
        train = train.groupby("label", group_keys=False).head(max_samples // 2).reset_index(drop=True)
        test = test.groupby("label", group_keys=False).head(max_samples // 2).reset_index(drop=True)
    return train, test


class IMDBDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = frame["review"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(self.texts[index], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0), torch.tensor(self.labels[index], dtype=torch.long)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, max_length: int, dropout: float):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(embed_dim, head_dim, bias=False) for _ in range(3))
        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length, dtype=torch.bool)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        valid = self.mask[: x.size(1), : x.size(1)] & attention_mask[:, None, :].bool()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return weights @ v


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, max_length: int, dropout: float):
        super().__init__()
        head_dim = embed_dim // heads
        self.attention = nn.ModuleList([AttentionHead(embed_dim, head_dim, max_length, dropout) for _ in range(heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = torch.cat([head(self.norm1(x), mask) for head in self.attention], dim=-1)
        x = x + self.dropout(self.proj(y))
        return x + self.ff(self.norm2(x))


class DemoGPT(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embed_dim: int = 128, heads: int = 4, layers: int = 3, classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, max_length, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)


@torch.no_grad()
def calculate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct = total = 0
    for input_ids, mask, labels in loader:
        outputs = model(input_ids.to(device), mask.to(device))
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        correct += (logits.argmax(1).cpu() == labels).sum().item(); total += labels.numel()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional balanced subset for smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, test_df = load_imdb(args.data_dir, args.max_samples)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=args.seed, stratify=train_df.label)
    print(train_df.info()); print(train_df.label.value_counts()); print(train_df.review.str.len().describe())
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loaders = [DataLoader(IMDBDataset(df, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=name == "train", num_workers=0) for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]]
    model = DemoGPT(tokenizer.vocab_size, args.max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for input_ids, mask, labels in tqdm(loaders[0], desc=f"Epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids.to(device), mask.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f} val_accuracy={calculate_accuracy(model, loaders[1], device):.2f}%")
    test_accuracy = calculate_accuracy(model, loaders[2], device); print(f"test_accuracy={test_accuracy:.2f}%")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args), "vocab_size": tokenizer.vocab_size}, out / "sentiment_model.pt")
    (out / "metrics.json").write_text(json.dumps({"test_accuracy": test_accuracy}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
