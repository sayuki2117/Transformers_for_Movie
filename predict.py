"""Run inference with a trained sentiment model."""
import argparse
import torch
from transformers import AutoTokenizer
from train import DemoGPT

parser = argparse.ArgumentParser(); parser.add_argument("review"); parser.add_argument("--checkpoint", default="models/sentiment_model.pt")
args = parser.parse_args(); checkpoint = torch.load(args.checkpoint, map_location="cpu")
cfg = checkpoint["config"]; tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = DemoGPT(checkpoint["vocab_size"], cfg["max_length"]); model.load_state_dict(checkpoint["model"])
model.eval()
batch = tokenizer(args.review, max_length=cfg["max_length"], padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = model(batch["input_ids"], batch["attention_mask"])
    probabilities = torch.softmax(logits, dim=-1)[0]
label = "positive" if probabilities.argmax().item() == 1 else "negative"
print({"sentiment": label, "confidence": round(probabilities.max().item(), 4)})
