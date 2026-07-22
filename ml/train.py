"""
Fine-tune DistilBERT with two classification heads:
  - Emotion : single-label (CrossEntropy over softmax)
  - Topics  : multi-label  (BCEWithLogits over sigmoid)

Run:
    python train.py --data data_sample.csv --out ../ml-service/model --epochs 4

The exported directory contains:
    config.json, model.safetensors, tokenizer files, labels.json
It is loaded by ml-service/main.py.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from safetensors.torch import save_file

EMOTION_LABELS = ["Happy", "Angry", "Sad", "Frustrated", "Neutral"]
TOPIC_LABELS   = ["Pricing", "UI", "Service", "Quality", "Delivery", "Support"]


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Example:
    text: str
    emotion_id: int
    topic_vec: List[float]


class ReviewDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emotion": torch.tensor(ex.emotion_id, dtype=torch.long),
            "topics": torch.tensor(ex.topic_vec, dtype=torch.float),
        }


def load_examples(csv_path: str) -> List[Example]:
    df = pd.read_csv(csv_path)
    e2i = {e: i for i, e in enumerate(EMOTION_LABELS)}
    t2i = {t: i for i, t in enumerate(TOPIC_LABELS)}
    out: List[Example] = []
    for _, row in df.iterrows():
        vec = [0.0] * len(TOPIC_LABELS)
        for t in str(row["topics"]).split("|"):
            t = t.strip()
            if t in t2i:
                vec[t2i[t]] = 1.0
        out.append(Example(text=str(row["text"]),
                           emotion_id=e2i[row["emotion"]],
                           topic_vec=vec))
    return out


# ---------------------------------------------------------------------------
# Multi-head model
# ---------------------------------------------------------------------------
class MultiHeadTransformer(nn.Module):
    """DistilBERT backbone + two classification heads."""

    def __init__(self, backbone_name: str,
                 n_emotions: int, n_topics: int, dropout: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden, n_emotions)
        self.topic_head   = nn.Linear(hidden, n_topics)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler; use CLS token from last_hidden_state
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.emotion_head(cls), self.topic_head(cls)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    examples = load_examples(args.data)
    train_ex, val_ex = train_test_split(examples, test_size=0.2,
                                        random_state=42, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_ds = ReviewDataset(train_ex, tokenizer, args.max_len)
    val_ds   = ReviewDataset(val_ex, tokenizer, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = MultiHeadTransformer(args.backbone,
                                 n_emotions=len(EMOTION_LABELS),
                                 n_topics=len(TOPIC_LABELS)).to(device)

    ce_loss  = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(grouped, lr=args.lr)

    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    best_score = -1.0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emo  = batch["emotion"].to(device)
            top  = batch["topics"].to(device)

            optim.zero_grad()
            emo_logits, top_logits = model(ids, mask)
            loss = ce_loss(emo_logits, emo) + bce_loss(top_logits, top)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); scheduler.step()
            running += loss.item()

        # ---- validation ----
        model.eval()
        emo_true, emo_pred = [], []
        top_true, top_pred = [], []
        with torch.no_grad():
            for batch in val_dl:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                emo_logits, top_logits = model(ids, mask)
                emo_pred += emo_logits.argmax(-1).cpu().tolist()
                emo_true += batch["emotion"].tolist()
                top_pred += (torch.sigmoid(top_logits) > 0.5).int().cpu().tolist()
                top_true += batch["topics"].int().tolist()

        emo_acc = accuracy_score(emo_true, emo_pred)
        top_f1  = f1_score(top_true, top_pred, average="micro", zero_division=0)
        score   = (emo_acc + top_f1) / 2
        print(f"[epoch {epoch}] loss={running/len(train_dl):.4f} "
              f"emo_acc={emo_acc:.3f} topic_f1={top_f1:.3f}")

        if score > best_score:
            best_score = score
            save_artifacts(model, tokenizer, args.out, args.backbone)
            print(f"  saved to {args.out} (score={score:.3f})")

    print(f"[done] best score = {best_score:.3f}")


def save_artifacts(model: MultiHeadTransformer, tokenizer, out_dir: str, backbone_name: str):
    os.makedirs(out_dir, exist_ok=True)
    # 1) backbone + tokenizer via HF (config.json + tokenizer files)
    model.backbone.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    # 2) heads as separate safetensors file
    heads = {
        "emotion_head.weight": model.emotion_head.weight.detach().cpu(),
        "emotion_head.bias":   model.emotion_head.bias.detach().cpu(),
        "topic_head.weight":   model.topic_head.weight.detach().cpu(),
        "topic_head.bias":     model.topic_head.bias.detach().cpu(),
    }
    save_file(heads, os.path.join(out_dir, "heads.safetensors"))
    # 3) label metadata
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump({
            "backbone": backbone_name,
            "emotion_labels": EMOTION_LABELS,
            "topic_labels":   TOPIC_LABELS,
        }, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     default="data_sample.csv")
    p.add_argument("--out",      default="../ml-service/model")
    p.add_argument("--backbone", default="distilbert-base-uncased")
    p.add_argument("--epochs",     type=int,   default=4)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--max_len",    type=int,   default=128)
    p.add_argument("--lr",         type=float, default=2e-5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
