"""
VeriLLM Phase 7: Reproducible Training Pipeline for Claim-Evidence Verification on RAGTruth.
Fine-tunes a lightweight pretrained encoder (e.g., cross-encoder/nli-deberta-v3-xsmall or microsoft/deberta-v3-small)
with class-weighted Cross-Entropy loss, context windowing, validation evaluation, early stopping, and checkpoint saving.
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import precision_recall_fscore_support

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent

LABEL2ID = {
    "CONTRADICTED": 0,
    "SUPPORTED_PROXY": 1,
    "UNSUPPORTED": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Backward-compatibility aliases
LABEL_TO_ID = LABEL2ID
ID_TO_LABEL = ID2LABEL


class RAGTruthDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed RAGTruth JSONL records.
    Excludes MIXED_HALLUCINATION to preserve clean 3-class boundary.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                lbl = item.get("primary_label")
                if lbl in LABEL_TO_ID:
                    self.examples.append({
                        "context": item["context_text"],
                        "sentence": item["sentence_text"],
                        "label": LABEL_TO_ID[lbl]
                    })
                if max_samples and len(self.examples) >= max_samples:
                    break

        print(f"Loaded {len(self.examples)} records from {jsonl_path.name}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ctx = ex["context"]
        sent = ex["sentence"]
        lbl = ex["label"]

        # Context windowing: Truncate context if excessively long while preserving complete sentence
        ctx_words = ctx.split()
        if len(ctx_words) > 380:
            ctx = " ".join(ctx_words[:380])

        encoding = self.tokenizer(
            ctx,
            sent,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(lbl, dtype=torch.long)
        }


def compute_class_weights(dataset: RAGTruthDataset) -> torch.Tensor:
    """Computes inverse class frequency weights to counteract RAGTruth class imbalance."""
    counts = [0] * 3
    for ex in dataset.examples:
        counts[ex["label"]] += 1
    total = sum(counts)
    weights = [total / (3.0 * max(c, 1)) for c in counts]
    # Normalize
    mean_w = sum(weights) / 3.0
    norm_w = [w / mean_w for w in weights]
    print(f"Class counts: {counts} | Computed class weights: {[round(w, 2) for w in norm_w]}")
    return torch.tensor(norm_w, dtype=torch.float)


def evaluate(model, dataloader, device, criterion):
    """Evaluates model on validation dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_golds.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    p, r, f1, _ = precision_recall_fscore_support(all_golds, all_preds, average="macro", zero_division=0)
    per_class_f1 = precision_recall_fscore_support(all_golds, all_preds, average=None, zero_division=0)[2]

    return {
        "val_loss": round(avg_loss, 4),
        "macro_precision": round(p * 100, 2),
        "macro_recall": round(r * 100, 2),
        "macro_f1": round(f1 * 100, 2),
        "class_f1": {
            ID_TO_LABEL[i]: round(f * 100, 2) for i, f in enumerate(per_class_f1)
        }
    }


def train_pipeline(args):
    """Main training execution function."""
    print("=" * 80)
    print(" VeriLLM Phase 7: Model Training Pipeline on RAGTruth")
    print(f" Model Checkpoint : {args.model_name}")
    print(f" Output Directory : {args.output_dir}")
    print(f" Batch Size       : {args.batch_size}")
    print(f" Epochs           : {args.epochs}")
    print(f" Learning Rate    : {args.learning_rate}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer & Model
    print(f"Loading tokenizer and sequence classifier for: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(device)

    # Datasets
    train_dataset = RAGTruthDataset(
        Path(args.train_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples
    )
    val_dataset = RAGTruthDataset(
        Path(args.val_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_val_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Class-weighted loss
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_macro_f1 = 0.0
    history = []

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - t0

        print(f"\n--- Epoch {epoch} Results ({epoch_time:.1f}s) ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Macro-F1: {val_metrics['macro_f1']}% (Supported: {val_metrics['class_f1']['SUPPORTED_PROXY']}%, Contra: {val_metrics['class_f1']['CONTRADICTED']}%, Unsup: {val_metrics['class_f1']['UNSUPPORTED']}%)")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_metrics": val_metrics,
            "duration_seconds": round(epoch_time, 2)
        }
        history.append(epoch_record)

        # Checkpoint saving on improvement
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  -> New Best Model Saved! Val Macro-F1: {best_val_macro_f1}%\n")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "config": vars(args),
            "best_val_macro_f1": best_val_macro_f1,
            "history": history
        }, f, indent=2)

    print(f"\n[Complete] Training pipeline finished. Best Macro-F1: {best_val_macro_f1}%. Log: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="VeriLLM Phase 7 Training Script")
    parser.add_argument("--model_name", type=str, default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--train_path", type=str, default=str(BACKEND_DIR / "data" / "processed" / "ragtruth_train.jsonl"))
    parser.add_argument("--val_path", type=str, default=str(BACKEND_DIR / "data" / "processed" / "ragtruth_val.jsonl"))
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "checkpoint"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train_pipeline(parse_args())

