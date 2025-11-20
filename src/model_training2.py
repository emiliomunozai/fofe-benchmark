# src/model_training2.py
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataloaders import FOFETextDataset
from src.models.fofe import FOFENet
from src.models.bilstm import BiLSTMNet, count_parameters


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    max_len: int = 2048
    batch_size: int = 128
    emb_dim: int = 256
    hidden_dim: int = 256
    alpha: float = 0.9
    bidirectional: bool = True
    lr: float = 3e-4
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    correct = (preds == labels).sum().item()
    total = labels.shape[0]
    return 100.0 * correct / total if total > 0 else 0.0


# -------------------------
# Training Loops
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        logits = model(x).squeeze(-1)  # (B,)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, y.long())

    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            total_loss += loss.item()
            total_acc += compute_accuracy(logits, y.long())

    return total_loss / len(loader), total_acc / len(loader)


# -------------------------
# FOFE Training
# -------------------------
def train_one_bucket(bucket: list, bucket_max_len: int, vocab: dict, cfg: TrainConfig):
    set_seed(cfg.seed)
    random.shuffle(bucket)

    # No undersampling – keep full data
    neg = [ex for ex in bucket if ex.label == 0]
    pos = [ex for ex in bucket if ex.label == 1]
    bucket = neg + pos
    random.shuffle(bucket)

    split_idx = int(0.8 * len(bucket))
    train_ex, val_ex = bucket[:split_idx], bucket[split_idx:]

    pad_id = vocab["<PAD>"]

    train_ds = FOFETextDataset(train_ex, max_len=bucket_max_len, pad_id=pad_id)
    val_ds = FOFETextDataset(val_ex, max_len=bucket_max_len, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = FOFENet(
        vocab_size=len(vocab),
        emb_dim=cfg.emb_dim,
        alpha=cfg.alpha,
        bidirectional=cfg.bidirectional,
        hidden_dim=cfg.hidden_dim,
        pad_id=pad_id,
        num_classes=1,  # single logit
    ).to(cfg.device)

    print(f"Number of parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Class weights
    num_pos = sum(ex.label for ex in train_ex)
    num_neg = len(train_ex) - num_pos
    w_neg = len(train_ex) / (2 * num_neg) if num_neg > 0 else 1.0
    w_pos = len(train_ex) / (2 * num_pos) if num_pos > 0 else 1.0

    pos_weight = torch.tensor([w_pos / w_neg]).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Training bucket of size {len(bucket)} (train {len(train_ex)}, val {len(val_ex)})")

    best_val = float("inf")
    patience = 3
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return val_acc


# -------------------------
# BiLSTM Training
# -------------------------
def train_one_bucket_lstm(bucket: list, bucket_max_len: int, vocab: dict, cfg: TrainConfig):
    set_seed(cfg.seed)
    random.shuffle(bucket)

    neg = [ex for ex in bucket if ex.label == 0]
    pos = [ex for ex in bucket if ex.label == 1]
    bucket = neg + pos
    random.shuffle(bucket)

    split_idx = int(0.8 * len(bucket))
    train_ex, val_ex = bucket[:split_idx], bucket[split_idx:]

    pad_id = vocab["<PAD>"]

    train_ds = FOFETextDataset(train_ex, max_len=bucket_max_len, pad_id=pad_id)
    val_ds = FOFETextDataset(val_ex, max_len=bucket_max_len, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = BiLSTMNet(
        vocab_size=len(vocab),
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        bidirectional=True,
        pad_id=pad_id,
        num_classes=1,  # single logit
    ).to(cfg.device)

    print(f"Number of parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    num_pos = sum(ex.label for ex in train_ex)
    num_neg = len(train_ex) - num_pos
    w_neg = len(train_ex) / (2 * num_neg) if num_neg > 0 else 1.0
    w_pos = len(train_ex) / (2 * num_pos) if num_pos > 0 else 1.0

    pos_weight = torch.tensor([w_pos / w_neg]).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Training bucket of size {len(bucket)} (train {len(train_ex)}, val {len(val_ex)})")

    best_val = float("inf")
    patience = 3
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return val_acc
