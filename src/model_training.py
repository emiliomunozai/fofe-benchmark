import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataloaders import FOFETextDataset
from src.models.fofe import FOFENet

from src.models.bilstm import BiLSTMNet, count_parameters
@dataclass
class TrainConfig:
    max_len: int = 2048
    batch_size: int = 256
    emb_dim: int = 256
    hidden_dim: int = 256
    alpha: float = 0.7
    bidirectional: bool = True
    lr: float = 1e-3
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Sequence-level accuracy (% of correctly classified examples)."""
    correct = (preds == labels).sum().item()
    total = labels.shape[0]
    return 100.0 * correct / total if total > 0 else 0.0

def train_one_epoch(model, loader, optimizer, criterion, device):
    
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)                # (B, 2)
        loss = criterion(logits, y)      # y: (B,)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item()
        total_acc += compute_accuracy(preds, y)

    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            preds = logits.argmax(dim=-1)
            total_loss += loss.item()
            total_acc += compute_accuracy(preds, y)

    return total_loss / len(loader), total_acc / len(loader)

def train_one_bucket(bucket: list, bucket_max_len: int, vocab: dict, cfg: TrainConfig) -> float:
    
    set_seed(cfg.seed)
    random.shuffle(bucket)

    # ---- mild undersampling of majority class ----
    neg = [ex for ex in bucket if ex.label == 0]
    pos = [ex for ex in bucket if ex.label == 1]

    target_neg_ratio = 0.7  # keep ~65% negatives overall
    target_neg = int(target_neg_ratio / (1 - target_neg_ratio) * len(pos))
    target_neg = min(target_neg, len(neg))

    neg_sampled = random.sample(neg, target_neg)
    bucket = neg_sampled + pos
    random.shuffle(bucket)

    split_idx = int(0.8 * len(bucket))
    train_ex, val_ex = bucket[:split_idx], bucket[split_idx:]

    pad_id = vocab["<PAD>"]

    train_ds = FOFETextDataset(train_ex, max_len=bucket_max_len, pad_id=pad_id)
    val_ds   = FOFETextDataset(val_ex,   max_len=bucket_max_len, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    model = FOFENet(
        vocab_size=len(vocab),
        emb_dim=cfg.emb_dim,
        alpha=cfg.alpha,
        bidirectional=cfg.bidirectional,
        hidden_dim=cfg.hidden_dim,
        pad_id=pad_id,
    ).to(cfg.device)

    print(f"Number of parameters: {count_parameters(model)}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    # Optimizer change
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # Count labels
    num_pos = sum(ex.label for ex in train_ex)
    num_neg = len(train_ex) - num_pos

    # Inverse-frequency weights
    w_neg = len(train_ex) / (2 * num_neg) if num_neg > 0 else 1.0
    w_pos = len(train_ex) / (2 * num_pos) if num_pos > 0 else 1.0

    # Print class balance as percentages
    total = num_pos + num_neg
    percent_neg = (num_neg / total) * 100 if total > 0 else 0
    percent_pos = (num_pos / total) * 100 if total > 0 else 0
    # print(f"Class balance: neg={percent_neg:.2f}%, pos={percent_pos:.2f}%")
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(cfg.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

    print(f"Training bucket of size {len(bucket)} (train {len(train_ex)}, val {len(val_ex)})")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

    return val_acc

def train_one_bucket_lstm(bucket: list, bucket_max_len: int, vocab: dict, cfg: TrainConfig) -> float:
    
    set_seed(cfg.seed)
    random.shuffle(bucket)

    # ---- mild undersampling of majority class ----
    neg = [ex for ex in bucket if ex.label == 0]
    pos = [ex for ex in bucket if ex.label == 1]

    target_neg_ratio = 0.7
    target_neg = int(target_neg_ratio / (1 - target_neg_ratio) * len(pos))
    target_neg = min(target_neg, len(neg))

    neg_sampled = random.sample(neg, target_neg)
    bucket = neg_sampled + pos
    random.shuffle(bucket)

    split_idx = int(0.8 * len(bucket))
    train_ex, val_ex = bucket[:split_idx], bucket[split_idx:]

    pad_id = vocab["<PAD>"]

    train_ds = FOFETextDataset(train_ex, max_len=bucket_max_len, pad_id=pad_id)
    val_ds   = FOFETextDataset(val_ex,   max_len=bucket_max_len, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    # ---- BiLSTM model instead of FOFE ----
    model = BiLSTMNet(
        vocab_size=len(vocab),
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        bidirectional=True,
        pad_id=pad_id
    ).to(cfg.device)

    print(f"Number of parameters: {count_parameters(model)}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # ---- class weights ----
    num_pos = sum(ex.label for ex in train_ex)
    num_neg = len(train_ex) - num_pos

    w_neg = len(train_ex) / (2 * num_neg) if num_neg > 0 else 1.0
    w_pos = len(train_ex) / (2 * num_pos) if num_pos > 0 else 1.0

    total = num_pos + num_neg
    percent_neg = (num_neg / total) * 100 if total > 0 else 0
    percent_pos = (num_pos / total) * 100 if total > 0 else 0
    #print(f"Class balance: neg={percent_neg:.2f}%, pos={percent_pos:.2f}%")

    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(cfg.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    print(f"Training bucket of size {len(bucket)} (train {len(train_ex)}, val {len(val_ex)})")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, cfg.device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

    return val_acc
