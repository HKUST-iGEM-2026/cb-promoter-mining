# task20_cnn_scratch/train_cnn.py
"""
Custom 1D CNN training from scratch for promoter prediction
Owner: Yany
Due: 26 Jun
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = X
        self.y = y
        self.w = w

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long),
            torch.tensor(self.w[i], dtype=torch.float32),
        )


class PromoterCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        sequence_length: int = 81,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        conv3_filters: int = 256,
        kernel_size: int = 7,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, conv1_filters, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(conv1_filters) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv2_filters) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv2_filters, conv3_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv3_filters) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv3_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass
class CNNConfig:
    data_dir: str
    output_dir: str
    sweep_trial: int = 0
    learning_rate: float = 1e-3
    batch_size: int = 64
    dropout_rate: float = 0.3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    patience: int = 8
    seed: int = 42
    num_workers: int = 4
    use_batch_norm: bool = True
    conv1_filters: int = 64
    conv2_filters: int = 128
    conv3_filters: int = 256
    kernel_size: int = 7


def load_split(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(data_dir) / f"{split}.npz"
    z = np.load(path, allow_pickle=False)
    X = z["X"]
    y = z["label"]
    w = z["weight"]

    if X.ndim != 3 or X.shape[1:] != (81, 4):
        raise ValueError(f"Expected X shape (N,81,4) in {path}, got {X.shape}")

    X = np.transpose(X, (0, 2, 1))
    return X.astype(np.float32), y.astype(np.int64), w.astype(np.float32)


def build_loaders(config: CNNConfig):
    train_X, train_y, train_w = load_split(config.data_dir, "train")
    val_X, val_y, val_w = load_split(config.data_dir, "val")
    test_X, test_y, test_w = load_split(config.data_dir, "test")

    train_ds = CNNDataset(train_X, train_y, train_w)
    val_ds = CNNDataset(val_X, val_y, val_w)
    test_ds = CNNDataset(test_X, test_y, test_w)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class CNNTrainer:
    def __init__(self, model: nn.Module, device: torch.device, config: CNNConfig):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
        )
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.best_model_path = Path(config.output_dir) / "best_model.pt"

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y, w in tqdm(loader, desc="Training", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            w = w.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(x)
            per_sample_loss = self.criterion(logits, y)
            loss = (per_sample_loss * w).sum() / w.sum().clamp(min=1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for x, y, _w in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y).mean()
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        metrics = {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
            "auprc": average_precision_score(all_labels, all_probs),
        }
        if len(set(all_labels)) > 1:
            metrics["auroc"] = roc_auc_score(all_labels, all_probs)
        else:
            metrics["auroc"] = float("nan")
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        history = []
        best_val_auprc = -float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            logger.info("Epoch %d/%d", epoch, self.config.num_epochs)
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
            }
            history.append(row)
            logger.info(
                "train_loss=%.4f train_f1=%.4f val_loss=%.4f val_auprc=%.4f val_auroc=%.4f",
                row["train_loss"], row["train_f1"], row["val_loss"], row["val_auprc"], row["val_auroc"]
            )

            if val_metrics["auprc"] > best_val_auprc:
                best_val_auprc = val_metrics["auprc"]
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered after epoch %d", epoch)
                    break

        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        return history, best_val_auprc


def parse_args() -> CNNConfig:
    ap = argparse.ArgumentParser(description="Train custom 1D CNN on T16 CNN tensors")
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing train/val/test.npz")
    ap.add_argument("--output-dir", type=str, required=True, help="Directory to save model and metrics")
    ap.add_argument("--sweep-trial", type=int, default=0, help="External sweep trial id")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dropout-rate", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-batch-norm", action="store_true", default=False)
    ap.add_argument("--conv1-filters", type=int, default=64)
    ap.add_argument("--conv2-filters", type=int, default=128)
    ap.add_argument("--conv3-filters", type=int, default=256)
    ap.add_argument("--kernel-size", type=int, default=7)
    args = ap.parse_args()
    return CNNConfig(**vars(args))


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Config: %s", json.dumps(asdict(config), indent=2))

    train_loader, val_loader, test_loader = build_loaders(config)

    model = PromoterCNN(
        input_channels=4,
        sequence_length=81,
        conv1_filters=config.conv1_filters,
        conv2_filters=config.conv2_filters,
        conv3_filters=config.conv3_filters,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
    )

    trainer = CNNTrainer(model, device, config)
    history, best_val_auprc = trainer.fit(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)

    metrics = {
        "model": "cnn",
        "trial": config.sweep_trial,
        "weighted": True,
        "val_auprc": best_val_auprc,
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "epochs_ran": len(history),
        "config": asdict(config),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Validation AUPRC: %.4f", best_val_auprc)
    logger.info("Test AUPRC: %.4f", test_metrics["auprc"])
    logger.info("Saved best model to %s", trainer.best_model_path)
    logger.info("Training complete")


if __name__ == "__main__":
    main()