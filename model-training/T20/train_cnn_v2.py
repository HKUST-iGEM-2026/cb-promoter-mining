"""
Custom 1D CNN training from scratch for promoter prediction.

Advanced multi-encoding version with separate encoder branches for:
- one-hot nucleotide encoding (1D CNN)
- ENAC positional composition encoding (1D CNN)
- dinucleotide frequency encoding (MLP)
- trinucleotide frequency encoding (MLP)

The active feature set is configurable so you can run combinations such as:
- onehot,enac,trinuc
- onehot,enac
- onehot,dinuc,trinuc,enac

Owner: Yany
Last edited: 2026-07-31
"""

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

VALID_ENCODINGS = ("onehot", "dinuc", "trinuc", "enac")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in mapping:
            x[i, mapping[ch]] = 1.0
    return x


def build_kmer_index(k: int) -> Dict[str, int]:
    alphabet = ["A", "C", "G", "T"]
    index: Dict[str, int] = {}

    def rec(prefix: str, depth: int) -> None:
        if depth == 0:
            index[prefix] = len(index)
            return
        for a in alphabet:
            rec(prefix + a, depth - 1)

    rec("", k)
    return index


DINUC_INDEX = build_kmer_index(2)
TRINUC_INDEX = build_kmer_index(3)


def kmer_frequencies(seq: str, k: int, index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(4**k, dtype=np.float32)
    seq = seq.upper()
    total = max(len(seq) - k + 1, 1)
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if all(c in "ACGT" for c in kmer):
            vec[index[kmer]] += 1.0
    vec /= total
    return vec


def enac_matrix(seq: str, window: int = 5) -> np.ndarray:
    seq = seq.upper()
    length = len(seq)
    out = np.zeros((4, length), dtype=np.float32)
    half = window // 2

    for i in range(length):
        start = max(0, i - half)
        end = min(length, i + half + 1)
        sub = seq[start:end]
        denom = float(max(len(sub), 1))
        out[0, i] = sub.count("A") / denom
        out[1, i] = sub.count("C") / denom
        out[2, i] = sub.count("G") / denom
        out[3, i] = sub.count("T") / denom

    return out


class MultiEncodingSequenceDataset(Dataset):
    def __init__(
        self, sequences: List[str], y: np.ndarray, w: np.ndarray, enac_window: int = 5
    ):
        self.sequences = sequences
        self.y = y
        self.w = w
        self.enac_window = enac_window

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        seq = self.sequences[i]
        x_onehot = one_hot_encode_sequence(seq).T.astype(np.float32)  # (4, L)
        x_dinuc = kmer_frequencies(seq, 2, DINUC_INDEX).astype(np.float32)  # (16,)
        x_trinuc = kmer_frequencies(seq, 3, TRINUC_INDEX).astype(np.float32)  # (64,)
        x_enac = enac_matrix(seq, window=self.enac_window).astype(np.float32)  # (4, L)
        return (
            torch.tensor(x_onehot, dtype=torch.float32),
            torch.tensor(x_dinuc, dtype=torch.float32),
            torch.tensor(x_trinuc, dtype=torch.float32),
            torch.tensor(x_enac, dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long),
            torch.tensor(self.w[i], dtype=torch.float32),
        )


class ConvBranch1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv1_filters: int,
        conv2_filters: int,
        conv3_filters: int,
        dropout_rate: float,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, conv1_filters, kernel_size=7, padding=3),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.flatten(x, start_dim=1)


class MultiEncodingPromoterCNN(nn.Module):
    def __init__(
        self,
        active_encodings: Set[str],
        dinuc_dim: int = 16,
        trinuc_dim: int = 64,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        conv3_filters: int = 256,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.active_encodings = set(active_encodings)

        if "onehot" in self.active_encodings:
            self.onehot_branch = ConvBranch1D(
                input_channels=4,
                conv1_filters=conv1_filters,
                conv2_filters=conv2_filters,
                conv3_filters=conv3_filters,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
            )
            self.onehot_proj = nn.Sequential(
                nn.Linear(conv3_filters, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

        if "enac" in self.active_encodings:
            self.enac_branch = ConvBranch1D(
                input_channels=4,
                conv1_filters=conv1_filters,
                conv2_filters=conv2_filters,
                conv3_filters=conv3_filters,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
            )
            self.enac_proj = nn.Sequential(
                nn.Linear(conv3_filters, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

        if "dinuc" in self.active_encodings:
            self.dinuc_branch = nn.Sequential(
                nn.Linear(dinuc_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

        if "trinuc" in self.active_encodings:
            self.trinuc_branch = nn.Sequential(
                nn.Linear(trinuc_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

        fusion_input = fusion_dim * len(self.active_encodings)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 2),
        )

    def forward(
        self,
        x_onehot: torch.Tensor,
        x_dinuc: torch.Tensor,
        x_trinuc: torch.Tensor,
        x_enac: torch.Tensor,
    ) -> torch.Tensor:
        features = []

        if "onehot" in self.active_encodings:
            features.append(self.onehot_proj(self.onehot_branch(x_onehot)))
        if "dinuc" in self.active_encodings:
            features.append(self.dinuc_branch(x_dinuc))
        if "trinuc" in self.active_encodings:
            features.append(self.trinuc_branch(x_trinuc))
        if "enac" in self.active_encodings:
            features.append(self.enac_proj(self.enac_branch(x_enac)))

        fused = torch.cat(features, dim=1)
        return self.fusion_head(fused)


@dataclass
class CNNConfig:
    data_dir: str
    output_dir: str
    encodings: str = "onehot,dinuc,trinuc,enac"
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
    fusion_dim: int = 256
    enac_window: int = 5


def parse_encodings(encodings: str) -> Set[str]:
    active = {x.strip().lower() for x in encodings.split(",") if x.strip()}
    unknown = sorted(active - set(VALID_ENCODINGS))
    if unknown:
        raise ValueError(
            f"Unknown encodings: {unknown}; valid values are {VALID_ENCODINGS}"
        )
    if not active:
        raise ValueError("At least one encoding must be enabled")
    return active


def reconstruct_sequences_from_onehot(X: np.ndarray) -> List[str]:
    """Recover ACGTN strings from a T16 one-hot array (N, L, 4).

    T16 / encode.py write the CNN split as X=(N,81,4) one-hot with non-ACGT bases
    encoded as an all-zero row. argmax alone would map those zero rows to 'A', so we
    detect them (row sum == 0) and emit 'N'. This makes v2 consume the SAME
    results/T16/cnn/{split}.npz the production CNN uses — no separate raw-sequence
    export, and the two CNNs stay on identical data for a fair comparison.
    """
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[2] != 4:
        raise ValueError(f"expected one-hot X of shape (N, L, 4), got {X.shape}")
    lut = np.array(list("ACGT"))
    idx = X.argmax(axis=2)            # (N, L) -> 0..3
    has_base = X.sum(axis=2) > 0      # (N, L) True where some channel is set
    seqs = []
    for row_idx, row_has in zip(idx, has_base):
        seqs.append("".join(np.where(row_has, lut[row_idx], "N")))
    return seqs


def load_split(data_dir: str, split: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load a split as (sequences, labels, weights).

    Accepts either raw-sequence npz keys ('seq'/'sequence') or the T16 one-hot 'X'
    (N,L,4), which is reconstructed to strings so v2's multi-encoding features can be
    computed. label/weight carry the T15 silver-aware per-sample weight.
    """
    path = Path(data_dir) / f"{split}.npz"
    z = np.load(path, allow_pickle=False)
    if "seq" in z:
        seqs = [str(s) for s in z["seq"]]
    elif "sequence" in z:
        seqs = [str(s) for s in z["sequence"]]
    elif "X" in z:
        seqs = reconstruct_sequences_from_onehot(z["X"])
    else:
        raise ValueError(f"{path} must contain 'seq', 'sequence', or one-hot 'X'")
    y = z["label"]
    w = z["weight"]
    if not (len(seqs) == len(y) == len(w)):
        raise ValueError(f"{path}: length mismatch seq={len(seqs)} label={len(y)} weight={len(w)}")
    return seqs, y.astype(np.int64), w.astype(np.float32)


def build_loaders(config: CNNConfig):
    train_seq, train_y, train_w = load_split(config.data_dir, "train")
    val_seq, val_y, val_w = load_split(config.data_dir, "val")
    test_seq, test_y, test_w = load_split(config.data_dir, "test")

    train_ds = MultiEncodingSequenceDataset(
        train_seq, train_y, train_w, enac_window=config.enac_window
    )
    val_ds = MultiEncodingSequenceDataset(
        val_seq, val_y, val_w, enac_window=config.enac_window
    )
    test_ds = MultiEncodingSequenceDataset(
        test_seq, test_y, test_w, enac_window=config.enac_window
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
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

    def _forward_batch(self, batch):
        x1, x2, x3, x4, y, w = batch
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)
        x3 = x3.to(self.device, non_blocking=True)
        x4 = x4.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        w = w.to(self.device, non_blocking=True)
        logits = self.model(x1, x2, x3, x4)
        return logits, y, w

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in tqdm(loader, desc="Training", leave=False):
            logits, y, w = self._forward_batch(batch)
            self.optimizer.zero_grad()
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
            "mcc": matthews_corrcoef(all_labels, all_preds),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            logits, y, _w = self._forward_batch(batch)
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
            "mcc": matthews_corrcoef(all_labels, all_preds),
            "auprc": average_precision_score(all_labels, all_probs),
        }
        metrics["auroc"] = (
            roc_auc_score(all_labels, all_probs)
            if len(set(all_labels)) > 1
            else float("nan")
        )
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
                "train_mcc": train_metrics["mcc"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_mcc": val_metrics["mcc"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
            }
            history.append(row)
            logger.info(
                "train_loss=%.4f train_f1=%.4f train_mcc=%.4f val_loss=%.4f val_auprc=%.4f val_auroc=%.4f val_mcc=%.4f",
                row["train_loss"],
                row["train_f1"],
                row["train_mcc"],
                row["val_loss"],
                row["val_auprc"],
                row["val_auroc"],
                row["val_mcc"],
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

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        return history, best_val_auprc


# ─────────────────────────────────────────────────────────────────────
# Reusable inference API (imported by T22 predict_and_score + genome_inference)
# ─────────────────────────────────────────────────────────────────────


def build_model_from_config(cfg: Dict, device: str = "cpu") -> "MultiEncodingPromoterCNN":
    """Rebuild the multi-branch model from a saved config dict (config.json /
    metrics.json['config']) so a checkpoint's state_dict loads. The active
    encoding set is read from the config; forward() imposes a fixed branch order
    (onehot, dinuc, trinuc, enac), so a set is sufficient to reproduce the exact
    fusion layout."""
    if cfg.get("active_encodings"):
        active = set(cfg["active_encodings"])
    else:
        active = parse_encodings(cfg.get("encodings", "onehot,dinuc,trinuc,enac"))
    model = MultiEncodingPromoterCNN(
        active_encodings=active,
        conv1_filters=int(cfg.get("conv1_filters", 64)),
        conv2_filters=int(cfg.get("conv2_filters", 128)),
        conv3_filters=int(cfg.get("conv3_filters", 256)),
        dropout_rate=float(cfg.get("dropout_rate", 0.3)),
        use_batch_norm=bool(cfg.get("use_batch_norm", True)),
        fusion_dim=int(cfg.get("fusion_dim", 256)),
    )
    return model.to(device)


def _fit_to_window(seq: str, window: int) -> str:
    """Center-pad with 'N' or center-crop a sequence to exactly `window` bp, matching
    the CNN one-hot convention (one_hot_81) so the trained window is preserved at
    inference and ragged inputs (e.g. variable-length external-holdout promoters) batch
    cleanly. A no-op when len(seq) == window (the T16 / genome-window case)."""
    s = seq.upper()
    if len(s) < window:
        pad = window - len(s)
        s = ("N" * (pad // 2)) + s + ("N" * (pad - pad // 2))
    elif len(s) > window:
        off = (len(s) - window) // 2
        s = s[off:off + window]
    return s


def _featurize_batch(seqs: List[str], enac_window: int, window: int = 81):
    """Turn a list of sequences into the four model-input tensors (all four are
    built regardless of active encodings; unused ones are ignored by forward). Each
    sequence is first normalized to `window` bp so ragged inputs batch cleanly and
    match the trained length."""
    seqs = [_fit_to_window(s, window) for s in seqs]
    x1 = torch.tensor(np.stack([one_hot_encode_sequence(s).T for s in seqs]),
                      dtype=torch.float32)                                    # (B,4,L)
    x2 = torch.tensor(np.stack([kmer_frequencies(s, 2, DINUC_INDEX) for s in seqs]),
                      dtype=torch.float32)                                    # (B,16)
    x3 = torch.tensor(np.stack([kmer_frequencies(s, 3, TRINUC_INDEX) for s in seqs]),
                      dtype=torch.float32)                                    # (B,64)
    x4 = torch.tensor(np.stack([enac_matrix(s, window=enac_window) for s in seqs]),
                      dtype=torch.float32)                                    # (B,4,L)
    return x1, x2, x3, x4


@torch.no_grad()
def predict_proba(model: nn.Module, sequences: List[str], device: str,
                  enac_window: int = 5, batch_size: int = 256) -> np.ndarray:
    """P(positive) for each sequence. Shared by T22 and genome_inference so the
    feature pipeline at inference is byte-identical to training."""
    model.eval()
    out = []
    for i in range(0, len(sequences), batch_size):
        b = sequences[i:i + batch_size]
        x1, x2, x3, x4 = _featurize_batch(b, enac_window)
        logits = model(x1.to(device), x2.to(device), x3.to(device), x4.to(device))
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def parse_args() -> CNNConfig:
    ap = argparse.ArgumentParser(
        description="Train advanced multi-encoding CNN on promoter data"
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing train/val/test.npz",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save model and metrics",
    )
    ap.add_argument("--encodings", type=str, default="onehot,dinuc,trinuc,enac")
    ap.add_argument("--sweep-trial", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dropout-rate", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--enac-window", type=int, default=5)
    ap.add_argument(
        "--no-batch-norm", dest="use_batch_norm", action="store_false", default=True
    )
    ap.add_argument("--conv1-filters", type=int, default=64)
    ap.add_argument("--conv2-filters", type=int, default=128)
    ap.add_argument("--conv3-filters", type=int, default=256)
    ap.add_argument("--fusion-dim", type=int, default=256)
    args = ap.parse_args()
    return CNNConfig(**vars(args))


def main() -> None:
    config = parse_args()
    active_encodings = parse_encodings(config.encodings)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Active encodings: %s", ",".join(sorted(active_encodings)))
    logger.info("Config: %s", json.dumps(asdict(config), indent=2))

    train_loader, val_loader, test_loader = build_loaders(config)
    model = MultiEncodingPromoterCNN(
        active_encodings=active_encodings,
        conv1_filters=config.conv1_filters,
        conv2_filters=config.conv2_filters,
        conv3_filters=config.conv3_filters,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
        fusion_dim=config.fusion_dim,
    )

    trainer = CNNTrainer(model, device, config)
    history, best_val_auprc = trainer.fit(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)

    metrics = {
        "model": "cnn_multi_branch_fusion",
        "trial": config.sweep_trial,
        "weighted": True,
        "active_encodings": sorted(active_encodings),
        "val_auprc": best_val_auprc,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_mcc": test_metrics["mcc"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info("Saved best model to %s", output_dir / "best_model.pt")
    logger.info("Saved metrics to %s", output_dir / "metrics.json")
    logger.info("Final test metrics: %s", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
