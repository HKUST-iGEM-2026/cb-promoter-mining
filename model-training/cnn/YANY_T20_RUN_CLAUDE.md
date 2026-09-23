# CLAUDE.md — T20 (custom CNN): fix + run, one complete guide (for Yany)

> **Cluster setup (login, env, SLURM):** see `infrastructure/HPC4_GUIDE.md` — do that first.

You're helping **Yany** turn `model-training/cnn/train_cnn_template.py` into a finished, runnable
**`train_cnn.py`** that (1) loads the T16 CNN tokens, (2) consumes the per-sample `weight` correctly,
(3) selects on validation AUPRC, and (4) is a **single-config CLI trainer** the T21 sweep can drive.
This file is the one place for T20 — it merges the weight fix and the run steps.

> **Audited 2026-06 (AI council).** The original weight-fix's loss formula and line refs are correct, but
> three things it understated are folded in below — **don't skip them**: the **input shape / channel /
> length mismatch**, replacing **FocalLoss + F1-selection** with weighted-CE + AUPRC, and **removing the
> template's internal Optuna/wandb sweep** (T21 drives the sweep from the outside).

## The two contracts (identical to T18/T19 — keep them aligned)
- **Weighted loss.** Apply the T15 per-sample `weight` (silver 0.3 × the 50/50 class-balance factor) as a
  per-sample LOSS weight. This is the AT-trap guard.
- **Validation-AUPRC selection** (`sklearn.metrics.average_precision_score` on the positive-class prob —
  never F1/accuracy/AUROC) + the **sweep CLI** so T21 can call it.

## Why the current code is wrong (the bug)
T15 writes a per-row `weight` encoding **two things**: label confidence (silver T13 = 0.3; verified = 1.0)
**and** class balance (TRAIN negatives rescaled so train is effectively 50/50 → neg weight ≈ 0.56). T16
carries it into `results/T16/cnn/{split}.npz` as the `weight` array (row-aligned with `X`, `label`).

`train_cnn_template.py` **ignores it and does the opposite**: lines **538–544** build a
`WeightedRandomSampler` from **inverse class frequency** (`class_weights = 1.0/class_counts`). That throws
away the silver-0.3 down-weight (→ the CNN trains on ~75% silver positives = the AT-trap) **and**
double-counts class balance (T15 already balanced it). Fix = consume `npz["weight"]` as a loss weight and
delete the sampler.

## The changes (do ALL — `main()` and the train/val loops need a real rewrite, not a one-line patch)

**1. Load from the T16 npz; drop the parquet/encoder path.** Replace `pd.read_parquet` + `DNAEncoder` +
the string-based `CNNDataset` with a direct load:
```python
z = np.load(f"{args.data_dir}/{split}.npz", allow_pickle=False)   # data_dir = results/T16/cnn
X, y, w = z["X"], z["label"], z["weight"]      # X is (N, 81, 4) one-hot — confirmed in T16 tokenise.py
```
Drop `DNAEncoder`, the string augmentation, and the parquet reads.

**2. ⚠️ Fix the input shape — the silent killer.** T16's `X` is **(N, 81, 4)** = (batch, **length**,
**channels**). PyTorch `Conv1d` needs **(batch, channels, length) = (N, 4, 81)**. So **transpose**, and fix
the model dims:
```python
X = np.transpose(X, (0, 2, 1))     # (N,81,4) -> (N,4,81)   [or .permute(1,0) per item in the Dataset]
```
- `input_channels = 4`  (NOT 5 — T16 is one-hot ACGT, there is no 'N' channel)
- `sequence_length = 81` (NOT 300)
Change these in the `PromoterCNN`/`DilatedCNN`/`ResidualCNN` defaults **and** everywhere they're built
(`objective`/`main`). The template's 5-channel/300-length values are leftovers from the old 300 bp string path.

**3. Dataset returns `(x, label, weight)`:**
```python
class CNNDataset(Dataset):
    def __init__(self, X, y, w): self.X, self.y, self.w = X, y, w
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.float32),
                torch.tensor(self.y[i], dtype=torch.long),
                torch.tensor(self.w[i], dtype=torch.float32))
```

**4. Plain DataLoader — delete the sampler.** Remove lines **538–544** (the `class_counts` /
`class_weights` / `sample_weights` / `sampler` block) and the `sampler=sampler` arg:
```python
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
```

**5. Weighted CE loss — replace `FocalLoss`.** The criterion is `FocalLoss` (line 322), used with mean
reduction in `train_epoch` (L337) and `validate` (L371). Swap in per-sample weighted CE and unpack the
3-tuple:
```python
criterion = nn.CrossEntropyLoss(reduction="none")
for x, y_, w_ in train_loader:
    x, y_, w_ = x.to(dev), y_.to(dev), w_.to(dev)
    logits = model(x)
    loss = (criterion(logits, y_) * w_).sum() / w_.sum().clamp(min=1e-8)   # weighted mean
    loss.backward(); ...
```
**Eval unweighted** — val/test weights are natural prevalence, so just compute the metrics with no weighting.

**6. Select on val AUPRC + write metrics.json.** The template selects on `val_f1` and computes only AUROC.
Add AUPRC and select on it:
```python
from sklearn.metrics import average_precision_score
# collect probs = softmax(logits)[:,1] over the whole val set, then:
val_auprc = average_precision_score(y_true, probs)
# keep the epoch with the best val_auprc; at the end write:
json.dump({"model":"cnn", "trial":args.sweep_trial, "weighted":True,
           "val_auprc":val_auprc, "test_auprc":test_auprc},
          open(f"{args.output_dir}/metrics.json","w"), indent=2)
print(f"Validation AUPRC: {val_auprc:.4f}")     # T21 reads this (or metrics.json)
```

**7. ⚠️ Strip the internal Optuna + wandb + sqlite + 3-architecture loop from `main()`.** T21 runs the
sweep **externally** — it calls this script once per trial with CLI flags. The template's `main()` runs its
OWN `optuna.create_study(...)` over `["standard","dilated","residual"]` with `wandb` + a sqlite db, which
**conflicts** with being a single-config trainer. Rewrite `main()` to: parse args → load the 3 npz splits →
build ONE model from the CLI hyperparameters → train once (weighted loss + AUPRC selection) → write
metrics.json + save the model. **Remove `wandb`** (compute nodes are offline — it will hang; use
`WANDB_MODE=offline` if you want local logs).

**8. CLI — match what T21 passes** (the `cnn` block in `hyperparameter_sweep.py`):
```
--sweep-trial N  --output-dir <dir>  --data-dir results/T16/cnn
--learning-rate <f>  --batch-size <int>  --dropout-rate <f>   (+ --num-epochs --weight-decay --seed)
```
If you also sweep architecture lists (`num_filters`, `kernel_sizes`), accept them as strings like
`"32,64,128"` and parse inside (CLI can't pass a Python list).

## Run it (GPU job)
```bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=t20_cnn
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
cd /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"
python ../../../model-training/cnn/train_cnn.py \
  --data-dir results/T16/cnn \
  --output-dir /project/igem2026/cb-promoter-mining/results/T20
```
⚠️ Save `--output-dir` on the **shared `/project` clone**, not your `/home` — so T28 + the team can read the model.

## Verify
```python
import numpy as np, json
z = np.load("results/T16/cnn/train.npz", allow_pickle=False)
w, y = z["weight"], z["label"]
print("pos_mass", round(float(w[y==1].sum())), "neg_mass", round(float(w[y==0].sum())))   # ~equal => train 50/50
print("distinct weights", sorted({round(float(v),3) for v in w})[:6])  # ~{0.3 silver, ~0.56 train-neg, 1.0 verified}
m = json.load(open("results/T20/metrics.json"))
print("val_auprc", m["val_auprc"], "weighted", m["weighted"])          # weighted True; val_auprc >> 0.22
```
- `pos_mass ≈ neg_mass` → the CNN trains at ~50/50; the `0.3` confirms silver is down-weighted.
- val AUPRC well above the **~0.22** no-skill baseline (val is verified-only ~22% positive).
- The CNN is the ensemble **control** — it needn't beat the LMs, but must clearly beat chance.

## Hand to T21
With `train_cnn.py` honoring the CLI + writing `val_auprc`, T21 drives it:
`python model-training/hyperparameter-sweep/hyperparameter_sweep.py --models cnn …`. Sanity-check ONE trial first
(see `T21_SWEEP_RUN_CLAUDE.md`).

## What to commit / push
- **`train_cnn.py`** (your finished script) → commit it (hand the diff to Oak if HPC4 push 403s). Tell Oak
  whether to keep or remove `train_cnn_template.py` — don't silently delete it.
- **NOT** `results/T20/` (the trained CNN — git-ignored). It lives on shared `/project`; HF upload is post-T21.

## One-line summary
Rewrite the template into a single-config `train_cnn.py`: load the T16 npz (**transpose X to (N,4,81),
channels=4 / len=81**), return `(x,label,weight)`, **weighted-CE loss** (delete the inverse-frequency
sampler), **AUPRC selection** + metrics.json, and **strip the internal Optuna/wandb** so T21 can drive the
sweep → run on shared `/project`, verify `weighted=True` + AUPRC ≫ 0.22 → hand to T21.
