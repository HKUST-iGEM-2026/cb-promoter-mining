# CLAUDE.md — T16 (per-model tokenisation): complete, self-contained step-by-step for Leo

> **Cluster setup (login, env, SLURM):** see `infrastructure/HPC4_GUIDE.md` — do that first.

You are a Claude instance helping **Leo** run **Task T16** of the HKUST iGEM 2026 dry-lab "promoter
mining" pipeline. The script is **written, bug-fixed, and smoke-tested** (`dataset-construction/T16/
tokenise.py`). **Every decision is already made and written below — Leo does not need to ask anyone
(not Sanaa, not Yany, not Oak) anything.** Your job: run it on HPC4, verify the output, push the
script. Follow `prompt-guideline/CLAUDE.md` (think first, simplest thing, surgical changes, verify
against a goal).

---

## Context — what T16 is and why it matters (30 sec)

The team scores short DNA windows for whether they're **promoters**, mining _Clostridium butyricum_.
The classifier is an **ensemble of 3 models**, each needing DNA in a *different* numeric format:
- **ProkBERT-mini** (T18) → its own 6-mer token IDs.
- **Nucleotide Transformer v2 250M** (T19) → its own 6-mer token IDs.
- **Custom 1D CNN** (T20) → one-hot encoding.

T16 reads the finished **T15 splits** (already labelled, balanced, leakage-free, 81 bp) and emits the
tokenised tensors each model trains on — **carrying every metadata column through unchanged** and
**never re-splitting, re-windowing, re-balancing, or reordering rows**. It's a thin but load-bearing
adapter: if the tokens don't match what a model was pre-trained on, that fine-tune learns garbage; if
`weight`/`cluster_id`/`sigma_factor` get dropped, the silver-weighting (AT-trap guard), the
cluster-aware CV, and the T25 sigma-stratified eval all break.

## Status of the script (already done — don't rewrite it)

`tokenise.py` is written, **independently audited, and smoke-tested** (synthetic T15 → row count and
order preserved, one-hot correct, metadata byte-identical incl. `weight=0.3`, empty-split safe). Two
bugs the audit found are **already fixed**: (1) the pad width is now fitted over *all* splits so val/
test can't be silently truncated; (2) `cluster_id` is stored in the `.npz` as a string so
`np.load` doesn't choke. It also now carries **all** T15 columns (incl. `sigma_factor`). The only
thing not yet exercised is the *real* ProkBERT/NT tokenisation — that needs the cached checkpoints,
which is why it runs on HPC4. So your work is Steps 1–7 below.

---

## STEP-BY-STEP

### Step 0 — Prerequisites
T15 is done: `results/T15/{train,val,test}.parquet` exist on the **shared** clone
`/project/igem2026/cb-promoter-mining` (≈31,155 / 3,894 / 3,895 rows). Work on that shared clone — the
T15 output is git-ignored, so it isn't on a laptop unless you copy it.

### Step 1 — Get the code + activate the env
```bash
cd /project/igem2026/cb-promoter-mining
git pull                          # gets tokenise.py. If it says "local changes/divergent": git stash, or git pull --no-rebase --no-edit
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
#   ^ if that exact path doesn't exist: run `module load anaconda3` (or `conda info --base` to find <base>, then source <base>/etc/profile.d/conda.sh)
conda activate /project/igem2026/envs/promoter-mining-hpc4   # has transformers, torch, pyarrow
```
(This brief is git-ignored, so `git pull` does NOT fetch it — Oak hands it to you directly. `git pull`
is only for the `tokenise.py` script, which IS committed in the repo.)

### Step 2 — The two checkpoint IDs (LOCKED — do not change, do not ask anyone)
These are final. They are already the defaults inside `tokenise.py`; you do **not** pass any flag.
- **ProkBERT → `neuralbioinfo/prokbert-mini`** — plain mini (NOT `-mini-long`, NOT `…-promoter`). T18
  fine-tunes this exact base. Its tokenizer is a **custom `LCATokenizer`**, so the script loads it with
  `trust_remote_code=True` (already coded). 6-mer, overlapping (shift 1) → an 81 bp window becomes
  **78 tokens** (`[CLS]` + 76 + `[SEP]`), vocab 4101.
- **NT → `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`** — its tokenizer is the built-in
  `EsmTokenizer` (6-mer, non-overlapping) → an 81 bp window becomes **~18 tokens**, vocab 4107.

Why this is settled, not a question: the two streams are *meant* to be different lengths (78 vs ~18) —
each model gets its own tokenizer, and the script fits each pad width independently. Both tokenizers
require **uppercase** input; the script uppercases for you.

### Step 3 — Pre-stage the tokenizers on the LOGIN node (compute nodes are offline)
```bash
export HF_HOME=/project/igem2026/hf_cache
huggingface-cli download neuralbioinfo/prokbert-mini
huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
```
(If `huggingface-cli` is missing: `python -c "from huggingface_hub import snapshot_download as d; d('neuralbioinfo/prokbert-mini'); d('InstaDeepAI/nucleotide-transformer-v2-250m-multi-species')"`.)

### Step 4 — Skim the script so you own it
`dataset-construction/T16/tokenise.py`. It: loads each T15 split → uppercases `sequence` → tokenises
three ways (ProkBERT + NT via each model's `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`,
fitting one pad width per model over train+val+test; CNN via one-hot) → carries all metadata columns →
writes per-model outputs → asserts row counts are preserved. Don't hand-roll k-mer logic — using each
model's own tokenizer is the whole point.

### Step 5 — Run it
```bash
cd /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold
export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1
snakemake --cores 4 results/T16.done
# (equivalent direct run, from this same dir:)
# python ../../../dataset-construction/T16/tokenise.py results/T15/ results/T16/
```
Expect per-split lines like `train: 31155 rows -> prokbert, nt, cnn  (prokbert_len=78 nt_len=18)`.
(Note: the `Snakefile` comment near the T16 rule may still read "SCRIPT NOT YET WRITTEN" — that's
stale; the script exists and the rule runs it.)

### Step 6 — Verify the output (this is the QC — run it exactly)
Run from the **same dir and env** as Step 5 (so the paths and the cached tokenizer resolve):
```bash
cd /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold
export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1
python - <<'PY'
import pandas as pd, numpy as np
T15, T16 = "results/T15", "results/T16"
for s in ["train","val","test"]:
    src  = pd.read_parquet(f"{T15}/{s}.parquet"); n = len(src)
    pb   = pd.read_parquet(f"{T16}/prokbert/{s}.parquet")
    nt   = pd.read_parquet(f"{T16}/nt/{s}.parquet")
    cz   = np.load(f"{T16}/cnn/{s}.npz", allow_pickle=False)   # B2-fixed: loads without allow_pickle
    meta = pd.read_parquet(f"{T16}/cnn/{s}_meta.parquet")
    # 1. row COUNT + ORDER preserved (promoter_id sequence identical to T15)
    assert len(pb)==len(nt)==cz["X"].shape[0]==len(meta)==n, f"{s}: row count changed"
    assert list(pb.promoter_id)==list(src.promoter_id),      f"{s}: row ORDER changed"
    # 2. fixed token width per model, identical across rows (a DataLoader needs this)
    wpb = pb.input_ids.map(len).unique(); wnt = nt.input_ids.map(len).unique()
    assert len(wpb)==1 and len(wnt)==1, f"{s}: ragged token width"
    # 3. CNN one-hot: shape (n,81,4), at most one hot per position (ACGT=1, N/pad=0)
    X = cz["X"]; assert X.shape==(n,81,4) and set(np.unique(X))<= {0.0,1.0} and X.sum(2).max()<=1.0
    # 4. metadata carried: weights are only {0.3,1.0}; sigma_factor survived (T25 needs it)
    assert set(pb.weight.unique()) <= {0.3,1.0}, f"{s}: unexpected weights {set(pb.weight.unique())}"
    assert "sigma_factor" in pb.columns and "sigma_factor" in meta.columns, f"{s}: sigma_factor dropped"
    # 5. npz carries the arrays T20 reads
    assert {"X","label","weight","cluster_id"} <= set(cz.files), f"{s}: npz missing arrays"
    print(f"{s}: OK  n={n}  prokbert_width={wpb[0]}  nt_width={wnt[0]}  weights={sorted(set(pb.weight))}")
# 6. tokens actually encode the DNA. decode() does NOT round-trip for k-mer tokenizers, so check
#    structurally: the first real ProkBERT token must equal the sequence's first 6-mer.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
row = pd.read_parquet(f"{T16}/prokbert/test.parquet").iloc[0]
seq = pd.read_parquet(f"{T15}/test.parquet").iloc[0]["sequence"].upper()
first = tok.convert_ids_to_tokens(int(row.input_ids[1]))     # input_ids[0] is [CLS]
assert first == seq[:6], f"token/DNA mismatch: {first} != {seq[:6]}"
print("token<->DNA OK:", first, "==", seq[:6], "| ALL CHECKS PASSED")
PY
```
Pass = the block prints `ALL CHECKS PASSED` with no `AssertionError`. (Do **not** test by decoding
tokens back to a string and comparing to the DNA — k-mer tokenizers decode to space-joined k-mers, so
that would falsely "fail" a correct run. The structural check in step 6 above is the right test.)

### Step 7 — Commit + push the SCRIPT (not the output)
```bash
cd /project/igem2026/cb-promoter-mining
git add dataset-construction/T16/tokenise.py
git commit -m "T16 per-model tokenisation (ProkBERT-mini/NT/CNN), reads T15 splits"
git push                          # if "divergent branches": git pull --no-rebase --no-edit, then push
```
**Do NOT** `git add results/T16/` — it's generated output, git-ignored, and lives on this shared clone
for the training tasks to read directly.

### Step 8 — Where the outputs are (for T18/T19/T20 — FYI, nothing to ask)
The outputs are on the shared clone at `results/T16/`; the training scripts read them directly:
- `results/T16/prokbert/{train,val,test}.parquet` and `results/T16/nt/{…}.parquet` → columns
  `input_ids`, `attention_mask`, **plus every T15 metadata column** (`label`, `weight`, `cluster_id`,
  `split`, `in_genus`, `strand`, `source`, `species`, `sigma_factor`, `promoter_id`).
- `results/T16/cnn/{train,val,test}.npz` → `X` (N,81,4 one-hot), `label`, `weight`, `cluster_id`; the
  full metadata is row-aligned in `cnn/{split}_meta.parquet`.

---

## Locked decisions (all final — no one to consult)
- **Checkpoint IDs** — `neuralbioinfo/prokbert-mini` + `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`. Fixed (Step 2).
- **Output format** — parquet for the two LM models, `.npz` (+ a row-aligned `_meta.parquet`) for the
  CNN. Fixed. `--models prokbert,nt,cnn` lets you build a subset if you ever need to.
- **N handling for the CNN** — an `N` base becomes an all-zeros row. Fixed (rare anyway: T9, the only
  short/padded source, is held out of T15).
- **Pad width** — fitted **per model over train+val+test together**, then reused for all three splits
  (so widths are consistent and nothing is truncated). Fixed.
- **Metadata** — every T15 column except `sequence` is carried through unchanged.

## Gotchas (already handled in the script — just know them)
- ProkBERT needs `trust_remote_code=True`; NT's tokenizer doesn't (it's harmless there). Both need
  **uppercase** input — the script uppercases.
- The two token streams are different lengths (ProkBERT 78, NT ~18 for 81 bp) — expected, not a bug.
- Pre-cache the tokenizers + set `HF_HUB_OFFLINE=1` (compute nodes have no internet).
- The `weight` column only changes training if T18–T20 actually pass it as a sample/loss weight. T16's
  job is just to *carry* it (and it does); whether it's *used* is the training scripts' concern — this
  is recorded here so it isn't lost, not something for you to chase.

## Why it matters (one line)
T16 is the adapter between the clean T15 dataset and all three models — exact tokenizers + every
metadata column carried through, or the fine-tunes (T18/T19), the silver-weighting, the cluster-CV,
and the T25 sigma-stratified eval all silently break downstream.
