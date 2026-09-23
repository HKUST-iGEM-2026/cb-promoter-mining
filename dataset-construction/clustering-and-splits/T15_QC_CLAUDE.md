# CLAUDE.md — T15 QC & sign-off (for Sine)

> **Cluster setup (login, env, SLURM):** see `infrastructure/HPC4_GUIDE.md` — do that first.

You are a Claude instance helping **Sine** quality-check and sign off **Task T15** of the HKUST
iGEM 2026 dry-lab "promoter mining" pipeline. T15 is **already written and has run successfully** —
your job is **not to build it**, it's to help Sine *verify the output is sound, decide the two
tunable knobs, and sign it off* so the next task (T16) can use it. Read this top to bottom.

---

## 1. 30-second project context (why any of this matters)

The team is computationally mining the genome of **_Clostridium butyricum_** for **promoters** (short
DNA stretches that switch genes on). A promoter classifier is trained on **positive** windows (real
promoters) vs **negative** windows (non-promoters).

The core hazard — call it **the AT-trap**: _Clostridium_ DNA is **very AT-rich (~70% AT)**, and the
promoter motif (the −10/−35 boxes) is *also* AT-rich. So a lazy model can score high just by
detecting "AT-rich = promoter" **without ever learning the motif** — and then it flags every AT-rich
region as a promoter when mining the real genome (a flood of false positives). **Every QC check below
exists to make sure the dataset can't be solved by composition or by leakage — only by the motif.**

**T15's role:** take the labelled positive + negative 81 bp windows from 5 upstream sources, remove
near-duplicate **leakage** by clustering with CD-HIT, and write **cluster-disjoint 80/10/10
train/val/test splits**. Everything downstream (T16 tokenization → model training → evaluation →
genome mining) depends on this split being clean. T15 is the keystone.

## 2. What's already done (don't redo these)

- The script `dataset-construction/clustering-and-splits/cdhit_splits.py` is written, smoke-tested, and **ran clean on
  HPC4** → `results/T15/{train,val,test}.parquet`.
- The **T13 proximal-81 crop is already biologically verified** (own analysis + AI-council
  cross-check): the −10 signal peaks at ~−24 to −40 bp *inside* the 81 bp crop, ~65–70% captured, the
  far-upstream half is depleted; the ~⅓ that sits further upstream is why T13 is silver-weighted. **Do
  NOT re-verify the crop — that question is settled.**
- The reference run produced: **80/10/10** (train 31,155 / val 3,894 / test 3,895), **~50/50** pos:neg
  (19,472 each), **35,785 clusters**, **leakage PASS**.

## 3. Where the data is

Do this on the **shared HPC4 clone** `/project/igem2026/cb-promoter-mining` — that's where the inputs
and the T15 output already live (they're git-ignored, so a `git pull` on a laptop does NOT bring them;
you'd have to scp them or re-run the upstream pipeline). Activate the env first:
```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
cd /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold
```
The splits are at `results/T15/{train,val,test}.parquet`. Schema (11 cols):
`sequence, label, cluster_id, split, in_genus, strand, source, species, sigma_factor, weight, promoter_id`.

---

## 4. The QC checks — run each, read what it means, apply the pass/fail

Run this once to load everything, then the checks below reference `alld`:
```python
import pandas as pd
d = {s: pd.read_parquet(f"results/T15/{s}.parquet") for s in ["train","val","test"]}
alld = pd.concat(d.values(), ignore_index=True)
```

### Check 1 — split sizes & schema
```python
for s,x in d.items(): print(s, x.shape)
```
- **Means:** the 80/10/10 ratio and the 11-column schema.
- **PASS:** `train (31155, 11)`, `val (3894, 11)`, `test (3895, 11)` (≈80/10/10, 11 cols).
- **FAIL:** wildly off ratios, or missing columns.

### Check 2 — leakage (THE critical one)
```python
spanning = alld.groupby("cluster_id")["split"].nunique()
print("clusters spanning >1 split:", int((spanning > 1).sum()))
```
- **Means:** whether any CD-HIT cluster (a group of ≥90%-identical sequences) is split across
  train/val/test. If one is, a near-duplicate of a test sequence is in train → the model "memorizes"
  it → **CV scores are inflated and dishonest.**
- **PASS:** `0`. **FAIL:** anything `> 0` (stop — the split is leaking; don't sign off).

### Check 3 — class balance
```python
print(alld.groupby(["split","label"]).size().unstack())
```
- **Means:** positives (label=1) vs negatives (label=0) per split.
- **PASS:** each split ≈ 50/50 (so AUPRC's no-skill baseline is ~0.5 and accuracy isn't gamed by
  prevalence). **FAIL:** strongly skewed splits.

### Check 4 — GC/AT balance (the AT-trap guard — the most important interpretive check)
```python
alld["gc"] = alld.sequence.str.count("[GC]") / alld.sequence.str.len()
print(alld.groupby("label").gc.agg(["mean","std"]))        # overall pos vs neg
print(alld.groupby(["in_genus","label"]).gc.mean())        # << the key view
print(alld.groupby("source").gc.mean().sort_values())      # which source is the outlier
```
- **Why:** if positives are systematically a different GC% from negatives, the model can separate them
  on **composition alone** and skip the motif — the AT-trap. The SRS negative class (dinucleotide
  shuffles of the positives) exists precisely to pin this balance.
- **How to read it — the key line is `(in_genus, label)`:**
  - `(True, 1)` ≈ `(True/any, 0)` GC, both ~0.30 → **in-genus positives and the negatives are
    composition-matched → the model MUST learn the motif. GOOD.** This is the line that matters,
    because the real mining target is all in-genus AT-rich DNA.
  - `(False, 1)` will read **higher** GC (~0.43–0.50) → that's the cross-species T8 (_E. coli_/
    _B. subtilis_) positives. **Expected and fine** — it's handled downstream (T26 AT-content
    calibration; and the target genome is all AT-rich, so the "high-GC = positive" shortcut can't
    transfer).
- **PASS:** in-genus pos ≈ neg GC; the only GC-elevated group is T8 (`T8_RegulonDB`/`T8_DBTBS`).
- **RED FLAG:** a **broad** pos-vs-neg GC gap that ISN'T explained by T8 → the negatives aren't
  composition-matched → investigate before signing off.

### Check 5 — source mix & the silver weight
```python
print(alld.groupby(["source","label"]).size())
print(alld.groupby("source").weight.first())
```
- **Means / PASS:** positives = `T8_RegulonDB`(~3827) + `T8_DBTBS`(~373) cross-species verified,
  `Hocq2022`(~1434) in-genus verified, `T13_inferred`(~13838) in-genus **silver**; negatives =
  `SRS`/`T14_CDS`/`T14_Downstream_3Prime`/`T14_Intergenic_NonPromoter`. No `Yang2017` (T9 must be held
  out). `T13_inferred` must carry `weight = 0.3`; everything else `1.0`.
- **Interpretation:** T13 is ~71% of positives *by count* but only ~42% *by effective weight*
  (13,838 × 0.3 ≈ 4,150 vs 5,634 verified). That down-weighting is the AT-trap guard for the inferred
  data — confirm it's present and you're comfortable with the ratio.

### Check 6 — window length & clean bases
```python
print("all 81 bp:", bool((alld.sequence.str.len()==81).all()))
print("only ACGTN:", bool(alld.sequence.str.fullmatch("[ACGTN]+").all()))
print("sequences containing N:", int(alld.sequence.str.contains("N").sum()))
```
- **PASS:** all 81 bp, all ACGT(N), and **~0 sequences contain `N`** (N-padding only happens for the
  short T9 promoters, which are held out — so seeing many N's means T9 leaked into training).

---

## 5. The two decisions to sign off (both are tunable hyperparameters — yours to own)

Defaults produced the clean run above; accept them or sweep:
- **`--silver-weight 0.3`** (T13 down-weight). 0 = drop T13; 1.0 = T13 dominates (71% of positives).
  0.3 balances effective weight to ≈58% verified / 42% silver. **To set rigorously: sweep
  {0, 0.1, 0.3, 0.5, 1.0} and pick by validation AUPRC.**
- **`--cdhit-identity 0.9`** (clustering threshold). The anti-leakage knob. **To validate: sweep
  0.85–0.95 and confirm CV scores are stable** (they shouldn't swing if leakage is controlled).
- (`--neg-per-pos 1.0` gives the ~50/50 balance; `--window 81` is fixed to the verified-positive
  framing — don't change it.)

To re-run with different knobs (from the scaffold dir):
```bash
python ../../../dataset-construction/clustering-and-splits/cdhit_splits.py \
  --positives inputs/reference_promoters.parquet inputs/published_tss_promoters.parquet \
              inputs/literature_promoters_train.parquet results/T13/upstream_regions.tsv \
  --negatives results/T14/negative_set.tsv \
  --identity 0.9 --silver-weight 0.3 --seed 42 --out results/T15/
```

> ⚠️ **The `weight` column only matters if the training step (T18–T20) actually passes it as a
> sample/loss weight.** If training ignores it, T13 trains at full weight and the silver-weight guard
> is inert — flag this to whoever wires training.

## 6. When you've signed off

1. Tell **Leo** the output schema (above) — T16 (tokenization) is his and reads it directly.
2. Note any knob you changed from the defaults in `CLAUDE.md` §7 (the conventions section).

## 7. Sine's other open task (not T15)

**PT1** — the GPU env was rebuilt and verified this session, so re-run the ProkBERT fine-tune loop in
a GPU job and confirm `torch.cuda.is_available()` is `True` (it reportedly ran ~6 h on CPU before).
That's the one PT1 item still open. T10 and T11 are done.

---

**In one line:** verify the split is leakage-free (Check 2), composition-balanced for the in-genus
data (Check 4), correctly sourced/weighted (Check 5), and clean (Checks 1/3/6) — then decide the
silver-weight and identity knobs and sign off, because a dataset that's solvable by GC or by leakage
would make the whole downstream model chase AT-content instead of real promoters.
