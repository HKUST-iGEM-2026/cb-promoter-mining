# CLAUDE.md — T7b data refresh → full retrain cascade (for Yany)

> **Cluster setup (login, env, SLURM):** see `infrastructure/HPC4_GUIDE.md`. Per-step run guides: T15 `T15_QC_CLAUDE.md`, T16 `T16_BUILD_CLAUDE.md`, T18/19 `SANAA_T18_T19_RUN_CLAUDE.md`, T20 `model-training/T20/YANY_T20_RUN_CLAUDE.md`, T21 `model-training/T21/T21_SWEEP_RUN_CLAUDE.md`.

You're driving an **end-to-end data refresh**: add verified in-genus *Clostridium* promoters (T7b),
then re-run everything that depends on the dataset. **Adding data to T7b invalidates every split, token,
and model** — so this is a cascade, not a local edit. Do the phases in order; each depends on the
previous. Detailed per-task instructions live in the referenced sub-briefs.

```
T7b (more verified in-genus)
   └─> T15 re-split ──> T16 re-tokenise ──> T18/T19/T20 retrain ──> T21 hyperparameter sweep
```

**Why we're doing it:** val/test currently hold only ~268 verified in-genus promoters (mostly the model
is graded on *E. coli*). T7b is the only verified in-genus source feeding val/test. Growing it ~4× lifts
the in-genus eval slice to ~1,000+ **and** makes val/test bigger (more verified → bigger eligible pool →
bigger 10% val/test). Expected: verified in-genus ~1,434 → ~5,700+.

---

## Phase 0 — setup (HPC4, once)
```bash
ssh <you>@superpod.ust.hk            # HKUST VPN if off-campus
cd /project/igem2026/cb-promoter-mining
git pull                              # updated build_published_tss.sh, cdhit_splits.py, tokenise.py
git lfs pull                          # MATERIALIZE the Hocq xlsx + T5 genomes zip (they're LFS pointers otherwise)
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
```

## Phase 1 — DOWNLOAD / stage the data
Two in-genus species to add (the third, *C. acetobutylicum*, is already in T7b):
1. **C. beijerinckii — no download needed.** Its TSS are already in the committed Hocq workbook
   (`data-acquisition-and-curation/T7/published_tss/hocq2022_cac_tss.xlsx`, sheets `DSM_6423_25RPM`,
   `NCIMB_8052_25RPM`). `git lfs pull` (Phase 0) materializes it. Genomes `GCF_000016965.1` (NCIMB 8052)
   and `GCF_900010805.1` (DSM 6423) are in the **T5 zip** (or `datasets download genome accession <acc>`).
2. **C. ljungdahlii (Al-Bassam 2018) — one download.** Get the **TSS** supplementary table from the paper
   (NB: "Supplementary Data 2" is the gene/TE table — get the TSS list, not that). Save as
   `data-acquisition-and-curation/T7/published_tss/albassam2018_clj_tss.{tsv,csv,xlsx}` (commit via
   Git-LFS). Genome `GCF_000143685.1` is in the T5 zip. *(Download from the journal yourself — note its
   column names for the next phase.)*

## Phase 2 — T7b: extend + run  →  `inputs/published_tss_promoters.parquet`
Follow **`data-acquisition-and-curation/T7/T7B_EXTEND_CLAUDE.md`** (full code). In short: refactor the
Hocq stanza in `build_published_tss.sh` into a per-strain loop over the 3 strains (`intergenic_sense`
class, 25 RPM sheets — confirm each strain's `Locus`→accession contig-map first), and set the
`CLJ_POS_COL`/`CLJ_STRAND_COL` for Al-Bassam. Then:
```bash
bash data-acquisition-and-curation/T7/build_published_tss.sh
```
**Verify:** parquet has ~5,700+ rows, `species` shows all 3 (acetobutylicum + beijerinckii ×2 +
ljungdahlii), `in_genus` all True, sequences 81 bp clean ACGT, `promoter_id` unique. (Validation snippet
is in the sub-brief.)

## Phase 3 — T15: re-split  →  `results/T15/{train,val,test}.parquet`
The new positives flow into the split automatically (Snakefile rule already lists `published`).
```bash
cd infrastructure/T3/snakemake_scaffold
snakemake --cores 4 results/T15.done          # re-runs CD-HIT + verified-only split + 50/50 weighting
```
**QC (do it):** follow `dataset-construction/T15/T15_QC_CLAUDE.md`. Confirm: leakage check PASS;
val/test **verified-only** (0 silver); **TRAIN effective 50/50** (the summary prints pos-mass==neg-mass);
and — the whole point — the **`verified_in_genus` count per split is up ~4×** vs before.

## Phase 4 — T16: re-tokenise  →  `results/T16/{prokbert,nt,cnn}/...`
```bash
# pre-stage tokenizers on the LOGIN node first (compute nodes offline):
export HF_HOME=/project/igem2026/hf_cache
huggingface-cli download neuralbioinfo/prokbert-mini
huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
# then:
export HF_HUB_OFFLINE=1
snakemake --cores 4 results/T16.done
```
**QC:** follow `dataset-construction/T16/T16_BUILD_CLAUDE.md` (row counts preserved, `weight` carried into
the npz/parquet — this is what the next phase needs).

> **THE WEIGHT CONTRACT (Phases 5–6 — read this).** T16 carries the T15 `weight` per row (= silver 0.3 ×
> the 50/50 class-balance factor) into every model's input. **All three trainers MUST apply it as a
> PER-SAMPLE LOSS WEIGHT** — weighted cross-entropy `(<per-example CE> × weight).sum() / weight.sum()` —
> *not* as a `WeightedRandomSampler`, *not* ignored. T18/T19 already do this via a `WeightedTrainer`
> subclass (it pops `weight` before the forward, keeps `labels` so eval still works); **T20 must match**
> (`model-training/T20/YANY_T20_RUN_CLAUDE.md`). If a trainer ignores `weight`, the silver guard is inert and that model
> trains negative-skewed + silver-dominated (the AT-trap). All three also pick the best checkpoint on
> **validation AUPRC** (CLAUDE.md §7), never F1/accuracy.

## Phase 5 — T20: fix + retrain YOUR CNN on the new tokens
Apply the weight fix + run T20 per **`model-training/T20/YANY_T20_RUN_CLAUDE.md`** (consume `npz["weight"]` as a
per-sample loss weight; drop the inverse-class-frequency sampler; load + transpose the one-hot `X` from the T16 npz),
then retrain on `results/T16/cnn/`. **Verify** the CNN sees ~50/50 effective balance (the weight does it)
and report metrics on the (new, bigger, more in-genus) test split.

## Phase 6 — the other retrains + hyperparameter sweep (coordinate with Sanaa)
The LM models + the sweep were computed on the OLD dataset; the **fresh templates now exist** (written
2026-06) and just need to run on the new T16:
- **T18 ProkBERT (`model-training/T18/train_prokbert.py`) + T19 NT-LoRA (`model-training/T19/train_nt_lora.py`)
  — Sanaa runs them** on the new `results/T16/{prokbert,nt}/`. Both already implement the weight contract
  above and select on **validation AUPRC**. T18 fine-tunes plain `neuralbioinfo/prokbert-mini` with a fresh
  2-class head; T19 wraps NT v2 250M in LoRA (`target_modules=["query","value"]`, classifier head trainable),
  training ~1% of params on one GPU. (Pre-stage both checkpoints on the login node, `HF_HUB_OFFLINE=1`.)
- **T21 hyperparameter sweep (`model-training/T21/hyperparameter_sweep.py`; Sanaa+Oak; dep T18/T19/T20)** —
  rewritten: one Optuna study per model that launches each training script as a subprocess per trial and
  **maximises validation AUPRC** (the old template optimised F1 and had stale paths — both fixed; paths now
  point at the real `model-training/T18|T19|T20/…` scripts; selection metric is AUPRC, never on test = no
  leak). It is the **expensive** step (~25 trials × 3 models of full GPU training; tune `--n-trials`/`--models`),
  so either re-sweep or reuse the prior best config and re-validate. **Yany action for T21 to drive the CNN:**
  `train_cnn.py` (T20) must expose the *same* CLI contract as T18/T19 — accept `--sweep-trial`, `--output-dir`
  and the swept `--<hparam>` flags, and write `metrics.json` with `val_auprc` (+ print `Validation AUPRC: <x>`).
  T21's `cnn` block lists exactly the flags it will pass.

## Commit / push (from wherever you work)
- **Commit:** the Al-Bassam TSS table (Git-LFS) + any `build_published_tss.sh` edits + the T20 fix.
- **Do NOT commit** `results/` (T7b parquet output, T15/T16 outputs, model checkpoints) — all git-ignored
  and regenerated on the shared clone.

## Gotchas / dependencies
- **Strict order.** T15 needs Phase 2's parquet; T16 needs Phase 3; T20 needs Phase 4. Don't skip QC —
  a bad split silently corrupts everything after it.
- **Accession parity** is the #1 T7b failure (the extractor's zero-row assert) — match `Locus` labels to
  the exact FASTA record ids incl. the `.1`/`.2` suffix.
- **Target stays excluded.** This adds 3 *other* in-genus species; *C. butyricum* is never in training.
- **Scale check.** This grows the dataset; T15 (CD-HIT) and T16 (tokenise) take longer but are CPU-only.
  Training (Phase 5/6) is the GPU cost.

## One-line summary
Download Al-Bassam + use the in-workbook *C. beijerinckii* → re-run T7b → T15 → T16 → retrain (T20 now,
T18/T19 next) → re-sweep hyperparameters (T21). The data change invalidates the splits, tokens, and
models, so the cascade is: **T7b → T15 → T16 → retrain (T18/T19/T20) → T21 sweep.**
