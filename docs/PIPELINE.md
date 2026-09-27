# Reproducing the pipeline

Five phases: acquire data, build the dataset, train, evaluate, mine. Installing the environments and
running with or without Slurm are in [SETUP.md](SETUP.md); the models themselves are in
[MODELS.md](MODELS.md); findings and numbers are in [RESULTS.md](RESULTS.md).

Every command below assumes you have run `source env.sh` in the current shell. It sets
`$PROMOTER_REPO` and the conda env names (`$ENV_MAIN`, `$ENV_EVO2`, `$ENV_MEME`, `$ENV_NT`).

The target, *C. butyricum* DSM 10702, is **excluded from every training set**. Mining it is pure
inference.

> **Path convention.** The data steps write into `results/` *relative to the scaffold directory*. Run
> them from `infrastructure/T3/snakemake_scaffold/` and call scripts through `../../../`, which is
> why the commands below look the way they do.

---

## Step 0 — check the wiring

Dry-run the two workflow stages you will run. A dry run plans the jobs and executes nothing.

```bash
cd "$PROMOTER_REPO/infrastructure/T3/snakemake_scaffold"
conda activate "$ENV_MAIN"
snakemake -n -s local_prep.smk               # plans Phase A
snakemake -n results/T14/negative_set.tsv    # plans Phase B up to the negatives
```

A clean dry run validates **wiring only**. Do not read it as "the pipeline works". Don't dry-run the
default target (`snakemake -n`) on a fresh clone: it fails with `MissingInputException`, because it
needs Phase A's `inputs/` and `inputs/tss_sites.gff3`, which only the optional *C. difficile* track
(`tss_prep.smk`) produces. It also reaches old training-template rules that the commands below
replace. Cluster settings live in exactly one file, `profiles/slurm/config.yaml`.

---

## Phase A — acquire and curate

Turns the committed source data into the `inputs/` files that Phase B reads. `inputs/` is
git-ignored, so this is required on every fresh clone. It takes a few minutes and needs internet.

```bash
cd "$PROMOTER_REPO/infrastructure/T3/snakemake_scaffold"
conda activate "$ENV_MAIN"
snakemake -s local_prep.smk --cores 4
```

| Output in `inputs/` | Rows | Built from |
|---|---|---|
| `reference_promoters.parquet` | 4,336 | RegulonDB `PromoterS_Set.txt` (3,963) + `DBTBS_prom.fa` (373), both committed |
| `published_tss_promoters.parquet` | **5,759** | the committed Hocq 2022 workbook: *C. acetobutylicum* 1,434 + *C. beijerinckii* 2,482 + 1,843 |
| `literature_promoters.tsv` | 93 | the curated synthetic promoters; held out for evaluation |
| `literature_promoters_train.parquet` | 0 | the training share of those 93, empty because `t9_train_fraction` is 0 |
| `SRR_dRNAseq_for_TSSpredator.txt` | 20 | run accessions, used only by the optional *C. difficile* track |

Check the in-genus count. Its three genomes are not in the committed genome zip, so they are
downloaded with `ncbi-datasets-cli`. If a download fails, the script prints a message, skips that
strain and carries on, and Phase B then silently has fewer in-genus positives.

### Where the training positives come from

Counts are the rows each source contributes to the deployed split.

| Source | Rows | Species | Notes |
|---|---|---|---|
| Published in-genus TSS | 5,506 | *C. acetobutylicum*, *C. beijerinckii* | Hocq 2022, the verified in-genus signal; 5,759 before exact-sequence dedup |
| RegulonDB | 3,827 | *E. coli* | curated, cross-species; 3,963 before windowing |
| *B. licheniformis* | 888 | *Bacillus* | published TSS; misplaced in the deployed split, see below |
| *Ruminiclostridium cellulolyticum* | 663 | ex-*Clostridium* relative | published TSS |
| DBTBS | 373 | *B. subtilis* | curated |
| Housekeeping upstreams | 13,838 | in-genus, from the 166 genomes | **silver**, down-weighted 0.3; 14,659 operon leaders before windowing |

Source citations, including the two that exist only as NCBI accessions, are in the header of
`data-acquisition-and-curation/additional-published-tss/prep_new_tss.py`.

**Not in this table, and not training positives:** the *C. difficile* TSSpredator output (1,746 TSS;
out-of-genus, see *The TSS literature*), the 93 T9 synthetic promoters (`t9_train_fraction: 0`, held
out for evaluation), and the *C. ljungdahlii* and *C. autoethanogenum* published TSS (benchmark truth).
Only the six sources above reach `cdhit_splits.py --positives`.

These counts were read from the deployed split itself (`results/T15_xs/{train,val,test}.parquet`,
grouped by `source`), so they are what the models actually saw. They are smaller than the Phase A
outputs above because step 3 of `cdhit_splits.py` drops exact duplicate sequences, preferring
verified over silver over negative: the Hocq set enters with 5,759 rows and 5,506 survive.

Only Hocq 2022 reaches validation and test (4,412 / 547 / 547). Every other source is train-only, as
the pinning rule intends. The `source` values in the split are `Hocq2022`, `T13_inferred`,
`T8_RegulonDB`, `T8_DBTBS`, `Wiegand2013` and `Xu2015`.

Deployed split (`T15_xs`): **50,190 rows, 25,095 positives**, of which 19,344 are in-genus.
Cross-species sources are **pinned to train**, never val or test, so the deployment metric stays
honest. `xs` stands for *cross-species-augmented*: it is the base split plus the *B. licheniformis* and
*R. cellulolyticum* sets, which are what Phase B's extra step adds. Every deployed model trains on it.

The target, *C. butyricum* DSM 10702, is excluded from all training data; the 166-genome set was
downloaded with it removed. The two validation genomes are held out only in part. Their published
start sites were never training positives: *C. ljungdahlii*'s 1,245 primary TSS
(`evaluation-and-calibration/ljungdahlii_holdout_tss.tsv`) and *C. autoethanogenum*'s 2,885 are
used only as benchmark truth. But both strains are among the 166 genomes that supply the housekeeping
upstreams and the negative windows. *C. ljungdahlii* DSM 13528 is there as `GCF_000143685.1`, the
same assembly the benchmark scans. *C. autoethanogenum* DSM 10061 is there as `GCF_001484725.1`,
another assembly of the benchmarked strain. The genome-scale benchmark is therefore not fully
independent of training. All models share this, so their comparisons are like-for-like.

### Genome assemblies, verified against NCBI

TSS calling and lift-over need the exact assembly each paper used.

| Strain | RefSeq assembly | Records |
|---|---|---|
| *C. butyricum* DSM 10702 (**the target**) | `GCF_014131795.1` | `NZ_CP040626.1` chr1, `NZ_CP040627.1` chr2, `NZ_CP040628/9.1` plasmids |
| *C. acetobutylicum* ATCC 824 | `GCF_000008765.1` | `NC_003030.1` + pSOL1 `NC_001988.2` |
| *C. beijerinckii* NCIMB 8052 | `GCF_000016965.1` | `NC_009617.1` |
| *C. beijerinckii* DSM 6423 | `GCF_900010805.1` | chromosome + phage φ6423 + pNF1/pNF2 |
| *C. ljungdahlii* DSM 13528 = ATCC 55383 (**validation**; also in the 166) | `GCF_000143685.1` | `NC_014328.1` |
| *C. autoethanogenum* DSM 10061 (**validation**) | `GCF_000484505.1` | `NC_022592.1` |
| *C. difficile* 630 | `GCF_000009205.2` | `NC_009089.1` + pCD630 |
| *B. licheniformis* DSM 13 | `GCF_000008425.1` | `NC_006322.1` (the deployed split used the wrong one; see below) |
| *R. cellulolyticum* H10 | `GCF_000022065.1` | `NC_011898.1` |

**Known issue in the deployed models.** *B. licheniformis* DSM 13 has a second, independently
assembled RefSeq genome, `GCF_000011645.1` (`NC_006270.3`). The deployed `xs` split was built against
it, and the TSS table does **not** fit it, so its 888 *B. licheniformis* "positives" are effectively
random windows. The two assemblies differ by a **+147 to +196 bp** offset: every one of the 889 table
positions locates uniquely in both, 750 of them shifted by 147–149 bp and 139 by 194–196 bp.

The QC below is reproducible from the committed table and the two RefSeq assemblies. A window "carries
a −10 box" if some 6-mer starting at offset 44–50 is within one mismatch of `TATAAT`; the control
applies the same detector to windows taken 200 bp further upstream.

| Extracted from | −10 box | control, 200 bp further up | +1 purine |
|---|---|---|---|
| `NC_006270.3` (deployed) | 5% | 6% | 52% |
| `NC_006322.1` (correct) | **63%** | 5% | **95%** |

The 888 are 3.5% of the positives and all sit in train.
Validation and test hold only in-genus sequences, and the genome-scale benchmarks use the two
validation genomes, so no evaluation contains them. The deployed models, and so the candidate list, come from
this split, and Phase B uses the same assembly so it rebuilds it. With `BLICH=GCF_000008425.1` it
builds a split whose *B. licheniformis* windows are real promoters.

The *R. cellulolyticum* table fits H10, its only complete genome, though its TSS positions are
looser: by the same measure 34% carry a −10 box against 12% for the 200 bp-upstream control, and the
+1 purine rate is 58%, so there is no sharp +1 peak (real promoters read 95%).

An earlier mine used `GCF_000409755.1`, a 96-contig draft of the same target strain. It is
superseded. The two coordinate frames share **no contig ids**, so their outputs cannot be joined by
position.

### The published-TSS ingest

Phase A's `t7b_published_tss` rule runs `build_published_tss.sh`, which reads the committed Hocq 2022
workbook (`data-acquisition-and-curation/tss-literature/published_tss/hocq2022_cac_tss.xlsx`)
directly. For each strain's 25 RPM sheet it keeps the `intergenic_sense` class, downloads the matching
genome and cuts 81 bp windows. No manual download is needed.

The script ingests **any** table in `published_tss/`, including an Al-Bassam 2018 *C. ljungdahlii*
file if one is placed there. Don't: *C. ljungdahlii* is a validation genome, and its start sites are
the benchmark truth.

**Coordinate parity is the number-one silent failure.** The table's contig values must match the
FASTA headers *exactly*, including the version suffix: a table saying `NC_014328` against a FASTA
header `>NC_014328.1` drops every row. The ingester asserts a non-zero result and prints drop
reasons, so parity mismatches now fail loudly. Multi-replicon genomes require `--contig-col` plus a
contig map covering **every** label, or those TSS drop as unmapped.

Framing is fixed and hard-coded: **81 bp, TSS at 0-based offset 60**, window [−60, +20]. Coordinates
are treated as 1-based. Do not change it.

### The 47 dRNA-seq runs are not interchangeable

They span three assay types and only one kind works with TSSpredator, which needs a differential
5′-end pair (enriched versus untreated control).

| Method | Study | Species | Runs | Path |
|---|---|---|---|---|
| dRNA-seq (TEX±) | Fuchs 2021 | *Clostridioides difficile* 630 | 18 | TSSpredator |
| 5′-end RNA-seq (TAP±) | Soutourina 2020 | *C. difficile* 630Δerm | 2 | TSSpredator |
| Capp-Switch | Hocq 2022 | *C. acetobutylicum*, *C. beijerinckii* | 11 | published table |
| 5′ RNA-seq | Al-Bassam 2018 | *C. ljungdahlii* | 16 | published table |

20 runs to TSSpredator, 27 to published tables. For the Soutourina pair the enriched library is
**TAP+ = `SRR11573852`** and the control is `SRR11573851`; TAP+ is the same direction as TEX+, so
there is no inversion.

Its TSS are **not** training positives: the TSSpredator output is kept as out-of-genus supporting
evidence and never passed to `cdhit_splits.py --positives`.

**The taxonomy trap.** *C. difficile* was reclassified to ***Clostridioides*** in 2016. It is **not**
in genus *Clostridium* (taxon 1485) and not among the 166 training genomes. The only
TSSpredator-ready data is therefore out-of-genus, while the three species that *are* in-genus need
the published-table path. Map each dataset to its own genome, never to another.

TSSpredator itself needs care: an early run over-called by roughly 50×, producing 170,766 TSS where
the literature reports a few thousand. The causes, in order of impact, were no replicate
reproducibility filter, no expression floor, and two studies with different replicate counts mixed
into one run. The corrected configs split them (Fuchs 3×3 with `minNumRepMatches=2`, Soutourina 1×1)
and set `minNormalHeight` to 0.2, yielding 1,746 enriched primary TSS. Expect low thousands. If the
output is still enormous, raise the height thresholds and confirm the replicate matching.

---

## Phase B — build the dataset

This builds `T15_xs` and `T16_xs`, the split and encodings every deployed model trained on.

```bash
cd "$PROMOTER_REPO/infrastructure/T3/snakemake_scaffold"
conda activate "$ENV_MAIN"

# housekeeping genes -> operons -> upstream windows -> negatives
# (name the target: the default target runs the whole DAG, including old training templates)
snakemake --cores 8 results/T14/negative_set.tsv
# on a Slurm cluster: snakemake --workflow-profile profiles/slurm results/T14/negative_set.tsv

# the two cross-species TSS sets that make the split "xs" (needs internet for the genomes)
BLICH=GCF_000011645.1   # as deployed; misplaces the TSS (see Genome assemblies). Correct: GCF_000008425.1
TSS=../../../data-acquisition-and-curation/additional-published-tss
EXTRACT=../../../data-acquisition-and-curation/tss-literature/extract_published_tss.py
for acc in $BLICH GCF_000022065.1; do
  datasets download genome accession $acc --include genome --filename inputs/$acc.zip
  unzip -o -q inputs/$acc.zip -d inputs/$acc
done
python $EXTRACT --tss-table $TSS/blicheniformis_tss.tsv \
  --genome inputs/$BLICH/ncbi_dataset/data/$BLICH/*_genomic.fna \
  --species "Bacillus licheniformis" --source Wiegand2013 \
  --pos-col pos --strand-col strand --out-of-genus --out inputs/blicheniformis_promoters.parquet
python $EXTRACT --tss-table $TSS/ruminiclostridium_tss.tsv \
  --genome inputs/GCF_000022065.1/ncbi_dataset/data/GCF_000022065.1/*_genomic.fna \
  --species "Ruminiclostridium cellulolyticum" --source Xu2015 \
  --pos-col pos --strand-col strand --out-of-genus --out inputs/rcellulolyticum_promoters.parquet

# cluster-disjoint splits (the anti-leakage guard)
python ../../../dataset-construction/clustering-and-splits/cdhit_splits.py \
  --positives inputs/reference_promoters.parquet inputs/published_tss_promoters.parquet \
              inputs/literature_promoters_train.parquet \
              inputs/blicheniformis_promoters.parquet inputs/rcellulolyticum_promoters.parquet \
              results/T13/upstream_regions.tsv \
  --negatives results/T14/negative_set.tsv \
  --identity 0.6 --silver-weight 0.3 --seed 42 --out results/T15_xs/

# per-model encodings
python ../../../dataset-construction/tokenisation/tokenise.py results/T15_xs/ results/T16_xs/
```

The two extractions should report **888** and **663** windows — 888 on either *B. licheniformis*
assembly, since both drop the same single contig-end window from the table's 889 rows. The committed `*_tss.tsv` tables are `prep_new_tss.py`'s output from the journal workbooks in
the same folder; rerun it only if you change those workbooks. The deployed split came out at 50,190
rows: 44,085 train, 3,052 val and 3,053 test.
Its positives were 19,344 in-genus plus 5,751 cross-species (3,827 + 888 + 663 + 373).

Leave out the two cross-species files and you get the plain `T15`/`T16` split that the earlier
0.6-clustering models used.

### Housekeeping genes, operons, upstream windows

Housekeeping genes are selected **by product description, not gene symbol**, because the light Bakta
database leaves symbols sparse but products populated. Selection favours precision over recall:
these regions become positives, so regulated or ambiguous genes are excluded even at the cost of
fewer hits. Sigma factors, heat-shock chaperones and RecA are deliberately out, since their promoters
are regulated. Coding sequences only; rRNA and tRNA promoters are atypical.

That gives about 22,760 housekeeping genes across 166 genomes. Operon detection collapses them to
14,659 leaders, clustering co-directional runs then taking the most upstream gene **strand-aware**:
minimum start on `+`, maximum stop on `−`. Getting that backwards on the minus strand was a real bug.
Upstream extraction then takes the 300 bp immediately 5′ of each leader, dropping any that would run
off a contig. The split step later keeps the 81 bp nearest the gene.

### The four negative classes

Coding-sequence windows, intergenic non-promoter, downstream/3′, and **SRS**: dinucleotide-preserving
shuffles of real promoters, which keep base composition but destroy the −10/−35 signal. Realistic
negatives are the main defence against the AT-rich false-positive trap. SRS is training-only; the
empirical-FDR null is the convergent-intergenic set instead.

### Two details that matter more than they look

**Cluster-disjoint splits.** Sequences sharing 60% identity land in one cluster, and whole clusters
stay on one side of the split. Without this, near-duplicates leak across train and test and every
score inflates. CD-HIT cannot cluster DNA below about 0.8 because of its word-size floor, so anything
under that is routed automatically to MMseqs2. The threshold was tightened from 0.9 to 0.6 on a
reviewer's recommendation; balanced AUPRC barely moved, which showed the earlier scores were **not**
leakage-inflated, but the genome-scale ranking did shift.

**Silver weighting.** Housekeeping upstreams are *inferred* positives, reached through three stacked
assumptions, and a −10-crop check found about a third do not frame the promoter. They carry weight
0.3 against 1.0 for verified promoters. At full weight the 13,838 AT-rich inferred windows
would teach exactly the "AT-rich means promoter" shortcut the project exists to avoid. The `weight`
column also class-balances train to effectively 50/50.

The `weight` column only does anything if the training script applies it as a **per-sample loss
weight**. An early CNN template instead built a sampler from inverse class frequency, which threw the
silver down-weighting away and double-counted class balance. All three trainers now apply it
correctly.

### QC the split before trusting it

```python
import pandas as pd
d = {s: pd.read_parquet(f"results/T15_xs/{s}.parquet") for s in ["train","val","test"]}
alld = pd.concat(d.values(), ignore_index=True)

# leakage — the critical check. MUST be 0.
print("clusters spanning >1 split:", int((alld.groupby("cluster_id")["split"].nunique() > 1).sum()))

# composition — the AT-trap guard
alld["gc"] = alld.sequence.str.count("[GC]") / alld.sequence.str.len()
print(alld.groupby(["in_genus","label"]).gc.mean())

# window length and clean bases
print("all 81 bp:", bool((alld.sequence.str.len()==81).all()))
```

Reading the composition check: in-genus positives and the negatives should both be AT-rich, meaning
the model **must** learn the motif. The published in-genus promoters average 0.23 GC. Cross-species
promoters differ by source: RegulonDB 0.41, DBTBS 0.30, *R. cellulolyticum* 0.25. The deployed
*B. licheniformis* windows read 0.44, because they are random genomic DNA. A broad
positive-versus-negative GC gap that the sources do *not* explain means the negatives are not
composition-matched. Investigate before going further.

### Tokenisation

Each model needs a different numeric format, and using each model's own tokenizer is the whole point
— do not hand-roll k-mer logic. ProkBERT uses its custom `LCATokenizer`, 6-mer overlapping, so an
81 bp window becomes **78 tokens**. The CNN gets one-hot `(N, 81, 4)`. The ProkBERT and NT tokenizers
both need uppercase input, and ProkBERT needs `trust_remote_code=True`.

Pad width is fitted per model over train, val and test together, so nothing is silently truncated.
Every metadata column is carried through unchanged: dropping `weight`, `cluster_id` or `sigma_factor`
silently breaks the silver weighting, the cluster-aware cross-validation and the stratified
evaluation respectively.

One shape trap: the tokeniser writes the CNN array as `(N, 81, 4)` = (batch, length, channels), but
PyTorch `Conv1d` needs `(N, 4, 81)`. Transpose it, and build the model with 4 input channels and
length 81.

---

## Phase C — train

The mine loads three things, all trained on the `xs` split:

- `results/T21_xs/prokbert/best_model`
- `results/T21_xs/cnn_v2/best_model.pt`
- the Evo2 probe `results/regimeC/evo2/evo2_probe_xs.pkl`

The commands below rebuild them with the same settings. The hyperparameter values were read back
from the deployed checkpoints themselves: ProkBERT's `training_args.bin` and model config, and
CNN-v2's `config.json`. ProkBERT's are identical to `T21_final_06`'s, the winners of the sweep on the
0.6 split. CNN-v2's sit inside that sweep's search range. Reuse them; no re-sweep is needed.

```bash
cd "$PROMOTER_REPO/infrastructure/T3/snakemake_scaffold"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"

python ../../../model-training/prokbert/train_prokbert.py \
  --data-dir results/T16_xs/prokbert --output-dir results/T21_xs/prokbert \
  --learning-rate 4.675283456079137e-05 --batch-size 16 --num-epochs 10 \
  --warmup-ratio 0.11090738634337446 --weight-decay 0.07894770663194649 \
  --dropout-rate 0.1025172750748932 --seed 42

python ../../../model-training/cnn/train_cnn_v2.py \
  --data-dir results/T16_xs/cnn --output-dir results/T21_xs/cnn_v2 \
  --encodings onehot,dinuc,trinuc,enac --learning-rate 0.0003748224111698842 \
  --batch-size 32 --dropout-rate 0.10382857322260322 --weight-decay 0.00024777455052193346 \
  --num-epochs 30 --patience 8 --seed 42
```

The deployed CNN-v2 recorded validation AUPRC 0.738 and test AUPRC 0.726 in its `metrics.json`, so
expect to land close to that. ProkBERT's metrics were not kept with its checkpoint.

Keep every `--output-dir` under `results/`. The scoring, evaluation and mining scripts look for
checkpoints there, and each job script lists the exact paths it loads near its top. Trained models
are git-ignored.

Both trainers honour two contracts:

- **Weighted loss.** The per-sample `weight` is applied as a loss weight. This is the AT-trap guard.
- **Selection on validation AUPRC**, plus a sweep CLI (`--sweep-trial`, `--output-dir`, `--<hparam>`)
  so the hyperparameter sweep can drive them. They write `metrics.json` and print the AUPRC.

Pass criteria: `weighted == True` in `metrics.json`, and a validation AUPRC well clear of the
no-skill baseline, which equals the validation positive prevalence. For the deployed `xs` split that
is **≈0.18** (547 positives in 3,052 validation rows); for the earlier plain 0.6 split it was ≈0.29.
Sitting at the baseline means it did not learn.

### The Evo2 probe

Evo2 is trained differently: the backbone stays frozen and only a scikit-learn probe is fitted on
mean-pooled embeddings of `T15_xs`. Logistic regression and random forest are both tried and selected
on weighted validation AUPRC; logistic regression wins. Both steps run in `$ENV_EVO2`, from the repo
root.

```bash
cd "$PROMOTER_REPO"

# 1) frozen Evo2-7B embeddings of T15_xs -> evo2_embeddings/  (GPU; or run it with bash on a GPU machine)
sbatch model-training/Evo2/extract_embeddings.sbatch

# 2) once that job has finished, fit the probe  (CPU)
conda activate "$ENV_EVO2"
mkdir -p infrastructure/T3/snakemake_scaffold/results/regimeC/evo2
python model-training/Evo2/evo2_promoter_classifier.py --emb-dir evo2_embeddings \
  --out infrastructure/T3/snakemake_scaffold/results/regimeC/evo2/evo2_probe_xs.pkl
```

Check the printout: about 44k training rows, logistic regression picked, and a test AUPRC near 0.70.
The saved file is the winner refit on train plus validation, so it has seen 47,137 rows.

The deployed probe is also committed, as `model-training/Evo2/evo2_promoter_classifier.pkl`. To use
it instead of running steps 1 and 2, copy it to the path Phases D and E read:

```bash
mkdir -p infrastructure/T3/snakemake_scaffold/results/regimeC/evo2
cp model-training/Evo2/evo2_promoter_classifier.pkl \
  infrastructure/T3/snakemake_scaffold/results/regimeC/evo2/evo2_probe_xs.pkl
```

More on the probe in `model-training/Evo2/README.md`.

### Which checkpoint folder is which

Several training rounds left folders under `results/`. Only `T21_xs` is deployed.

| Folder | What it holds |
|---|---|
| `T18`, `T19`, `T20` | the first standalone runs of ProkBERT, NT-LoRA and CNN-v1, with each trainer's defaults |
| `T21`, `T21_06` | the hyperparameter sweeps on the 0.9 and 0.6 splits; `<model>/best_params.json` holds each winner |
| `T21_final`, `T21_final_06` | full retrains on the plain 0.6 split, with the 0.9-swept and 0.6-swept settings |
| **`T21_xs`** | **deployed**: retrained on the `xs` split. ProkBERT uses exactly `T21_final_06`'s settings. The mine, the `calib_xs` calibration, the FDR run and the Hugging Face ProkBERT and CNN all come from here |

`T21_xs` also holds NT-LoRA and CNN-v1 checkpoints, which were used only in the model comparisons. They
train the same way from `results/T16_xs/nt` and `results/T16_xs/cnn`, but their settings were not
recorded, so their numbers will not match exactly.

### Hyperparameter sweep

The sweep is an orchestrator, not a trainer. Per model it runs an Optuna study that launches that
model's training script as a subprocess once per trial, passes sampled hyperparameters as CLI flags,
and maximises validation AUPRC read back from each trial's `metrics.json`.

```bash
python ../../../model-training/hyperparameter-sweep/hyperparameter_sweep.py \
  --models prokbert,cnn_v2 --n-trials 15 --sweep-epochs 4 \
  --out results/T21
```

On compute nodes without internet, pre-stage the base models first and set `HF_HUB_OFFLINE=1`; see
[SETUP.md](SETUP.md).

The sweep trains on the plain split: each trainer's default `--data-dir` is `results/T16/<model>`, and
CNN-v2's is set in `SWEEP_CONFIG`. To sweep on the `xs` split, set `--data-dir results/T16_xs/<model>`
in each model's `extra_args` in `hyperparameter_sweep.py`.

Cost discipline matters: one full training per trial. Prove the wiring with three trials first.
`--sweep-epochs 4` is the biggest lever, forcing cheap trials; **then retrain the winner at full
epochs**, because the cheap proxy only ranks configurations. The study is a sqlite database per
model, so a preempted job resumes on resubmission. It selects on validation AUPRC only, never test.

One finding worth keeping: the sweep had originally been run on 0.9-clustered data, so the 0.6 models
inherited stale hyperparameters. Re-sweeping recovered about two thirds of a genome-scale drop in
ProkBERT (`T21_final_06`), measured at step 1 like the earlier checkpoints. The `xs` ProkBERT was
scanned the same way: step 1, restricted, raw scores, *C. ljungdahlii*. At recall 0.5 it made
604 false-positive peaks (50.8% precision) against 905 (40.8%) for `T21_final_06`, and at recall 0.1
70 against 177 (`calib_xs/` and `calib_06/`, `regimeC_calibrated_analysis_*.md`). The two checkpoints
used identical hyperparameters, so the difference is the data: the `xs` split adds both
cross-species sets and matching extra negatives, and re-draws the train/val/test split. Each is also
a single training run, so this comparison does not show which data change produced the gain. On the
step-10 grid, the `xs` ProkBERT scores 47.7% at recall 0.5 against CNN-v2's 47.2%.

---

## Phase D — evaluate

Run from the scaffold directory, on the checkpoints Phase C produced:

```bash
# balanced test metrics (Regime A), one model at a time
python ../../../evaluation-and-calibration/metrics/predict_and_score.py --model prokbert \
  --model-path results/T21_xs/prokbert/best_model --data-dir results/T16_xs/prokbert --out results/T22_xs
python ../../../evaluation-and-calibration/metrics/predict_and_score.py --model cnn_v2 \
  --model-path results/T21_xs/cnn_v2/best_model.pt --data-dir results/T16_xs/cnn --out results/T22_xs

# cross-species external holdout
python ../../../evaluation-and-calibration/external-holdout/evaluate_external_holdout.py \
  --models prokbert,cnn_v2 --prokbert-model-path results/T21_xs/prokbert/best_model \
  --cnn-v2-model-path results/T21_xs/cnn_v2/best_model.pt --out results/T24_xs
```

Judge models on **genome-scale precision**, not balanced AUPRC. The three regimes, the numbers, and
what they mean are in [RESULTS.md](RESULTS.md).

### Genome-scale benchmark (Regime C)

Every deployment decision rests on these numbers. Each job scans a validation genome's candidate
regions at step 10 and counts false-positive peaks at fixed recall against its published start sites.
Stage the two genomes once, from the repo root:

```bash
cd "$PROMOTER_REPO"
RC=infrastructure/T3/snakemake_scaffold/results/regimeC && mkdir -p $RC
# C. ljungdahlii: genome and Bakta annotation from the committed zips (its truth table is committed)
unzip -p data-acquisition-and-curation/clostridium-genomes/Clostridium_genomes_v2_166.zip \
  'ncbi_dataset/data/GCF_000143685.1/*_genomic.fna' > $RC/clj.fna
unzip -p data-acquisition-and-curation/bakta-annotations/all_bakta_output.zip \
  'all_bakta_output/GCF_000143685.1_ASM14368v1_genomic_output/*_genomic.gff3' > $RC/clj.gff3
python evaluation-and-calibration/make_candidate_regions.py --gff $RC/clj.gff3 --contig NC_014328.1 \
  --up 300 --down 50 --out $RC/clj_candidate_regions.bed
# C. autoethanogenum: downloads the genome, builds its truth table and regions (needs internet)
bash evaluation-and-calibration/prep_cau_regimeC.sh

# GPU jobs (sbatch, or bash on a GPU machine)
MODELS="prokbert cnn_v2" sbatch evaluation-and-calibration/regimeC_step10_allmodels.sbatch  # add nt cnn if trained
sbatch model-training/Evo2/C_evo2_ljung_scan.sbatch
sbatch model-training/Evo2/C_evo2_cau_scan.sbatch
sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch      # ensemble + members, C. ljungdahlii
TAG=cau FASTA=$PWD/$RC/cau.fna REGIONS=$PWD/$RC/cau_candidate_regions.bed TRUTH=$PWD/$RC/cau_truth.tsv \
  sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch    # and C. autoethanogenum
```

Each writes `genome_fp_benchmark.md` tables under `results/regimeC/`. Compare models only within one
run: the members' numbers in the `mine_3model` runs differ slightly from the standalone scans (Evo2
69.4% against 71.0% at recall 0.1 on *C. ljungdahlii*).

### Calibration (optional; the mine does not use it)

```bash
cd "$PROMOTER_REPO/evaluation-and-calibration/calibration"
# the genomic negatives: the committed GFF3s embed their genome sequences after ##FASTA
for g in C_acetobutylicum C_phytofermentans; do sed -n '/^##FASTA/,$p' $g.gff3 | tail -n +2 > $g.fna; done
python extract_negatives.py --output-dir negative_generation   # 1,134 convergent + 126,811 CDS windows
cd "$PROMOTER_REPO"
sbatch evaluation-and-calibration/calibration/run_calibration_xs.sh   # writes calib_xs/<model>/
```

`run_calibration_xs.sh` calibrates all four `xs` checkpoints (ProkBERT, CNN-v1, CNN-v2, NT-LoRA); remove
the lines for any you did not train. The calibrator files are not committed; the reports are.

Calibration fits a Platt regressor with an AT-content covariate per model, choosing unregularised
versus L2 by stratified K-fold cross-validation on out-of-fold log-loss and Brier score, then
refitting on the full set. It writes a `<model>_calibrator_at.joblib` alongside metrics and per-row
predictions. Calibration is only needed for a fixed threshold; the mine itself is calibration-free
because rank percentile is scale-invariant.

---

## Phase E — mine the target

This produced the delivered candidate list. Inference only; the target is never trained on. Run it
from the repo root.

The mining job loads exactly the files Phase C writes: `results/T21_xs/…` for ProkBERT and CNN-v2
(`MPATH` near the top of `mine_3model.sbatch`), and `PROBE=results/regimeC/evo2/evo2_probe_xs.pkl`.
If you trained somewhere else, edit `MPATH` or pass `PROBE=…`. The preflight stops with `MISSING …`
if a path is wrong.

```bash
cd "$PROMOTER_REPO"

# 1) stage the complete genome + candidate regions  (needs internet)
bash mining-and-motif-annotation/genome-scan/prep_g131795.sh
OUT=infrastructure/T3/snakemake_scaffold/results/regimeC/mine3_g131795

# 2) 3-model mine  (GPU). TRUTH="" — empty, not unset — disables benchmarking
#    sbatch on a Slurm cluster; on a single GPU machine, run the same script with bash
TAG=g131795 TRUTH="" MODELS="prokbert cnn_v2 evo2" STEP=10 \
  FASTA=$PWD/$OUT/g131795.fna REGIONS=$PWD/$OUT/g131795_candidate_regions.bed \
  sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch

# 3) call peaks  (CPU, pure pandas)
python mining-and-motif-annotation/peak-calling/call_peaks.py \
  --track $OUT/g131795_shortlist_3m_restricted.csv \
  --min-score 0.9 --merge 81 --out $OUT/g131795_peaks.csv

# 4) pull summit sequences, then annotate -10 / -35 with FIMO
python mining-and-motif-annotation/motif-annotation/extract_call_sequences.py \
  --calls $OUT/g131795_peaks.csv --genome $OUT/g131795.fna \
  --min-windows 5 --top-n 500 \
  --out-fasta $OUT/g131795_call_seqs.fasta \
  --out-top $OUT/g131795_call_seqs_top.fasta \
  --out-calls $OUT/g131795_calls_with_id.tsv

conda activate "$ENV_MEME"
bash mining-and-motif-annotation/motif-annotation/annotate_motifs.sh \
  $OUT/g131795_call_seqs_top.fasta $OUT/g131795_call_seqs.fasta $OUT/t31_g131795
conda activate "$ENV_MAIN"

# 5) build the candidate table
python mining-and-motif-annotation/candidate-table/build_candidate_table.py \
  --calls $OUT/g131795_calls_with_id.tsv \
  --fimo $OUT/t31_g131795/fimo/fimo.tsv \
  --out $OUT/g131795_candidates.tsv --jsonl $OUT/g131795_candidates.jsonl
```

The mining job scans each model, auto-switching into `$ENV_EVO2` for that member and back, then
reconciles the tracks onto a common grid and writes the ensemble and the shortlist. Run **one job per
tag** at a time, since per-model tracks are cached by tag. It is resume-safe under preemption. A
complete ~4.7 Mb genome takes roughly 17 minutes for the ProkBERT and CNN-v2 tracks; the Evo2
restricted scan adds about a GPU-hour, so budget ~1.3 GPU-hours for a cold 3-model run.

**"Restricted" means** two different things, so keep them apart. In the **per-model Regime C scans**,
ProkBERT and CNN-v2 score the whole genome and only windows inside candidate promoter regions count,
so the rank percentiles stay genome-wide while the false-positive denominator covers only where
promoters live. In the **3-model mine**, Evo2 scans only the candidate regions
(`evo2_scan_restricted.py --regions`), and `align_tracks.py` intersects every track onto that grid, so
the ensemble's mean rank percentile is computed over the **candidate space**, not the genome. The ≥0.9
cut that defines the 929 candidates is therefore a percentile within the candidate space.

Step 5 of `run_fdr_mine.sbatch` runs `annotate_motifs.py` in `$ENV_MEME`, which needs `pandas` and
`numpy` in that env (`environment-bio.yml` does not install them). The step is guarded, so without
them the job still succeeds but silently drops `top80_promoters_motif.csv`.

### Empirical FDR and the wet-lab ranking

The mine ranks and thresholds; the FDR step adds the statistical layer, an empirical p and q per
locus against a convergent-intergenic null, so a candidate is "significant at q ≤ 0.05" rather than
merely "scored ≥ 0.9". Run it on the **same assembly** as the mine or the coordinates will not join.

```bash
MINE=$PWD/infrastructure/T3/snakemake_scaffold/results/regimeC/mine3_g131795
FOUT=$PWD/evaluation-and-calibration/empirical-fdr/results/cbut_g131795_xs
mkdir -p "$FOUT"
cp "$MINE/g131795_candidate_regions.bed" "$FOUT/cbut_candidate_regions.bed"   # reuse exact windows

# the null pool: composition-matched convergent-intergenic windows, at 81 bp (the default is 52,
# which fdr_stats.py rejects). No null TSV is committed, so build one first.
NULL=$PWD/evaluation-and-calibration/calibration/c_butyricum_annotation
mkdir -p "$NULL" && cp "$MINE/g131795.fna" "$NULL/CBUT.fna" && cp "$MINE/g131795.gff3" "$NULL/CBUT.gff3"
python evaluation-and-calibration/calibration/extract_negatives.py \
  --input-dir "$NULL" --output-dir "$NULL" --fna CBUT.fna --gff3 CBUT.gff3 --window 81

GENOME=$MINE/g131795.fna GFF=$MINE/g131795.gff3 OUT=$FOUT MODELSET=xs STEP=10 \
  NULL_TSVS="$NULL/convergent_intergenic_negatives.tsv" \
  sbatch evaluation-and-calibration/empirical-fdr/run_fdr_mine.sbatch

python evaluation-and-calibration/empirical-fdr/join_fdr_qvalues.py \
  --candidates $MINE/g131795_candidates.tsv \
  --fdr-peaks  $FOUT/peaks_fdr.csv \
  --out        $MINE/g131795_candidates_fdr.tsv

python mining-and-motif-annotation/prioritize_candidates.py \
  $MINE/g131795_candidates_fdr.tsv --tag g131795 --outdir final-candidates/
```

GPU saver: the mine's Evo2 candidate track is already restricted on the same grid, so
`cp $MINE/g131795_evo2.csv $FOUT/cand_evo2.csv` skips roughly a GPU-hour. The grid reconciler fails
loudly on a mismatch, so it is safe to try.

A candidate with a **blank q** was not called above the p-gate. That is a "believe it less" flag, not
proof it is a non-promoter.

### Mining gotchas

- **Contig-id parity** is enforced in staging and again in the job preflight. If it complains, the
  FASTA headers and the BED/GFF seqids disagree. Re-stage; never hand-edit scores.
- **`STEP=10` for every model.** The tracks will not reconcile otherwise. At step 1 the plus and
  minus grids coincide, but at step 10 they diverge, and a mismatched minus strand silently drops
  half the windows. The symptom is a low maximum achievable recall in the ensemble benchmark.
- **`TRUTH=""` must be empty, not unset.** An earlier version used a shell default that treats an
  empty string as unset, so it benchmarked the target against another genome's TSS and aborted on a
  contig mismatch.
- **Do not commit `results/`.** It is git-ignored and regenerated by the pipeline.

---

## A separate track: inducible promoters

`inducible_promoters/` applies the same three-model ensemble to oxidative-stress genes in the target,
scoring the upstream region of each gene rather than the whole genome. It reuses the mine, peak
calling and motif annotation unchanged; the per-gene best call is the useful output. Note the motif
model there is still the constitutive sigma-A −10/−35, so it tells you where a promoter is, not
whether it is inducible. Detecting the inducible signal itself, such as a PerR operator, would need a
separate scan, which the repo does not include.

## A separate track: long-context positives

`positives_with_context/` builds a 384 bp-context version of the positive set for the NTv3
experiment, carrying both an 81 bp core and a long window per row. This is **additive and separate**
from the deployed 81 bp pipeline; do not repoint the main split at it. To train on context
end-to-end, the negatives must also be given context first, because the splitter enforces context on
all sources or none.
