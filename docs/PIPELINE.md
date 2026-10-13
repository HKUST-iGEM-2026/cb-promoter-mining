# Reproduce the pipeline

Start with [setup](SETUP.md) and source `env.sh`. Run data stages from `workflow/`; root-stage commands below use PROMOTER_REPO. Source scripts live under `pipeline/`, not the obsolete numbered task-source directories. Names such as `T15_xs` and `T21_xs` below are existing data/run-output directories, not instructions to use the former source layout. Commands identify inputs and dependencies, not a one-command recreation of every historical run. Large recovered inputs are local, not committed.

## Step 0 — check the wiring

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"
snakemake -n -s local_prep.smk               # plans Phase A
snakemake -n results/T14/negative_set.tsv    # plans Phase B up to the negatives
```

Use explicit targets. The default legacy DAG reaches obsolete training wiring and optional C. difficile inputs.

## Phase A — acquire and curate

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"
snakemake -s local_prep.smk --cores 4
```

This stages curated promoters, synthetic evaluation sequences and published TSS. T11–T14 use the configured Bakta archive/directory separately. Inspect sizes and parsed row counts: existing zero-byte inputs are placeholders; skipped genome downloads can produce partial output.

Training positives come from RegulonDB/DBTBS, Hocq in-genus TSS, Wiegand/Xu cross-species starts and inferred housekeeping upstreams. RegulonDB includes weak/inferred evidence. The synthetic-promoter set has 93 cleaned 49–52-bp sequences and is held out; study strength units are not pooled. [Published-TSS sources](../pipeline/acquisition/tss-literature/README.md) and [synthetic-source curation](../pipeline/acquisition/synthetic-promoters/README.md) give the source papers and roles.

### Genome assemblies and coordinate provenance

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters. Putative-negative windows are not experimentally established promoter-free DNA.

The historical cohort contains 166 versioned assemblies, with no C. butyricum target. Preserve [the inventory](../pipeline/acquisition/clostridium-genomes/genome_list.tsv) and LFS archive. Both benchmark strains supplied silver positives and putative-negative training windows; published benchmark starts were withheld.

**Known error:** deployed B. licheniformis extraction used GCF_000011645.1/NC_006270.3, but the source coordinates fit GCF_000008425.1/NC_006322.1. The 888 windows are training-only and remain in all deployed members. The recipe below records the historical choice. Correct extraction followed by regenerated splits/retraining is a new run, not repaired old weights.

Published windows use transcription-oriented −60:+20 DNA with TSS offset 60. Genomic extraction is 0-based half-open; downstream candidate regions are 1-based inclusive. Confirm accession/contig/strand conventions before joining tables.

### The published-TSS ingest

[build_published_tss.sh](../pipeline/acquisition/tss-literature/build_published_tss.sh) selects Hocq's three 25RPM sheets and intergenic_sense starts. Historical pre-split positives were 5,759, versus 5,506 retained xs windows. Al-Bassam's 1,245 primary C. ljungdahlii starts are benchmark truth, never a file in the training ingest directory.

### The 47 dRNA-seq runs are not interchangeable

The 20 C. difficile libraries enter a prepared-only TEX/TAP TSSpredator path, not deployed labels. The other 27 Hocq/Al-Bassam runs use published tables. See [TSS sources](../pipeline/acquisition/tss-literature/README.md).

## Phase B — build the dataset

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"

# housekeeping loci -> inferred housekeeping clusters -> upstream windows -> putative negatives
# (name the target: the default target runs the whole DAG, including old training templates)
snakemake --cores 8 results/T14/negative_set.tsv
# The Slurm profile does not submit T11–T14: these are localrules in the current Snakefile.
# Run Snakemake on a site-approved CPU host/allocation.

# the two cross-species TSS sets that make the split "xs" (needs internet for the genomes)
BLICH=GCF_000011645.1   # as deployed; misplaces the TSS (see Genome assemblies). Correct: GCF_000008425.1
TSS=../pipeline/acquisition/additional-published-tss
EXTRACT=../pipeline/acquisition/tss-literature/extract_published_tss.py
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

# cluster-disjoint splits (measured sequence-leakage control, not a guarantee of independence)
python ../pipeline/datasets/clustering-and-splits/cdhit_splits.py \
  --positives inputs/reference_promoters.parquet inputs/published_tss_promoters.parquet \
              inputs/literature_promoters_train.parquet \
              inputs/blicheniformis_promoters.parquet inputs/rcellulolyticum_promoters.parquet \
              results/T13/upstream_regions.tsv \
  --negatives results/T14/negative_set.tsv \
  --identity 0.6 --silver-weight 0.3 --seed 42 --out results/T15_xs/

# per-model encodings
python ../pipeline/datasets/tokenisation/tokenise.py results/T15_xs/ results/T16_xs/
```

T12 groups only supplied housekeeping loci, not all CDSs or confirmed operons. Silver examples are annotation/position-derived putative promoter-positive gene-proximal crops with no mapped TSS or measured promoter activity; in deployed T15_xs, weight 0.3 expresses sample confidence for training loss, not biological probability or promoter prevalence. Curated non-81-bp inputs are center-cropped/N-padded.

Four negative classes are coding, putative intergenic, downstream/3′ and SRS shuffles of cropped silver upstreams. None is experimentally guaranteed promoter-free. The custom Eulerian shuffle's plain-shuffle fallback needs per-sequence QC for exact dinucleotide preservation.

The deployed split used **MMseqs2**, despite the historical `cdhit_splits.py` filename. The recovered `T15_xs/_cluster_mmseqs/all.fasta` and `mmseqs_cluster.tsv` account for all 50,190 DNA rows, and their compact cluster IDs exactly match every stored split row: 42,350 clusters, none shared across partitions. The retained recipe uses minimum identity 0.6, coverage 0.8 and coverage mode 1; the original invocation log/version is not recovered, so the membership table alone does not prove those flags. The splitter selects CD-HIT-EST only for its higher-identity option.

The splitter deduplicates DNA and pins clusters containing silver or cross-species positives to training. Overall labels are balanced, but partitions are not: train/validation/test contain 44,085/3,052/3,053 windows and 24,001/547/547 positive labels. Silver positives carry confidence weight 0.3; training negatives are rescaled to 0.7127265484963153, giving equal total positive and negative loss weight. Every validation/test weight is 1. The test's positive fraction, 547/3,053 = 0.179168, is the **random-ranking/no-skill AP baseline**, not any model's measured AP.

### Recover the original clustering command and version

Backend and membership recovery is complete; exact invocation/version recovery is not. Eight local MMseqs cache entries are symbolic links to the original `/project/igem2026/cb-promoter-mining/.../_cluster_mmseqs/` paths, with unavailable local targets. They are not proof that the source-host files are empty or deleted. The two retained cache directories are `tmp/8435348419779229844` and `tmp/9316599144162474585`; their `clu_rep.source` links point to each directory's `input.source`. Original generated shell scripts, job logs, Snakemake metadata and contemporaneous environment records may still exist on the source host.

The [read-only collector](../pipeline/datasets/clustering-and-splits/collect_mmseqs_provenance.py) prints candidate command/banner contexts, cache link targets, conda package/history records and executable hashes. It uses Python's standard library, does not run clustering or execute discovered binaries, and never infers a historical version from the current installation. Shell-history searches return matching commands only, without surrounding unrelated history.

From a Mac terminal with working authorised HKUST campus/VPN SSH access, run from the repository root:

```bash
REPORT="$(mktemp "$HOME/Downloads/mmseqs-provenance.XXXXXX")"
ssh -o StrictHostKeyChecking=yes osoo@superpod.ust.hk \
  'python3 - --root /project/igem2026/cb-promoter-mining \
    --environment-root /project/igem2026/envs \
    --environment-root "$HOME/.conda/envs" \
    --environment-root "$HOME/miniconda3/envs" \
    --history-file "$HOME/.bash_history" \
    --history-file "$HOME/.zsh_history"' \
  < pipeline/datasets/clustering-and-splits/collect_mmseqs_provenance.py \
  > "$REPORT" &&
printf 'Evidence saved to %s\n' "$REPORT"
```

The source mapping identifies `osoo@superpod.ust.hk`; `ust-hpc4` is the configured alternative alias for `osoo@hpc4.ust.hk`. If the original files remain on HPC4 rather than SuperPod, substitute `ust-hpc4` for the SSH destination in the same command. Both hosts timed out from the audit machine; no source-host collector output is claimed.

To inspect only the retained local material:

```bash
python3 pipeline/datasets/clustering-and-splits/collect_mmseqs_provenance.py \
  --root "$HOME/cb-promoter-original-run"
```

A recovered command must be tied to the deployed `T15_xs` output and a contemporaneous version source, such as the same job's MMseqs banner, its environment transaction/package record or an identified container/executable. `mmseqs version` in an unrelated current environment and the current Python recipe are not evidence of the original run. Preserve the collected JSONL so that these associations can be audited before updating the historical claim.

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

# window length and clean bases (N is accepted by the splitter but is not clean A/C/G/T)
print("all 81 bp:", bool((alld.sequence.str.len()==81).all()))
print("all A/C/G/T:", bool(alld.sequence.str.fullmatch("[ACGT]+").all()))
```

Zero shared cluster IDs is necessary, not proof of no related-sequence or genome exposure. Tokenization carries source metadata/weights and must preserve row order across encodings. [Verified recovery](MODELS.md#verified-original-data-and-encodings) checks the original data, not a newly generated split.

## Phase C — train

These are the recorded deployed settings; rerunning training does not guarantee bit-identical weights.

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"

python ../pipeline/training/prokbert/train_prokbert.py \
  --data-dir results/T16_xs/prokbert --output-dir results/T21_xs/prokbert \
  --learning-rate 4.675283456079137e-05 --batch-size 16 --num-epochs 10 \
  --warmup-ratio 0.11090738634337446 --weight-decay 0.07894770663194649 \
  --dropout-rate 0.1025172750748932 --seed 42

python ../pipeline/training/cnn/train_cnn_v2.py \
  --data-dir results/T16_xs/cnn --output-dir results/T21_xs/cnn_v2 \
  --encodings onehot,dinuc,trinuc,enac --learning-rate 0.0003748224111698842 \
  --batch-size 32 --dropout-rate 0.10382857322260322 --weight-decay 0.00024777455052193346 \
  --num-epochs 30 --patience 8 --seed 42 --enac-window 5
```

### The Evo2 probe

```bash
cd "$PROMOTER_REPO"

# 1) frozen Evo2-7B embeddings of T15_xs -> evo2_embeddings/  (GPU; or run it with bash on a GPU machine)
sbatch pipeline/training/Evo2/extract_embeddings.sbatch

# 2) once that job has finished, fit the probe  (CPU)
conda activate "$ENV_EVO2"
mkdir -p workflow/results/regimeC/evo2
python pipeline/training/Evo2/evo2_promoter_classifier.py --emb-dir evo2_embeddings \
  --out workflow/results/regimeC/evo2/evo2_probe_xs.pkl
```

The backbone stays frozen. The probe selects weighted validation AP, then refits on train+validation; test stays separate. Current extraction mean-pools blocks.28.mlp.l3 to 4096 features, but exact historical revision/layer remains unresolved. To use the trusted deployed probe instead of fitting:

```bash
mkdir -p workflow/results/regimeC/evo2
cp pipeline/training/Evo2/evo2_promoter_classifier.pkl \
  workflow/results/regimeC/evo2/evo2_probe_xs.pkl
```

Saved-feature test evaluation requires recovered features, not a GPU:

```bash
uv run --no-project --python 3.12 \
  --with numpy==2.5.4 --with scikit-learn==1.9.0 --with joblib==1.5.3 \
  python - "$PROMOTER_REPO/pipeline/training/Evo2/evo2_promoter_classifier.pkl" \
  "$HOME/cb-promoter-original-run/evo2_embeddings" <<'PY'
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

probe = joblib.load(sys.argv[1])
emb = Path(sys.argv[2])
X = np.load(emb / "test_X.npy", mmap_mode="r", allow_pickle=False)
y = np.load(emb / "test_y.npy", allow_pickle=False)
w = np.load(emb / "test_w.npy", allow_pickle=False)
p = probe.predict_proba(X)[:, 1]
if not np.isfinite(p).all():
    raise SystemExit("FAIL: probe produced nonfinite scores")
print(f"TEST rows={len(y)}, positives={int(y.sum())}")
print(f"Weighted TEST AUPRC: {average_precision_score(y, p, sample_weight=w):.8f}")
print(f"Weighted TEST AUROC: {roc_auc_score(y, p, sample_weight=w):.8f}")
PY
```

The recorded result is weighted AP 0.70258010/AUROC 0.89534660. This does not audit fresh DNA-to-embedding inference.

### Checkpoints and sweep

T21_xs is the deployed ProkBERT/CNN-v2 run; T21_final records earlier plain-0.6 comparisons. Keep checkpoint/split identity with every metric. Retune after changing the split:

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"
python ../pipeline/training/hyperparameter-sweep/hyperparameter_sweep.py \
  --models prokbert,cnn_v2 --n-trials 15 --sweep-epochs 4 \
  --out results/T21
```

## Phase D — evaluate

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"
# sampled held-out classifier metrics (historically called Regime A), one model at a time
python ../pipeline/evaluation/metrics/predict_and_score.py --model prokbert \
  --model-path results/T21_xs/prokbert/best_model --data-dir results/T16_xs/prokbert --out results/T22_xs
python ../pipeline/evaluation/metrics/predict_and_score.py --model cnn_v2 \
  --model-path results/T21_xs/cnn_v2/best_model.pt --data-dir results/T16_xs/cnn --out results/T22_xs

# cross-species external holdout
python ../pipeline/evaluation/external-holdout/evaluate_external_holdout.py \
  --models prokbert,cnn_v2 --prokbert-model-path results/T21_xs/prokbert/best_model \
  --cnn-v2-model-path results/T21_xs/cnn_v2/best_model.pt --out results/T24_xs
```

Loss weighting and AP weighting differ: ProkBERT/CNN-v2 trainers use unweighted AP, whereas the Evo2 probe supplies sample weights. [Results](RESULTS.md) distinguishes these records.

### Genome-scale benchmark (Regime C)

```bash
cd "$PROMOTER_REPO"
conda activate "$ENV_MAIN"
RC=workflow/results/regimeC && mkdir -p $RC
# C. ljungdahlii: genome and Bakta annotation from the committed zips (its truth table is committed)
unzip -p pipeline/acquisition/clostridium-genomes/Clostridium_genomes_v2_166.zip \
  'ncbi_dataset/data/GCF_000143685.1/*_genomic.fna' > $RC/clj.fna
unzip -p pipeline/acquisition/bakta-annotations/all_bakta_output.zip \
  'all_bakta_output/GCF_000143685.1_ASM14368v1_genomic_output/*_genomic.gff3' > $RC/clj.gff3
python pipeline/evaluation/make_candidate_regions.py --gff $RC/clj.gff3 --contig NC_014328.1 \
  --up 300 --down 50 --out $RC/clj_candidate_regions.bed
# C. autoethanogenum: downloads the genome, builds its truth table and regions (needs internet)
bash pipeline/evaluation/prep_cau_regimeC.sh

# GPU jobs (sbatch, or bash on a GPU machine)
MODELS="prokbert cnn_v2" sbatch pipeline/evaluation/regimeC_step10_allmodels.sbatch  # add nt cnn if trained
sbatch pipeline/training/Evo2/C_evo2_ljung_scan.sbatch
sbatch pipeline/training/Evo2/C_evo2_cau_scan.sbatch
sbatch pipeline/mining/genome-scan/mine_3model.sbatch      # ensemble + members, C. ljungdahlii
TAG=cau FASTA=$PWD/$RC/cau.fna REGIONS=$PWD/$RC/cau_candidate_regions.bed TRUTH=$PWD/$RC/cau_truth.tsv \
  sbatch pipeline/mining/genome-scan/mine_3model.sbatch    # and C. autoethanogenum
```

Wait for each prerequisite job to finish. This is a restricted, same-strand window-coverage benchmark: windows expand ±40 bp; unmatched starts merge at ≤81 bp. Precision proxy is covered TSS/(covered TSS+merged unmatched peaks), not the fraction of called peaks matching. Common aligned coordinates are required before member ranking.

### Calibration (optional; the mine does not use it)

```bash
cd "$PROMOTER_REPO/pipeline/evaluation/calibration"
conda activate "$ENV_MAIN"
# the genomic negatives: the committed GFF3s embed their genome sequences after ##FASTA
for g in C_acetobutylicum C_phytofermentans; do sed -n '/^##FASTA/,$p' $g.gff3 | tail -n +2 > $g.fna; done
python extract_negatives.py --output-dir negative_generation   # historical: 1,134 convergent + 126,811 CDS windows
cd "$PROMOTER_REPO"
sbatch pipeline/evaluation/calibration/run_calibration_xs.sh   # writes calib_xs/<model>/
```

The historical 279-row mixture contains 93 T9 positives and 186 putative negatives. Short sequence context and sampled prevalence limit transfer. Matching original calib_xs maps/cohort remain unavailable; older maps are not substitutes. [Calibration notes](../pipeline/evaluation/calibration/CALIBRATION_NOTES.md) explain score-only/AT models.

## Phase E — mine the target

```bash
cd "$PROMOTER_REPO"
conda activate "$ENV_MAIN"

# 1) stage the complete genome + candidate regions  (needs internet)
bash pipeline/mining/genome-scan/prep_g131795.sh
OUT=workflow/results/regimeC/mine3_g131795

# 2) 3-model mine  (GPU). TRUTH="" — empty, not unset — disables benchmarking
#    sbatch on a Slurm cluster; on a single GPU machine, run the same script with bash
TAG=g131795 TRUTH="" MODELS="prokbert cnn_v2 evo2" STEP=10 \
  FASTA=$PWD/$OUT/g131795.fna REGIONS=$PWD/$OUT/g131795_candidate_regions.bed \
  sbatch pipeline/mining/genome-scan/mine_3model.sbatch

# Wait for the mine job to finish successfully before running steps 3–5; sbatch only submits it.

# 3) call peaks  (CPU, pure pandas)
python pipeline/mining/peak-calling/call_peaks.py \
  --track $OUT/g131795_shortlist_3m_restricted.csv \
  --min-score 0.9 --merge 81 --out $OUT/g131795_peaks.csv

# 4) pull summit sequences, then annotate -10 / -35 with FIMO
python pipeline/mining/motif-annotation/extract_call_sequences.py \
  --calls $OUT/g131795_peaks.csv --genome $OUT/g131795.fna \
  --min-windows 5 --top-n 500 \
  --out-fasta $OUT/g131795_call_seqs.fasta \
  --out-top $OUT/g131795_call_seqs_top.fasta \
  --out-calls $OUT/g131795_calls_with_id.tsv

conda activate "$ENV_MEME"
bash pipeline/mining/motif-annotation/annotate_motifs.sh \
  $OUT/g131795_call_seqs_top.fasta $OUT/g131795_call_seqs.fasta $OUT/t31_g131795
conda activate "$ENV_MAIN"

# 5) build the candidate table
python pipeline/mining/candidate-table/build_candidate_table.py \
  --calls $OUT/g131795_calls_with_id.tsv \
  --fimo $OUT/t31_g131795/fimo/fimo.tsv \
  --out $OUT/g131795_candidates.tsv --jsonl $OUT/g131795_candidates.jsonl
```

The complete target is GCF_014131795.1. Set TRUTH to an empty string to disable benchmarking. ProkBERT/CNN-v2 and Evo2 tracks are intersected on contig/start/strand before average-tie percentile ranks are averaged. Do not pool independently ranked scanner outputs.

At least five supporting windows define the confident export. Recovered MEME/FIMO 5.3.0 files identify five de-novo motifs learned from the top 500 summit sequences (widths 14/7/8/18/10), then scanned across all 929 windows. The wrapper uses `--norc` without an explicit `--thresh`; the retained original FIMO settings record emitted p≤0.0001 and NRDB background A/C/G/T = 0.282/0.222/0.229/0.267. Builder p≤0.001 and upstream centre bins 3–18/22–42 assign −10/−35 labels without checking motif ID. The width-six `sigma_a_promoter.meme` is QC-only, not this scan's input. Starts are inferred from the 81-bp frame, not RNA mapping; `spacer_bp` is rounded centre separation.

### Prioritize the candidate table

```bash
MINE=$PWD/workflow/results/regimeC/mine3_g131795
python pipeline/mining/prioritize_candidates.py \
  $MINE/g131795_candidates.tsv --tag g131795 --outdir results/candidates/
```

Sort is motif tier, score, supporting windows, then best motif p-value. Current nested counts are 929/180/45. Sequence-enriched/spreadsheet hand-offs are committed, but their packaging generator is not bundled. Keep raw tracks, source reference and actual runtime with a new mine. Cached outputs are not proof of successful processing.

## Separate analyses

[Inducible-neighborhood code](../pipeline/mining/inducible/) selects oxidative-stress-associated loci; proximity does not prove inducibility. Its 24 FDR-screening peaks are not the ordinary catalogue. [FDR report](../pipeline/evaluation/empirical-fdr/results/cbut_xs/analysis.md) uses candidate+null pooled ranks, not ordinary target ranks; selected minimum window p-values do not establish calibrated peak FDR.

[Long-context positives](../pipeline/datasets/positives-with-context/) are an NTv3 experiment, not the deployed 81-bp dataset. Flanks require overlap/provenance checks. [Candidate interpretation](RESULTS.md#interpreting-candidates-and-downstream-units) distinguishes downstream clusters from transcripts. [DH5α strength](ECOLI_STRENGTH_PREDICTIONS.md) is an independent CPU analysis, not a member score.
