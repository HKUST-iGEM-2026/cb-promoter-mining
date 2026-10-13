# Promoter mining in Clostridium butyricum

**HKUST iGEM 2026 — Computational promoter mining.** We rank promoter-like DNA in *C. butyricum* DSM 10702 with three classifiers, then annotate candidate peaks with motifs. The complete-reference catalogue, **GCF_014131795.1**, contains **929 regions**, including **180 with placed −10** and **45 with both placed motif classes**.

Catalogue tiers help us organise predictions; they do not mean that every promoter has been experimentally validated. The ensemble rank is neither expression strength nor a probability. To explore reporter activity separately, we reproduced a [DH5α strength model](docs/ECOLI_STRENGTH_PREDICTIONS.md) for the 50-bp cores of the first ten promoters with both −10 and −35 motifs. The guide includes every prediction, the reference and the start-site assumptions.

The team reports Cb characterisation, iHP-informed decisions and *E. coli* GFP reporter assays. Time constraints meant validation stopped at *E. coli*; native Cb promoter activity is not established. [Results](docs/RESULTS.md) distinguishes this reported work from computational predictions.

## Repository layout

```text
pipeline/
  acquisition/   genomes, promoter databases and published TSS
  datasets/      labels, background controls, splits and encodings
  training/      ProkBERT, fusion CNN and frozen-Evo2 probe
  evaluation/    metrics, benchmarks, calibration and FDR
  mining/        scoring, rank ensemble, peaks, motifs and strength
workflow/        Snakemake files, profiles, inputs and local results
environments/    conda recipes
release/         Hugging Face cards and uploader
results/candidates/  delivered sequence catalogues
analysis/phylogeny/  historical tree analysis
archive/prototypes/ early experiments
docs/            technical guides
```

Source archives use Git LFS. Checkpoints and large recovered run inputs are ignored by git; a fresh clone does not include them. [Setup](docs/SETUP.md) explains environments and staging.

## Score DNA

The deployed trio is fine-tuned ProkBERT-mini, from-scratch fusion CNN-v2 and a logistic probe on frozen Evo2-7B embeddings. The authenticated CloBiT inventory lists private component packages, not a public promoter-prediction service or a packaged three-model ensemble. Use authorised local checkpoints or obtain access before following the conditional Hub examples in [model loading](docs/MODELS.md).

From the repository root, with compatible packages and local checkpoints:

```bash
source env.sh
conda activate "$ENV_MAIN"
python pipeline/mining/score_promoters.py --model cnn_v2 --fasta seqs.fa \
  --models-dir "$PROMOTER_RESULTS/T21_xs" --out cnn_scores.csv
```

Input is transcription-oriented 81-bp DNA. The legacy CSV column `prob` is a raw positive-class score. This CLI runs one neural classifier, not Evo2 or the ensemble. Default results are under `workflow/results/`.

### Predict DH5α core activity

The fitted SVR is included in [this source tree](release/huggingface/dh5alpha-promoter-strength-svr). It is separate from the three genome-mining classifiers and can run with NumPy alone; the command below does not require Hugging Face access:

```bash
uv run --no-project --with numpy==2.5.3 python \
  pipeline/mining/strength-prediction/predict_dh5alpha_strength.py \
  --model release/huggingface/dh5alpha-promoter-strength-svr \
  --input results/candidates/g131795_tier1_canonical_seq.tsv \
  --top-n 10 --tss-offset 60 --out-prefix dh5alpha_first10
```

The [strength guide](docs/ECOLI_STRENGTH_PREDICTIONS.md#loading-the-fitted-model) gives local loading and fitting commands, plus conditional private-package downloads. The output compares predicted activity with a same-model predicted reference, not measured fluorescence or native Cb expression.

## Read the results

- [929 ranked sequences](results/candidates/g131795_ranked_seq.tsv) and [45 promoters with both −10 and −35 motifs](results/candidates/g131795_tier1_canonical_seq.tsv).
- [Results](docs/RESULTS.md): benchmark definitions, motif evidence and downstream-unit interpretation.
- [Pipeline](docs/PIPELINE.md): stage commands; the legacy default Snakemake target is not an end-to-end recipe.

Here, **silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters. Negative windows are putative negatives, not experimentally established promoter-free DNA.

At recall 0.5, historical paired TSS-matching proxies are 61.6% for *C. ljungdahlii* and 76.6% for *C. autoethanogenum*. The benchmark genomes supplied silver positives and putative-negative training windows; only their published TSS truth was withheld. These are restricted scans, not measured target promoter precision. The deployed set also retains 888 wrong-assembly *B. licheniformis* positive windows.

Project code/documentation is [MIT](LICENSE). ProkBERT, CNN/probe, external code and source data have separate terms; see [release licences](release/huggingface/README.md#licences) and [RegulonDB notice](REGULONDB_LICENSE.md).

## Publish source updates

The local wiki is maintained separately and is not included in the GitHub source repository. Do not move it into another tracked directory or link repository documentation to its local files. `.gitignore` excludes it; already-tracked wiki files must also be removed from Git's index without deleting the local copies.

The source-layout branch is `reorganise`; the GitHub default `main` branch retains the older layout. Historical names such as `T21_xs` identify retained run-output directories, not current source directories. Review the current working tree and index before publishing; do not use a blanket “take theirs” resolution to restore obsolete source directories or undo scientific corrections.

For ordinary source updates when no merge is in progress:

```bash
git switch reorganise
git rm -r --cached --ignore-unmatch -- wiki
git add -A -- . ':(top,exclude)wiki/**' ':(top,exclude)wiki' ':(glob,exclude)**/.DS_Store'
git diff --cached --stat
git diff --cached --name-status
# Review before committing: wiki entries must be deletions, never additions/modifications.
git commit -m "Correct scientific claims, update documentation, keep wiki local"
git fetch origin
git merge --no-edit origin/reorganise
# If Git reports conflicts, resolve and commit them before the following command.
git push origin HEAD:reorganise
```

If a merge has already stopped with conflicts, resolve each file first and stage only its intended resolution. Preserve remote additions in their current `pipeline/` or `release/huggingface/` locations; delete obsolete `huggingface_release/` paths rather than restoring duplicate implementations. Keep the wiki files locally, but remove any conflicted wiki entry from Git's index with `git rm -r --cached --ignore-unmatch -- wiki`.

After the index has no unresolved entries, finish the existing merge rather than starting another one:

```bash
git diff --cached --stat
git diff --cached --name-status
git commit --no-edit
git push origin HEAD:reorganise
```

This uses a normal merge and push, not a force-push; it preserves remote changes. Hugging Face publication is a separate operation described in [release instructions](release/huggingface/README.md#publish).
