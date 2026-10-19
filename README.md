# Promoter mining in *Clostridium butyricum*

**HKUST iGEM 2026.** We rank promoter-like DNA in *C. butyricum* DSM 10702 using fine-tuned ProkBERT-mini, a fusion CNN and a logistic probe on frozen Evo2-7B embeddings. Motif annotations then help prioritise sequences for testing. A separate DH5α-trained SVR estimates reporter-core activity; it is not part of the mining ensemble.

## Results and downloads

| Output | Result | Interpretation |
|---|---|---|
| Complete-reference mine, **GCF_014131795.1** | **929 regions; 180 with placed −10; 45 with both placed motif classes** | Nested computational shortlist, not 929 validated promoters |
| Paired genomic benchmark, precision proxy at recall 0.5 | **61.6%** (*C. ljungdahlii*); **76.6%** (*C. autoethanogenum*) | Restricted published-TSS recovery, not measured biological purity |
| Deployed held-out test AP | ProkBERT **0.82494**; CNN **0.72558**; Evo2 probe **0.70258** | Classification of putative/curated labels, not expression strength |
| First-ten DH5α core predictions | `cand_00098` leads at **32.8751 AU**, **474.57%** of the primary predicted `pptb_fU` core | A model/reference-framing comparison, not measured improvement |
| Oxidative-stress-associated search | **24 peaks**; no motif test passes adjusted q≤0.05 | Separate exploratory candidates; activity and stress protection unvalidated |

**[Download results, DNA and provenance](results/README.md)**: ranked catalogue, 45 both-hit sequences, named DL1–DL10 FASTA/TSV, reporter/reference DNA, gene context, strength predictions, benchmark CSV, oxidative-stress candidates and 166-assembly training manifest.

Read **[Results](docs/RESULTS.md)** for protocols and limitations and **[DH5α strength predictions](docs/ECOLI_STRENGTH_PREDICTIONS.md)** for all ten numerical estimates and framing sensitivity. **DL10 is `cand_00101` / `P_prfB`; `pptb_fU` is an independent reference**, despite its older notebook alias.

The team reports plasmid ordering, transformation and GFP testing in *E. coli*. Candidate-linked quantitative assay data and reporter strain were not supplied here; native Cb validation was not reached. Neither the separate codon optimiser nor oxidative-stress candidates were used in experimental constructs. These computational results do not establish improved expression, constitutivity or protection.

## Use the models

All four [CloBiT model packages](docs/MODELS.md#public-hugging-face-repositories) are **public and ungated**: project-package downloads need no account or token. Uploading requires write authorisation. They are component packages, not a hosted prediction service or a packaged three-model ensemble. Review remote code before execution, trust only verified pickle sources, and preserve upstream/data licences.

The fitted DH5α SVR is also included locally. From the repository root, score the real retained first-ten inputs without refitting:

```bash
uv run --no-project --with numpy==2.5.3 python \
  pipeline/mining/strength-prediction/predict_dh5alpha_strength.py \
  --model release/huggingface/dh5alpha-promoter-strength-svr \
  --reference results/candidates/pptb_fU_reference.json \
  --input results/candidates/g131795_tier1_canonical_seq.tsv \
  --top-n 10 --tss-offset 60 --out-prefix /tmp/dh5alpha_first10
```

For one neural classifier, first install [compatible dependencies](docs/SETUP.md) and stage its public checkpoint as described in [Models](docs/MODELS.md):

```bash
source env.sh
conda activate "$ENV_MAIN"
python pipeline/mining/score_promoters.py --model cnn_v2 \
  --fasta results/candidates/g131795_first10_canonical_named.fasta \
  --models-dir "$PROMOTER_RESULTS/T21_xs" --out /tmp/cnn_scores.csv
```

Input is transcription-oriented 81-bp DNA, with nominal TSS offset 60. The CLI's legacy `prob` column is an uncalibrated positive-class score; this command runs neither Evo2 nor the ensemble. The 81-bp frame is a dataset convention, not ProkBERT's context limit.

## Source and reproduction

```text
pipeline/
  acquisition/   genomes, promoter databases and published TSS
  datasets/      labels, controls, splits and encodings
  training/      ProkBERT, fusion CNN and frozen-Evo2 probe
  evaluation/    metrics, benchmarks, calibration and FDR
  mining/        scoring, peaks, motifs, strength and inducible/ source
workflow/        Snakemake files, profiles, inputs and local run outputs
environments/    conda recipes
release/         Hugging Face cards, fitted artifacts and uploader
results/         public result/download index and retained outputs
docs/            technical guides
analysis/phylogeny/ and archive/prototypes/  historical analyses
```

[Setup](docs/SETUP.md) explains dependencies and staging; [Pipeline](docs/PIPELINE.md) gives stage-specific commands. The legacy default Snakemake target is not an end-to-end reproduction recipe. Source archives use Git LFS; a clone does not contain ignored checkpoints or the large recovered splits/score tracks. Public summary artifacts do not replace those reconstruction prerequisites.

**Important limits:** silver positives are annotation/position-derived putative promoters and negatives are not proven promoter-free. The benchmark genomes supplied silver/negative training windows; only their published TSS truth was withheld. The supervised target genome was excluded, but backbone pretraining exposure is unknown. All deployed members retain **888 wrong-assembly *B. licheniformis* positive windows**; changing documentation does not repair those weights. Reference TSS catalogues are incomplete, and motif/rank support does not measure activity.

Project code/documentation is [MIT](LICENSE). Model artifacts, executable upstream code and source data have distinct terms: see [release licences](release/huggingface/README.md#licences) and the [RegulonDB notice](REGULONDB_LICENSE.md).

## Publish source updates

The earlier public-results GitHub/Hugging Face publication is complete. This **new factual SRS correction** is prepared on **`fix/srs-split-documentation`** in the separate `cb-promoter-srs-correction` checkout, based on published `main`. Publish this branch through a pull request to **`main`**; do not stage from the original dirty or prior publication checkouts. Scope staging to these correction paths:

```bash
git switch fix/srs-split-documentation
git add -- README.md docs/MODELS.md docs/RESULTS.md docs/PIPELINE.md \
  results/README.md results/training_negative_sources.csv \
  pipeline/datasets/clustering-and-splits/cdhit_splits.py \
  release/huggingface/README.md \
  release/huggingface/prokbert-mini-clostridium-promoter/README.md \
  release/huggingface/cnn-fusion-clostridium-promoter/README.md \
  release/huggingface/evo2-logreg-clostridium-promoter-probe/README.md
git diff --cached --stat
git diff --cached --name-status
git diff --cached
# Confirm only the intended public payload is staged before continuing.
git commit -m "Correct SRS negative split documentation and publish source counts"
git push -u origin fix/srs-split-documentation
gh pr create --base main --head fix/srs-split-documentation \
  --title "Correct SRS negative split documentation" \
  --body "Document SRS negatives across all splits with retained source counts; distinguish positive-label pins and CNN features without changing models or metrics."
```

These are manual instructions, not staging or publication actions performed for this correction. Merge the GitHub correction first so the new `main/results/training_negative_sources.csv` link resolves, then **reupload only README.md for the three classifier cards** using the [release instructions](release/huggingface/README.md#publish-these-documentation-changes-only). The independent DH5α card has no incorrect SRS claim and needs no upload. Historical verification/recovery JSON and receipts, weights, splits, metrics and licences remain unchanged. No retraining, full model release or visibility change is required. Keep local wiki/export directories out of the payload.
