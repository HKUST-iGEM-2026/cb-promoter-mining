---
license: cc-by-4.0
base_model: arcinstitute/evo2_7b
tags:
  - biology
  - genomics
  - dna
  - promoter-prediction
  - clostridium
  - evo2
  - probe
  - igem
---

# Evo2 frozen-embedding promoter probe

HKUST iGEM 2026 logistic probe on frozen [Evo2-7B](https://github.com/ArcInstitute/evo2), deployed with ProkBERT-mini and CNN-v2. The probe declares CC-BY-4.0; Evo2's separate base card declares Apache-2.0. **Backbone weights are not included.**

## Infer

Current extraction reads 81-bp oriented DNA, mean-pools blocks.28.mlp.l3 in float32 to 4096 features, then applies StandardScaler→LogisticRegression. Nominal TSS is offset 60. These defaults do not establish the exact historical revision/layer; new DNA needs a genuinely training-compatible CUDA/backbone stack.

```python
from evo2 import Evo2
from evo2_common import embed_sequences   # bundled in this repo
model = Evo2("evo2_7b")
X = embed_sequences(model, seqs, layer="blocks.28.mlp.l3", pool="mean")  # (N, 4096)
```

The private package needs authorised access. Match scikit-learn 1.9.0 and load only trusted pickles. apply_probe.py performs DNA-to-score inference; non-81-bp inputs warn rather than reject.

```bash
python apply_probe.py --fasta windows.fna --probe evo2_promoter_classifier.pkl --out scores.csv
```

## Files

| File | What it is |
|---|---|
| `evo2_promoter_classifier.pkl` | the deployed probe (scikit-learn `Pipeline`, ~116 KB) |
| `evo2_common.py` | batched Evo2 embedding helper (`embed_sequences`, `revcomp`) |
| `apply_probe.py` | sequences → Evo2 embeddings → raw classifier score (CSV column `p_promoter`) |
| `SOURCE_CODE_LICENSE` | root MIT copyright/license notice for bundled project source; included by the upload script, separate from probe artifact licensing |
| `REGULONDB_LICENSE.md` | training-source data license, attribution and changes; separate from probe artifact licensing |
| `verification.json` | model/data hashes, alignment checks, saved-feature evaluation and remaining limits |
| `recovery_SHA256SUMS` | original relative-path SHA-256 receipt identifying all 159 recovered files; inputs themselves remain local |


## Training data

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters.

The deployed T15_xs cohort has 50,190 81-bp windows, 25,095 labels of each class. Recovered MMseqs2 memberships keep clusters within one split. The retained recipe uses identity 0.6; the original invocation/version is not recovered. Train/validation/test are 44,085/3,052/3,053; positive counts 24,001/547/547. Silver positives and cross-species positives are training-only, leaving a test no-skill AP near 0.18.

| Source | Windows | Species | Kind |
|---|---:|---|---|
| Hocq et al. 2022 published TSS | 5,506 | *C. acetobutylicum* ATCC 824, *C. beijerinckii* NCIMB 8052 / DSM 6423 | in-genus, verified |
| Housekeeping-gene upstreams | 13,838 | in-genus, across the 166 genomes | **silver**: annotation/position-derived putative positives, no mapped TSS or measured promoter activity; training-only, sample-confidence weight 0.3 (not biological probability) |
| RegulonDB | 3,827 | *E. coli* K-12 | cross-species, curated |
| *Bacillus licheniformis* published TSS | 888 | *B. licheniformis* | cross-species — **see erratum** |
| *Ruminiclostridium cellulolyticum* H10 published TSS | 663 | ex-*Clostridium* relative | cross-species |
| DBTBS | 373 | *B. subtilis* | cross-species, curated |


Only Hocq supplies held-out positives (4,412/547/547); 19,344 positives are in-genus. Bakta products identify housekeeping loci: 22,760 selected loci reduce to 14,659 proximity-derived representatives among selected loci, not a full operon map. Silver crops lack mapped starts, receive sample-confidence weight 0.3 for deployed training loss (not biological probability) and do not prove constitutive expression.

Four putative-negative classes are CDS, intergenic, downstream and training-only SRS shuffles of silver cores. The custom shuffle has a fallback; no class is experimentally guaranteed promoter-free. All 166 annotations record Bakta v1.11.4/database v6.0 light.

**Erratum:** 888 B. licheniformis positives use GCF_000011645.1/NC_006270.3 instead of the source-compatible GCF_000008425.1/NC_006322.1. They are training-only but remain in deployed weights. Correcting documentation does not retrain the artifact. Target C. butyricum is excluded from supervised construction; backbone pretraining exclusion is unknown. Benchmark strains supplied silver (annotation/position-derived putative positive) and putative-negative windows, though their published TSS were withheld.

## Training and evaluation

Weighted validation AP selected logistic regression (0.7007 versus RF about 0.61), then refit train+validation (47,137 rows). LogisticRegression max_iter=5000 otherwise uses recorded scikit-learn defaults. The backbone is never fine-tuned.

| Regime | Metric | Score |
|---|---|---|
| **A** · in-genus test (`T15_xs`, test weights all 1; weighted/unweighted AP coincide) | AUPRC | 0.703 |
| **C** · genome scale, *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | **71.0% / 58.0%** |
| **C** · genome scale, *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | **85.8% / 71.4%** |


The table's genomic values are standalone runs. Paired deployment Evo2 is 69.4/58.4% and 87.6/72.5%; the trio is 74.4/61.6% and 87.0/76.6%. Use `--align-grid` when joining scanner outputs; equal stride is not equal reverse-strand grid.

## Benchmark interpretation

Restricted step-10 regions span 300 bp upstream/50 bp into annotated genes. Same-contig/strand 81-bp windows expand ±40 bp to cover published TSS; unmatched starts merge at ≤81 bp. Precision proxy is covered TSS/(covered TSS+unmatched peaks), not the fraction of calls matching or measured biological purity. Truth is 1,245 C. ljungdahlii starts (GCF_000143685.1/NC_014328.1) and 2,885 C. autoethanogenum starts (GCF_000484505.1/NC_022592.1). Compare within a genome; truth coverage differs.

Deployment intersects member coordinates before averaging raw-score percentiles. Mean rank is not probability, strength or unanimity; no calibrator is used. Labels/motifs do not establish activity, sigma use, constitutive expression or downstream transcripts. No experimental validation accompanies the release.

## CPU saved-feature check

The trusted probe has 4096 features, classes [0,1], coefficients (1,4096) and scaler count 47,137, not a unique identity proof. Matching recovered test features reproduced test AP 0.70258010/AUROC 0.89534660 on 3,053 rows/547 positives. Recovered test weights are all 1, so weighted and unweighted test AP/AUROC coincide; evaluation origins differ. Validation belongs to refitting. Obtain the large test arrays separately; this command does not fit or infer new DNA:

```bash
uv run --no-project --python 3.12 \
  --with numpy==2.5.4 --with scikit-learn==1.9.0 --with joblib==1.5.3 \
  python - evo2_promoter_classifier.pkl "$HOME/cb-promoter-original-run/evo2_embeddings" <<'PY'
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

Original 50,190 source rows/encodings align and all 159 recovery hashes pass. verification.json records the probe/input identities and scope. These checks do not audit fresh GPU extraction. Historical runtime around one GPU-hour varies with hardware/batching/coverage.

## Backbone credit

Evo 2, Arc Institute, Brixi et al.: [paper](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1), [code](https://github.com/ArcInstitute/evo2).

## Source credit and data terms

Cite the model base, this project and the data used:

- Hocq et al., 2022, in-genus training TSS, [doi:10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21).
- Al-Bassam et al., 2018, C. ljungdahlii benchmark, [doi:10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6).
- RegulonDB export 13.5 (2025-01-28); database v12.0 paper, [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072).
- [DBTBS](https://dbtbs.hgc.jp), B. subtilis promoters; export release unrecorded.
- Xu et al., 2015, R. cellulolyticum TSS, [doi:10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900), PRJNA247408.
- Wiegand et al., 2013, B. licheniformis TSS, [doi:10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667), PRJNA188930.
- de Souza Pinto Lemgruber et al., 2019, C. autoethanogenum benchmark, [doi:10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549), PRJNA428323.
- NCBI RefSeq genomes, Bakta annotation and MMseqs2 clustering.

The bundled REGULONDB_LICENSE.md records [official April 2025 CC-BY-NC-4.0 terms](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md). Preserve attribution/licence links and disclose sanitising, extraction, deduplication/clustering and mixed-label processing. Academic/noncommercial sharing/adaptation is permitted; commercial data use needs an agreement. Model/source-code licences remain separate. This package contains fitted parameters/helpers, not raw database exports.
