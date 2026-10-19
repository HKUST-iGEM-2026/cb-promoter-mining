---
license: cc-by-4.0
tags:
  - biology
  - genomics
  - dna
  - promoter-prediction
  - clostridium
  - cnn
  - pytorch
  - igem
pipeline_tag: text-classification
---

# Fusion CNN promoter classifier

HKUST iGEM 2026 from-scratch multi-encoding CNN-v2 for ranking promoter-like DNA, deployed alongside ProkBERT-mini and the frozen-Evo2 probe. **This classifier does not predict expression strength.** The package is public and ungated; downloads require no login/token. It is not a hosted prediction service.

- **Input:** transcription-oriented 81-bp DNA, nominal TSS at zero-based offset 60. The helper center-crops/N-pads other lengths, which does not establish a correct biological frame.
- **Output:** raw positive-class score, not calibrated target-genome probability, strength or sigma-factor assignment.
- **Architecture:** one-hot/ENAC convolution branches (kernels 7/5/3, filters 64/128/256, batch normalization/pooling) plus 16-dimensional dinucleotide and 64-dimensional trinucleotide MLP branches; fusion dimension 256, ENAC width 5. Concatenation order is onehot,dinuc,trinuc,enac; N contributes no base-channel count.

## Results

| Evaluation | Result | Scope |
|---|---|---|
| Deployed `T15_xs` validation AP | 0.7380835045313733 | Original deployed-run record, unweighted AP |
| Deployed `T15_xs` test AP | **0.7255828423854476** | 3,053 windows/547 positives; no-skill AP ≈0.179168 |
| Same test AUROC / accuracy / F1 / MCC | 0.898 / 0.834 / 0.627 / 0.543 | Rounded original-run metrics |
| Composition-matched real-versus-SRS AUROC | 0.924 | Earlier `T21_final`/plain-0.6 checkpoint, **not the deployed xs model** |
| Restricted genomic scan, *C. ljungdahlii* | 64.4% / 47.2% | Precision proxy at recall 0.1 / 0.5 |
| Restricted genomic scan, *C. autoethanogenum* | 81.0% / 66.0% | Precision proxy at recall 0.1 / 0.5 |
| Paired three-model ensemble, *C. ljungdahlii* | 74.4% / 61.6% | Ensemble, not CNN alone |
| Paired three-model ensemble, *C. autoethanogenum* | 87.0% / 76.6% | Ensemble, not CNN alone |

Historical test AP **0.848** uses an earlier checkpoint/split (967/3,318 test positives), not this deployed artifact. The 0.924 SRS control tests discrimination under a supplied shuffle/sampling procedure, not a unique −10/−35 mechanism. Historical matched-step values 63.1/48.2% and 81.2/67.1%, and the step-1 *C. ljungdahlii* result 71.6/51.9%, are separate runs rather than controlled replacements for the table.

Genomic scans restrict step-10 windows to 300 bp upstream/50 bp into annotated genes. Same-contig/strand 81-bp windows expand ±40 bp to cover catalogue TSS; unmatched starts at gaps ≤81 bp merge. Precision proxy is **covered TSS / (covered TSS + merged unmatched peaks)**, not measured biological purity or the fraction of calls matching. Truth contains 1,245 *C. ljungdahlii* starts (GCF_000143685.1/NC_014328.1) and 2,885 *C. autoethanogenum* starts (GCF_000484505.1/NC_022592.1). Catalogues are incomplete; compare within a genome/protocol. These strains supplied silver/putative-negative training windows, although published benchmark TSS were withheld.

The target *C. butyricum* DSM 10702 mine (GCF_014131795.1) retained **929 regions**, including **180 with a placed −10 hit** and **45 with both −10/−35 positional hits**. Ensemble scores intersect coordinates, convert member raw scores to average-tie percentiles on that common pool and average them. Mean rank is not probability, strength or unanimity; the independent DH5α SVR is not a fourth member.

### Downloads and reuse

The public [results and downloads index](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/README.md) links full benchmark tables, the 929-region catalogue, both-hit subset, first-ten named TSV/FASTA, gene-context files, reporter/reference DNA and separate DH5α predictions. **DL10 in the shortlist is `cand_00101` / `P_prfB`; `pptb_fU` is an independent reference**, despite an older DL10 notebook alias. Catalogue motifs are positional annotations, not confirmed promoter activity or sigma use. Nearest-downstream genes are hypotheses, not proven transcripts. No native *C. butyricum* validation or candidate-resolved quantitative assay accompanies these model outputs.

## Load a real candidate

Install PyTorch, NumPy, scikit-learn, tqdm and `huggingface_hub` in a compatible environment; see [setup](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/docs/SETUP.md). Review bundled Python before importing it. Pinning the package revision fixes weights and executable feature/model code together; no login is needed.

```python
import json
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download

path = Path(snapshot_download(
    "CloBiT/cnn-fusion-clostridium-promoter",
    revision="191fc88fffeb52ecbf4bdb27702a734365545815"))
sys.path.insert(0, str(path))
from train_cnn_v2 import build_model_from_config, predict_proba

cfg = json.loads((path / "config.json").read_text())
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(path / "best_model.pt", map_location="cpu", weights_only=True))
model.eval()
# DL1 / cand_00029: exact oriented summit_seq, inferred TSS offset 60.
seqs = ["AATGATTTACTCATACTATAAATATGCTATAATGGTATAAAATATTATTATATTTACAACCTTAAATTTTTAGGAGGATAA"]
scores = predict_proba(model, seqs, device="cpu", enac_window=cfg.get("enac_window", 5))
print(scores)  # raw positive-class scores, not expression predictions
```

## Training and provenance

The `T15_xs` cohort has **50,190 81-bp windows**, 25,095 per class. Train/validation/test: **44,085/3,052/3,053** rows and **24,001/547/547** positives. Recovered MMseqs2 memberships keep clusters within one split; identity recipe 0.6 is retained but the original invocation/version is not recovered.

| Positive source | Windows | Evidence |
|---|---:|---|
| Hocq et al. TSS | 5,506 | *C. acetobutylicum* ATCC 824 and *C. beijerinckii* NCIMB 8052/DSM 6423; only held-out positive source (4,412/547/547) |
| Housekeeping-gene upstreams | 13,838 | In-genus, annotation/position-derived **silver** labels across 166 genomes; training-only |
| RegulonDB | 3,827 | Curated cross-species *E. coli* K-12 |
| *B. licheniformis* TSS | 888 | Cross-species, wrong-assembly extraction, below |
| *R. cellulolyticum* H10 TSS | 663 | Cross-species relative |
| DBTBS | 373 | Curated cross-species *B. subtilis* |

Silver positives are **not mapped TSS or experimentally measured promoters**. Silver confidence weight 0.3 scales training loss, not biological probability; silver and cross-species positives are training-only. There are 19,344 in-genus positives. Housekeeping selection reduced 22,760 selected loci to 14,659 proximity-derived representatives among selected loci, not a full operon map. All 166 annotations record Bakta 1.11.4/database 6.0 light; an assembly manifest is in the public results index.

Putative-negative classes are CDS, intergenic, downstream and SRS shuffles of 81-bp silver cores. **SRS negatives occur in train, validation and test**, unlike training-only silver/cross-species positives: the splitter's training-pin condition requires `label == 1`. The custom shuffle intends dinucleotide preservation but has a plain-shuffle fallback; neither exact dinucleotide preservation nor motif destruction is guaranteed, and no class is experimentally guaranteed promoter-free.

| Negative source | Train | Validation | Test |
|---|---:|---:|---:|
| SRS | 5,505 | 652 | 701 |
| T14_CDS | 5,593 | 663 | 656 |
| T14_Downstream_3Prime | 5,477 | 721 | 663 |
| T14_Intergenic_NonPromoter | 3,509 | 469 | 486 |

The [negative-source CSV](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/training_negative_sources.csv) counts label-0 rows in retained T15_xs Parquets; all three SHA-256 values match the published training manifest. The saved config uses `results/T16_xs/cnn` and `onehot,dinuc,trinuc,enac`; dinucleotide/trinucleotide frequencies are **input features, not replacement negative generation**. Recorded alignment connects these windows to CNN encodings and Evo2 labels; Evo2 embeds the same labelled windows and its probe refits on train+validation. Earlier SRS-training-only prose was incorrect. This card-only factual correction changes no weights, splits or metrics.

**Retained error:** 888 *B. licheniformis* positives used GCF_000011645.1/NC_006270.3 rather than source-compatible GCF_000008425.1/NC_006322.1. They remain in deployed weights; this card update is not retraining. Target *C. butyricum* was excluded from supervised construction. This CNN was trained from scratch; unknown backbone pretraining exposure concerns the other ensemble members, not this CNN.

Training used AdamW/cosine-annealing warm restarts, sample-weighted loss and unweighted validation-AP early stopping: learning rate 0.0003748224111698842, batch 32, dropout 0.10382857322260322, weight decay 0.00024777455052193346, maximum 30 epochs/patience 8, seed 42.

Key files: `best_model.pt`, `config.json`, `train_cnn_v2.py`, `metrics.json`, `history.json`, `SOURCE_CODE_LICENSE`, `REGULONDB_LICENSE.md`, `verification.json` and `recovery_SHA256SUMS`. Recorded checks align all 50,190 source rows/encodings and hash 159 recovered files; prior CPU inference on the 45 both-hit candidates agreed with the local CLI within six-decimal rounding. These checks do not remeasure held-out metrics. Large training arrays/tracks are not bundled.

## Licences and citations

- **Fitted weights:** CC-BY-4.0. Bundled project helper code has the separate MIT notice `SOURCE_CODE_LICENSE`; neither declaration waives source-data restrictions.
- **Data:** `REGULONDB_LICENSE.md` records [official April 2025 CC-BY-NC-4.0 terms](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md). Academic/noncommercial sharing/adaptation requires attribution, licence links and change disclosure (sanitising, extraction, deduplication/clustering, mixed-label processing); commercial data use needs an agreement. This package contains fitted parameters/helpers, not raw database exports.
- Cite this HKUST iGEM 2026 project and training TSS: Hocq et al. (2022), [doi:10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21); Xu et al. (2015), [doi:10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900), PRJNA247408; Wiegand et al. (2013), [doi:10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667), PRJNA188930.
- RegulonDB export 13.5 (2025-01-28), database paper [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072); [DBTBS](https://dbtbs.hgc.jp), export release unrecorded; NCBI RefSeq, Bakta and MMseqs2.
- Benchmarks: Al-Bassam et al. (2018), [doi:10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6); de Souza Pinto Lemgruber et al. (2019), [doi:10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549), PRJNA428323.
