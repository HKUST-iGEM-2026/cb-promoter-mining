---
license: cc-by-nc-4.0
base_model: neuralbioinfo/prokbert-mini
tags:
  - biology
  - genomics
  - dna
  - promoter-prediction
  - clostridium
  - prokbert
  - igem
pipeline_tag: text-classification
library_name: transformers
---

# ProkBERT-mini promoter classifier

HKUST iGEM 2026 fine-tune of [neuralbioinfo/prokbert-mini](https://huggingface.co/neuralbioinfo/prokbert-mini) for ranking promoter-like DNA. It is one member of the ProkBERT/CNN-v2/frozen-Evo2 genome-mining ensemble, **not an expression-strength predictor**. This repository is public and ungated: downloads do not require a token. It provides weights and code, not a hosted prediction service.

- **Input:** 81-bp A/C/G/T DNA in transcription orientation; the discovery frame places the nominal TSS at zero-based offset 60 (60 upstream bases, TSS base, 20 downstream bases). Silver examples do not have mapped TSS.
- **Output:** raw positive-class softmax score. It is not a calibrated target-genome promoter probability, activity measurement or sigma-factor assignment.

## Results

| Evaluation | Result | Interpretation |
|---|---|---|
| Deployed `T15_xs` validation AP | 0.8080683612857302 | Original deployed-run record; unweighted AP |
| Deployed `T15_xs` test AP | **0.8249372466065321** | Original deployed-run record; 3,053 windows, 547 positives; no-skill AP ≈0.179168 |
| Restricted genomic scan, *C. ljungdahlii* | 64.4% / 47.7% | Precision proxy at recall 0.1 / 0.5 |
| Restricted genomic scan, *C. autoethanogenum* | 76.7% / 64.6% | Precision proxy at recall 0.1 / 0.5 |
| Paired three-model ensemble, *C. ljungdahlii* | 74.4% / 61.6% | Ensemble result, not this classifier alone |
| Paired three-model ensemble, *C. autoethanogenum* | 87.0% / 76.6% | Ensemble result, not this classifier alone |

The historical test AP **0.884** belongs to an earlier plain-0.6 checkpoint/split (967/3,318 test positives), not the deployed artifact. Do not substitute it for the deployed result.

Genomic scans restrict step-10 windows to 300 bp upstream/50 bp into annotated genes. Same-contig/strand 81-bp windows expand ±40 bp for published-TSS coverage; unmatched starts at gaps ≤81 bp merge. Precision proxy is **covered TSS / (covered TSS + merged unmatched peaks)**, not measured biological purity or the fraction of calls matching. Reference catalogues are incomplete: 1,245 *C. ljungdahlii* starts (GCF_000143685.1/NC_014328.1) and 2,885 *C. autoethanogenum* starts (GCF_000484505.1/NC_022592.1). Compare within a genome and protocol; these genomes supplied silver/putative-negative training windows, although their published benchmark TSS were withheld.

In the target *C. butyricum* DSM 10702 assembly GCF_014131795.1, the ensemble produced **929 retained regions, 180 with a placed −10 hit and 45 with both −10/−35 positional hits**. These are nested computational subsets, not validated promoters. The ensemble intersects member coordinates, converts raw scores to average-tie percentiles on the common pool and averages them; mean rank is neither probability, strength nor unanimity. The independent DH5α SVR is not an ensemble member.

### Downloads and reuse

The public [results and downloads index](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/README.md) provides full benchmark tables, the 929-region catalogue, 45-candidate subset, named first-ten TSV/FASTA, gene-context files, reporter/reference DNA and conditional DH5α predictions. **Shortlist DL10 is `cand_00101` / `P_prfB`; `pptb_fU` is a separate reference**, despite its historical DL10 notebook alias. Positional motif hits and nearest-downstream annotations do not establish activity, constitutivity or cotranscription. No native *C. butyricum* validation or candidate-resolved quantitative assay accompanies these model outputs.

## Load a real candidate

Install PyTorch, Transformers and the tokenizer dependencies in a compatible environment; see the project's [setup](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/docs/SETUP.md). Review custom Python before using `trust_remote_code=True`. The following pins identify inspected released weights, tokenizer files and external code separately; they are not proof of exact historical training revisions. No login is needed.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "CloBiT/prokbert-mini-clostridium-promoter"
# Tokenizer files are supplied by the base repository, not this fine-tune.
tok = AutoTokenizer.from_pretrained(
    "neuralbioinfo/prokbert-mini", trust_remote_code=True,
    revision="feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc",
    code_revision="366d9336b6b89d7c1feddf56d9b01be8065ccc43")
model = AutoModelForSequenceClassification.from_pretrained(
    REPO, trust_remote_code=True,
    revision="815f637e1a77ac25757b0add21a22668090cce31").eval()

# DL1 / cand_00029: exact transcription-oriented summit_seq, inferred TSS offset 60.
seq = "AATGATTTACTCATACTATAAATATGCTATAATGGTATAAAATATTATTATATTTACAACCTTAAATTTTTAGGAGGATAA"
inputs = tok([seq], padding="max_length", truncation=True,
             max_length=81, return_tensors="pt")
with torch.no_grad():
    score = torch.softmax(model(**inputs).logits, dim=1)[0, 1].item()
print(f"raw promoter-class score = {score:.3f}")
```

**81 bp is a dataset/deployment convention, not ProkBERT's architectural limit.** Overlapping six-mers with shift 1 give 76 sequence tokens plus CLS/SEP = **78 tokens**; `max_length=81` pads these to 81 token positions with masked padding. The checkpoint has `max_position_embeddings=1024`. Longer inputs are possible but do not inherit the released 81-bp evaluation; no retained length comparison establishes 81 bp as optimal.

## Training and provenance

The `T15_xs` cohort contains **50,190 windows**, 25,095 per class. Train/validation/test counts are **44,085/3,052/3,053**, with **24,001/547/547 positives**. Recovered MMseqs2 memberships keep clusters within one split; the retained identity recipe is 0.6, but the original invocation/version is not recovered.

| Positive source | Windows | Evidence |
|---|---:|---|
| Hocq et al. published TSS | 5,506 | *C. acetobutylicum* ATCC 824; *C. beijerinckii* NCIMB 8052/DSM 6423; only held-out positive source (4,412/547/547) |
| Housekeeping-gene upstreams | 13,838 | In-genus, annotation/position-derived **silver** labels across 166 genomes; training-only |
| RegulonDB | 3,827 | Curated cross-species *E. coli* K-12 |
| *B. licheniformis* published TSS | 888 | Cross-species; wrong-assembly extraction, below |
| *R. cellulolyticum* H10 TSS | 663 | Cross-species relative |
| DBTBS | 373 | Curated cross-species *B. subtilis* |

Silver means putative promoter-positive upstream DNA, **not mapped TSS or experimentally measured promoter activity**. Silver confidence weight 0.3 scales training loss, not biological probability; silver and cross-species positives are training-only. All 166 annotations record Bakta 1.11.4/database 6.0 light; the assembly manifest is in the public results index. Of 25,095 positives, 19,344 are in-genus. Housekeeping selection reduced 22,760 selected loci to 14,659 proximity-derived representatives among those loci, not a full operon map.

Four putative-negative classes are CDS, intergenic, downstream and SRS shuffles of 81-bp silver cores. **SRS negatives occur in train, validation and test**, unlike the positive-label restriction: the splitter's training-pin condition requires `label == 1`. The custom shuffle intends dinucleotide preservation but has a plain-shuffle fallback; neither exact dinucleotide preservation nor motif destruction is guaranteed, and no class is experimentally guaranteed promoter-free.

| Negative source | Train | Validation | Test |
|---|---:|---:|---:|
| SRS | 5,505 | 652 | 701 |
| T14_CDS | 5,593 | 663 | 656 |
| T14_Downstream_3Prime | 5,477 | 721 | 663 |
| T14_Intergenic_NonPromoter | 3,509 | 469 | 486 |

The [negative-source CSV](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/training_negative_sources.csv) counts label-0 rows in retained T15_xs Parquets; all three SHA-256 values match the published training manifest. CNN k-mer frequencies are **input features, not replacement negative generation**; Evo2 embeds the same labelled windows and its probe refits on train+validation. Recorded alignment connects the 50,190 source rows to model inputs. Earlier SRS-training-only prose was incorrect. This card-only factual correction changes no weights, splits or metrics.

**Retained training-data error:** 888 *B. licheniformis* positives used GCF_000011645.1/NC_006270.3 rather than source-compatible GCF_000008425.1/NC_006322.1. They remain in deployed weights; this documentation update does not retrain them. Target *C. butyricum* was excluded from supervised construction; backbone pretraining exposure is unknown.

The backbone and two-class attention-pooling head were fine-tuned with sample-weighted loss and unweighted validation-AP selection: learning rate 4.675283456079137e-05, train batch 16, maximum 10 epochs, weight decay 0.07894770663194649, hidden/classifier dropout 0.1025172750748932, seed 42 and bf16 training (stored weights float32). The saved warmup fraction 0.11090738634337446 is stored in float `warmup_steps`, not `warmup_ratio`. Architecture: hidden 384, 6 layers/6 heads, intermediate 3072, vocabulary 4101; 20,490,627 fitted parameters.

Key files are `model.safetensors`, `config.json`, bundled `models.py`, `training_args.bin`, `metrics.json`, `REGULONDB_LICENSE.md`, `verification.json` and `recovery_SHA256SUMS`. Recorded recovery checks align all 50,190 rows/encodings and hash 159 source files. Prior CPU checks scored the 45 both-hit candidates with helper/CLI agreement within six-decimal rounding; fixed/longest padding agreed on those inputs. These checks neither remeasure held-out AP nor audit fresh GPU scans. Large training arrays/tracks are not bundled.

## Licences and citations

- **Fitted weights:** CC-BY-NC-4.0. External [nbrg-transformers tokenizer/model code](https://huggingface.co/neuralbioinfo/nbrg-transformers) declares **CC-BY-NC-ND-4.0**; bundled `models.py` matches revision `366d9336b6b89d7c1feddf56d9b01be8065ccc43`. Project MIT terms do not relicense it.
- **Training data:** bundled `REGULONDB_LICENSE.md` records [official April 2025 CC-BY-NC-4.0 terms](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md). Preserve attribution, licence links and changes (sanitising, extraction, deduplication/clustering, mixed-label processing); commercial data use needs an agreement. Weight/code declarations do not waive source-data restrictions. This package contains fitted parameters/code, not raw database exports.
- ProkBERT: Ligeti et al. (2024), [doi:10.3389/fmicb.2023.1331233](https://doi.org/10.3389/fmicb.2023.1331233). Cite the base model and this HKUST iGEM 2026 project.
- Training TSS: Hocq et al. (2022), [doi:10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21); Xu et al. (2015), [doi:10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900), PRJNA247408; Wiegand et al. (2013), [doi:10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667), PRJNA188930.
- RegulonDB export 13.5 (2025-01-28), database paper [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072); [DBTBS](https://dbtbs.hgc.jp), export release unrecorded; NCBI RefSeq, Bakta and MMseqs2.
- Genomic benchmarks: Al-Bassam et al. (2018), [doi:10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6); de Souza Pinto Lemgruber et al. (2019), [doi:10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549), PRJNA428323.
