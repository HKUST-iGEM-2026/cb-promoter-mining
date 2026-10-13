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

HKUST iGEM 2026 fine-tune of [neuralbioinfo/prokbert-mini](https://huggingface.co/neuralbioinfo/prokbert-mini), deployed with fusion CNN-v2 and the frozen-Evo2 probe. The weight artifact declares **CC-BY-NC-4.0**. It scores promoter-like windows, not expression strength.

## Load

Input is 81-bp A/C/G/T DNA in transcription orientation, nominal TSS offset 60. The private repository needs authorised access. Pin trusted classifier, tokenizer-file and external-code revisions separately; max_length counts tokens.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "CloBiT/prokbert-mini-clostridium-promoter"
# The tokenizer is unchanged from the base model, so load it from there (this repo ships the
# fine-tuned weights + custom architecture, not a copy of the base's tokenizer):
tok = AutoTokenizer.from_pretrained(
    "neuralbioinfo/prokbert-mini", trust_remote_code=True,
    revision="feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc",
    code_revision="366d9336b6b89d7c1feddf56d9b01be8065ccc43")
model = AutoModelForSequenceClassification.from_pretrained(
    REPO, trust_remote_code=True, revision="815f637e1a77ac25757b0add21a22668090cce31").eval()

seq = "A" * 81  # API illustration only; replace with oriented DNA, inferred TSS at offset 60
inputs = tok([seq], padding="max_length", truncation=True, max_length=81, return_tensors="pt")
with torch.no_grad():
    score = torch.softmax(model(**inputs).logits, dim=1)[0, 1].item()
print(f"raw promoter-class score = {score:.3f}")
```

## Files

| File | What it is |
|---|---|
| `model.safetensors` | fine-tuned weights, 20,490,627 parameters, float32 |
| `config.json` | `ProkBertConfig`: hidden 384, 6 layers, 6 heads, intermediate 3072, attention pooling, vocab 4101 |
| `models.py` | the ProkBERT architecture, loaded via `trust_remote_code=True` |
| `training_args.bin` | the `TrainingArguments` of the deployed run |
| `REGULONDB_LICENSE.md` | training-source data license, attribution and changes; separate from the fitted-weight license |
| `metrics.json` | recovered original validation/test metrics; `weighted` refers to training loss, not AP |
| `verification.json` | model/data hashes, alignment checks, saved-feature evaluation and remaining limits |
| `recovery_SHA256SUMS` | original relative-path SHA-256 receipt identifying all 159 recovered files; inputs themselves remain local |


Load the tokenizer from the base model; it is not duplicated here. External [nbrg-transformers code](https://huggingface.co/neuralbioinfo/nbrg-transformers) declares CC-BY-NC-ND-4.0. Recovered models.py matches revision 366d9336b6b89d7c1feddf56d9b01be8065ccc43; this is not proof of historical training revisions. Project MIT terms do not relicense it.

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

## Training

The backbone and two-class head are fine-tuned with per-sample weighted loss; selection uses unweighted validation AP.

| | |
|---|---|
| learning rate | 4.675283456079137e-05 |
| batch size | 16 |
| epochs | 10 |
| warmup ratio | 0.11090738634337446 |
| weight decay | 0.07894770663194649 |
| dropout (hidden & classifier) | 0.1025172750748932 |
| classifier pooling | attention |
| precision | bf16 mixed precision (weights stored float32) |


## Evaluation

| Regime | Metric | Score |
|---|---|---|
| **A** · in-genus test (`T15_xs`, unweighted AP) — recovered deployed-run record | AUPRC | 0.824937 |
| **A** · in-genus test — *earlier checkpoint\** | AUPRC | 0.884 |
| **C** · genome scale, *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 64.4% / 47.7% |
| **C** · genome scale, *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 76.7% / 64.6% |


Deployed validation/test AP is 0.8080683612857302/0.8249372466065321, recovered from the original metrics JSON, not remeasured. The 0.884 comparison uses an earlier plain-0.6 checkpoint/split (967/3,318 positives).

## Benchmark interpretation

Restricted step-10 regions span 300 bp upstream/50 bp into annotated genes. Same-contig/strand 81-bp windows expand ±40 bp to cover published TSS; unmatched starts merge at ≤81 bp. Precision proxy is covered TSS/(covered TSS+unmatched peaks), not the fraction of calls matching or measured biological purity. Truth is 1,245 C. ljungdahlii starts (GCF_000143685.1/NC_014328.1) and 2,885 C. autoethanogenum starts (GCF_000484505.1/NC_022592.1). Compare within a genome; truth coverage differs.

Deployment intersects member coordinates before averaging raw-score percentiles. Mean rank is not probability, strength or unanimity; no calibrator is used. Labels/motifs do not establish activity, sigma use, constitutive expression or downstream transcripts. No experimental validation accompanies the release.

## Recovery checks

All 50,190 recovered source rows/encodings align; 159 recovered files passed source hashes. The classifier previously scored 45 promoters with both −10 and −35 motifs on macOS CPU (Python 3.13.5, torch 2.11.0, Transformers 5.3.0); helper/CLI agreed within six-decimal rounding and fixed/longest padding gave identical scores on those inputs. These checks do not remeasure held-out metrics or audit fresh GPU scans. verification.json records the scope; large inputs are not packaged.

```bibtex
@article{ligeti2024prokbert,
  title={ProkBERT family: genomic language models for microbiome applications},
  author={Ligeti, Bal\'azs and Szepesi-Nagy, Istv\'an and Bodn\'ar, Babett and Ligeti-Nagy, No\'emi and Juh\'asz, J\'anos},
  journal={Frontiers in Microbiology}, volume={14}, year={2024}, doi={10.3389/fmicb.2023.1331233}
}
```

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
