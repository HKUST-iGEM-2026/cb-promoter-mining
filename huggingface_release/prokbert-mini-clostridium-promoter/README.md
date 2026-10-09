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

# ProkBERT-mini — Clostridium promoter classifier (HKUST iGEM 2026)

A fine-tune of [`neuralbioinfo/prokbert-mini`](https://huggingface.co/neuralbioinfo/prokbert-mini)
that scores whether an 81 bp DNA window is a **σ<sup>A</sup> constitutive promoter**. Built for the
HKUST iGEM 2026 dry-lab project to mine promoters from the AT-rich genome of *Clostridium butyricum*
DSM 10702, and one of the three members of that project's deployment ensemble, alongside
[`CloBiT/cnn-fusion-clostridium-promoter`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter)
and
[`CloBiT/evo2-logreg-clostridium-promoter-probe`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe).

> ⚠️ **License is CC-BY-NC-4.0 (non-commercial).** This is inherited from the base model
> `prokbert-mini`, which is CC-BY-NC-4.0. A fine-tune is a derivative work, so it carries the same
> terms. Fine for academic / iGEM use; **not** for commercial use.

## Intended use

Binary classification of a fixed **81 bp** DNA window: `P(promoter)` vs background. It is designed to
be slid across a genome (with per-model calibration and empirical FDR downstream) to produce an
**enriched, ranked candidate list for experimental validation — not a precise genome-wide promoter
map.** See *Limitations*.

**Input convention.** 81 bp, A/C/G/T. The window is framed so the transcription start site (+1) sits
at 0-based offset 60 (1-based position 61); the −10 box then falls around offset ~50 and the −35 around
offset ~25. Match this framing at inference or scores will not mean what they did in training.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "CloBiT/prokbert-mini-clostridium-promoter"
# The tokenizer is unchanged from the base model, so load it from there (this repo ships the
# fine-tuned weights + custom architecture, not a copy of the base's tokenizer):
tok = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(REPO, trust_remote_code=True).eval()

seq = "A" * 81  # your 81 bp window, TSS at offset 60
inputs = tok(seq, return_tensors="pt")
with torch.no_grad():
    p = torch.softmax(model(**inputs).logits, dim=1)[0, 1].item()
print(f"P(promoter) = {p:.3f}")
```

## Files in this repo

| File | What it is |
|---|---|
| `model.safetensors` | fine-tuned weights, 20,490,627 parameters, float32 |
| `config.json` | `ProkBertConfig`: hidden 384, 6 layers, 6 heads, intermediate 3072, attention pooling, vocab 4101 |
| `models.py` | the ProkBERT architecture, loaded via `trust_remote_code=True` |
| `training_args.bin` | the `TrainingArguments` of the deployed run |
| `REGULONDB_LICENSE.md` | training-source data license, attribution and processing changes; separate from fitted-weight licensing |

The tokenizer is **not** duplicated here — load it from the base model as shown above (6-mer, shift 1).

## Training data

Trained on the project's cross-species-augmented split (`T15_xs`): **50,190 windows, ~50/50
positive/negative**, split **44,085 train / 3,052 validation / 3,053 test**. Sequences were clustered
with MMseqs2 at **0.6 identity** and the split is cluster-disjoint, so near-duplicates never straddle
train/val/test. Cross-species sources are **pinned to train**, so validation and test are in-genus only.

### Where the 25,095 positives come from

| Source | Windows | Species | Kind |
|---|---:|---|---|
| Hocq et al. 2022 published TSS | 5,506 | *C. acetobutylicum* ATCC 824, *C. beijerinckii* NCIMB 8052 / DSM 6423 | in-genus, verified |
| Housekeeping-gene upstreams | 13,838 | in-genus, across the 166 genomes | **silver** (inferred), weight 0.3 |
| RegulonDB | 3,827 | *E. coli* K-12 | cross-species, curated |
| *Bacillus licheniformis* published TSS | 888 | *B. licheniformis* | cross-species — **see erratum** |
| *Ruminiclostridium cellulolyticum* H10 published TSS | 663 | ex-*Clostridium* relative | cross-species |
| DBTBS | 373 | *B. subtilis* | cross-species, curated |

19,344 of the positives are in-genus. The counts above were read from the split itself, grouped by
`source`; only the Hocq set appears in validation and test (4,412 / 547 / 547), every other source is
train-only.

**Silver positives** are upstream regions of housekeeping genes, inferred rather than measured: a −10-crop check found about a third do not frame a promoter, so they
carry weight 0.3 against 1.0 for verified promoters. At full weight they would teach the "AT-rich
means promoter" shortcut this project exists to avoid.

Housekeeping genes were selected by Bakta **product description** (not gene symbol, which the light
Bakta database leaves sparse), excluding regulated genes such as sigma factors, heat-shock chaperones
and RecA. About 22,760 such genes across 166 genomes collapse to 14,659 operon leaders; the 300 bp
immediately 5′ of each leader is taken, and the split keeps the 81 bp nearest the gene.

> **Erratum.** The 888 *B. licheniformis* windows were cut from assembly `GCF_000011645.1`
> (`NC_006270.3`), which the published TSS coordinates do not fit — they match `GCF_000008425.1`
> (`NC_006322.1`), an offset of 147–196 bp. Those windows are therefore effectively random genomic
> sequence labelled positive, not promoters. They are 3.5% of the positives, are pinned to train, and
> appear in none of the evaluation data below.

### The negatives

Four classes, all windows of the same 81 bp geometry: **coding-sequence** interiors, **intergenic
non-promoter**, **downstream / 3′**, and **SRS** — dinucleotide-preserving shuffles of real promoters,
which keep base composition but destroy the −10/−35 signal. Composition-matched negatives are the
main defence against the AT-rich false-positive trap (*Clostridium* GC ~28–32%). SRS is training-only.

### Genomes

166 *Clostridium* RefSeq reference assemblies (NCBI taxon 1485), annotated with Bakta 1.11.4
(DB v6.0 light), supply the silver positives and the negative windows.

- **Target excluded:** *C. butyricum* DSM 10702 is in no training data, and no *C. butyricum* strain
  is among the 166.
- **Validation genomes:** the published start sites of *C. ljungdahlii* and *C. autoethanogenum* were
  never training positives, but both strains **are** among the 166, so they contributed silver
  positives and negative windows. The genome-scale numbers below are therefore not fully independent
  of training.

## Training procedure

Fine-tuned from `neuralbioinfo/prokbert-mini` with each window's `weight` column (silver 0.3 ×
class-balance) applied as a **per-sample loss weight**; the selection metric is weighted validation
AUPRC. Hyperparameters come from an Optuna sweep and were read back from the deployed checkpoint:

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

Three regimes, because a balanced test score does not predict genome-scale behaviour.

| Regime | Metric | Score |
|---|---|---|
| **A** · in-genus test, weighted — *earlier checkpoint\** | AUPRC | 0.884 |
| **C** · genome scale, *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 64.4% / 47.7% |
| **C** · genome scale, *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 76.7% / 64.6% |

\* Measured on the previous ProkBERT checkpoint, trained with earlier hyperparameters and without the
two cross-species TSS sets, on the plain 0.6 split (3,318-row test, 967 positives, so a no-skill AUPRC
of ≈0.29). No in-distribution test score was saved for the checkpoint published here, and it would sit
against a different baseline: this checkpoint's `T15_xs` test holds only verified in-genus positives
(547 of 3,053 rows, ~18%), because silver and cross-species positives are pinned to train. The whole
split is ~50/50; val and test are not, and their weights are left at natural prevalence.

**How Regime C is measured.** The model scores every window of a validation genome at step 10, but
only windows inside candidate promoter regions (300 bp upstream to 50 bp into each annotated gene, on
the gene's strand) count toward the result. A window within 40 bp of a published TSS on the same
contig and strand is a hit; recall is the fraction of that genome's published TSS hit at a given
threshold. Consecutive non-hit windows within 81 bp collapse into one false-positive **peak**, and
precision is hits / (hits + FP peaks). Truth sets: *C. ljungdahlii* DSM 13528 `GCF_000143685.1`
(`NC_014328.1`), 1,245 TSS from Al-Bassam et al. 2018; *C. autoethanogenum* DSM 10061
`GCF_000484505.1` (`NC_022592.1`), 2,885 TSS.

Absolute precision is **not** comparable between the two genomes — *C. autoethanogenum* has a denser
truth catalogue, so every model reads higher there. Only within-genome differences between models are
meaningful. In the deployed 3-model ensemble (mean rank-percentile) the same scan reaches 74.4% /
61.6% and 87.0% / 76.6%.

Balanced-test accuracy far overstates genome-scale precision, because real promoters are ~0.2% of
genomic windows; judge deployment by the *number* of false-positive peaks, not the balanced score.

## Limitations

- **No bacterial promoter predictor achieves high genome-wide precision** — the −10/−35 motif is too
  low-information in AT-rich DNA (*Clostridium* GC ~28–32%). Treat outputs as an enriched candidate
  list requiring wet-lab (e.g. dRNA-seq TSS) confirmation.
- Verified in-genus training and evaluation data is thin (~thousands of promoters from three species),
  so in-genus gains carry wide error bars.
- Trained for σ<sup>A</sup> constitutive promoters and the 81 bp / TSS-at-60 framing; other σ factors,
  other window sizes, or other framings are out of distribution.
- 3.5% of the training positives are mislabelled (see the erratum) and the validation genomes
  contributed negatives and silver positives, so the reported genome-scale numbers are a slight
  over-estimate of performance on a truly unseen genome.
- No wet-lab validation of the predictions had been completed when this card was written.

## Citation & attribution

Base model — ProkBERT (CC-BY-NC-4.0), which sets this repo's license:

```bibtex
@article{ligeti2024prokbert,
  title={ProkBERT family: genomic language models for microbiome applications},
  author={Ligeti, Bal\'azs and Szepesi-Nagy, Istv\'an and Bodn\'ar, Babett and Ligeti-Nagy, No\'emi and Juh\'asz, J\'anos},
  journal={Frontiers in Microbiology}, volume={14}, year={2024}, doi={10.3389/fmicb.2023.1331233}
}
```

Training and evaluation data — please cite these if you build on this model:

- **Hocq R, Jagtap S, Boutard M, Tolonen AC, Duval L, Pirayre A, Lopes Ferreira N, Wasels F.**
  Genome-Wide TSS Distribution in Three Related Clostridia with Normalized Capp-Switch Sequencing.
  *Microbiol Spectr.* 2022;10(2):e0228821. doi:[10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21)
  — the in-genus training positives.
- **Al-Bassam MM, Kim JN, Zaramela LS, Kellman BP, Zuniga C, Wozniak JM, Gonzalez DJ, Zengler K.**
  Optimization of carbon and energy utilization through differential translational efficiency.
  *Nat Commun.* 2018;9(1):4474. doi:[10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6)
  — the *C. ljungdahlii* benchmark truth.
- **RegulonDB** v13.5 (Salgado H, Gama-Castro S, et al., *RegulonDB v12.0*, *Nucleic Acids Research*
  2023, doi:[10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072)) — the
  [official April 2025 policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md)
  permits academic/noncommercial sharing and adaptation under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), with attribution, license links
  and disclosure of changes.
- **DBTBS**, the *B. subtilis* transcriptional regulation database ([dbtbs.hgc.jp](https://dbtbs.hgc.jp)).
- **Xu C, Huang R, Teng L, et al.** Cellulosome stoichiometry in *Clostridium cellulolyticum* is
  regulated by selective RNA processing and stabilization. *Nat Commun.* 2015;6:6900.
  doi:[10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900) — the *R. cellulolyticum* H10 TSS
  table (dRNA-seq, BioProject [PRJNA247408](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA247408)).
- **Wiegand S, Dietrich S, Hertel R, Bongaerts J, et al.** RNA-Seq of *Bacillus licheniformis*: active
  regulatory RNA features expressed within a productive fermentation. *BMC Genomics.* 2013;14:667.
  doi:[10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667) — the *B. licheniformis*
  DSM 13 TSS table (supplementary Table S5; dRNA-seq BioProject
  [PRJNA188930](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA188930)). This is the `Wiegand2013`
  source tag in the split.
- **de Souza Pinto Lemgruber R, Valgepea K, Gonzalez Garcia RA, de Bakker C, et al.** A TetR-Family
  Protein (CAETHG_0459) Activates Transcription From a New Promoter Motif Associated With Essential
  Genes for Autotrophic Growth in Acetogens. *Front Microbiol.* 2019;10:2549.
  doi:[10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549) — the *C. autoethanogenum*
  benchmark truth (dRNA-seq BioProject
  [PRJNA428323](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA428323)); evaluation only, never training.
- Genome assemblies from **NCBI RefSeq**; annotation with **Bakta** (Schwengers et al.); clustering
  with **MMseqs2** (Steinegger & Söding).

HKUST iGEM 2026 dry-lab, promoter-mining project.

## RegulonDB data licensing

See the [bundled data notice](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter/blob/main/REGULONDB_LICENSE.md)
and [official policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md).
Noncommercial reuse, sharing and adaptation of RegulonDB content are permitted under **CC BY-NC 4.0**;
commercial data use requires a separate agreement. Attribute RegulonDB and the original publications,
link the license and identify changes. Training preparation normalised/filtered sequences, prepared
81 bp windows, deduplicated/clustered them and combined sources and labels. This package contains fitted
parameters/helpers, not database exports or training tables. The fitted-weight **CC-BY-NC-4.0** and
separate source-code licenses remain unchanged.
