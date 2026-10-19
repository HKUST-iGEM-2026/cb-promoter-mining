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

HKUST iGEM 2026 logistic-regression promoter classifier on frozen [Evo2-7B](https://github.com/ArcInstitute/evo2) embeddings. It is one member of the ProkBERT/CNN-v2/Evo2 genome-mining ensemble. **It scores promoter-like DNA, not expression strength; backbone weights are not included.** This probe package is public and ungated; downloading it needs no login/token. It is not a hosted prediction service.

- **Input:** separate 81-bp transcription-oriented DNA windows, nominal TSS at zero-based offset 60. The completed extraction selected the `sequence` column for all 44,085/3,052/3,053 train/validation/test rows. It did **not** stitch examples together or append genomic flanks; the separate 384-bp trial used NTv3, not this probe.
- **Representation:** current inference mean-pools `blocks.28.mlp.l3` in float32 to 4,096 features, then applies `StandardScaler → LogisticRegression`.
- **Output:** raw positive-class score (`p_promoter` in the CLI), not calibrated target-genome probability, strength or sigma use.

## Results

| Evaluation | Result | Scope |
|---|---|---|
| Deployed `T15_xs` test AP | **0.70258010** | Reproduced on saved test embeddings: 3,053 rows/547 positives |
| Same test AUROC | **0.89534660** | Saved-feature evaluation, not fresh DNA-to-embedding inference |
| Standalone restricted scan, *C. ljungdahlii* | 71.0% / 58.0% | Precision proxy at recall 0.1 / 0.5 |
| Standalone restricted scan, *C. autoethanogenum* | 85.8% / 71.4% | Precision proxy at recall 0.1 / 0.5 |
| Evo2 in paired trio run, *C. ljungdahlii* | 69.4% / 58.4% | Separate aligned-grid comparison |
| Evo2 in paired trio run, *C. autoethanogenum* | 87.6% / 72.5% | Separate aligned-grid comparison |
| Three-model ensemble, *C. ljungdahlii* | 74.4% / 61.6% | Paired trio, not this probe alone |
| Three-model ensemble, *C. autoethanogenum* | 87.0% / 76.6% | Paired trio, not this probe alone |

All recovered test weights equal 1, so weighted/unweighted test AP and AUROC coincide; no-skill AP is ≈0.179168. Weighted validation AP selected logistic regression (0.7007 versus random forest about 0.61), then the retained pipeline was refitted on **train+validation: 47,137 rows**. Validation is therefore not held out from this final artifact. `LogisticRegression(max_iter=5000)` otherwise uses the recorded scikit-learn defaults; the backbone was never fine-tuned. Evo2 was not evaluated in the reported SRS composition-control regime.

Genomic scans restrict step-10 windows to 300 bp upstream/50 bp into annotated genes. Same-contig/strand 81-bp windows expand ±40 bp to cover published starts; above-threshold unmatched starts at gaps ≤81 bp merge. Precision proxy is **covered TSS / (covered TSS + merged unmatched peaks)**, not measured biological purity or the fraction of calls matching. Truth contains 1,245 *C. ljungdahlii* starts (GCF_000143685.1/NC_014328.1) and 2,885 *C. autoethanogenum* starts (GCF_000484505.1/NC_022592.1). Catalogues are incomplete. These genomes supplied silver/putative-negative training windows although their published benchmark TSS were withheld; compare within a genome/protocol. The ensemble leads at recall 0.5, not at every reported point. Equal stride does not guarantee identical reverse-strand grids: align coordinates before combining scanner tracks.

The target *C. butyricum* DSM 10702 ensemble mine (GCF_014131795.1) retained **929 regions**, including **180 with a placed −10 hit** and **45 with both −10/−35 positional hits**. Ensemble deployment intersects member coordinates and averages average-tie raw-score percentiles within the shared pool. Mean rank is not probability, expression strength or unanimity. The separate DH5α SVR is not a fourth ensemble member.

### Downloads and reuse

The public [results and downloads index](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/README.md) provides full benchmark tables, the 929-region catalogue, both-hit subset, named first-ten FASTA/TSV, gene-context files, reporter/reference DNA and independent DH5α predictions. **Shortlist DL10 is `cand_00101` / `P_prfB`; `pptb_fU` is a separate reference**, despite its older DL10 notebook alias. Motif/annotation support does not establish activity, constitutivity, sigma usage or cotranscription. No native *C. butyricum* validation or candidate-resolved quantitative assay accompanies these model outputs.

## Download and score DNA

Pin the project probe/helper package and review its Python and pickle before execution. The published package revision below fixes those files; it does **not** pin the separately installed Evo2 backbone/runtime or establish their exact historical extraction revisions. A compatible CUDA/Evo2 stack is required for new DNA; follow [Arc Institute installation instructions](https://github.com/ArcInstitute/evo2) and record the trusted code/weight revisions you install. Match **scikit-learn 1.9.0**, with NumPy and joblib. Do not load untrusted pickles.

```bash
hf download CloBiT/evo2-logreg-clostridium-promoter-probe \
  --revision 6e521c874dd61517df9b17a56f2a75847a054b87 --local-dir evo2_probe
curl -L --fail \
  https://raw.githubusercontent.com/HKUST-iGEM-2026/cb-promoter-mining/main/results/candidates/g131795_first10_canonical_named.fasta \
  -o first10.fasta
python evo2_probe/apply_probe.py \
  --fasta first10.fasta --probe evo2_probe/evo2_promoter_classifier.pkl \
  --model evo2_7b --layer blocks.28.mlp.l3 --pool mean --out scores.csv
```

The `hf` download CLI comes from `huggingface_hub`; downloads need no write credentials. `apply_probe.py` bundles the sequence→embedding→score path and warns rather than rejects non-81-bp inputs. Preserve the deployed orientation/frame instead of relying on that warning. The exact historical extraction code revision/layer remains unresolved: current defaults and agreement on saved embeddings are **not a fresh GPU DNA-to-embedding audit**. The released helper does not expose a backbone-revision flag; stage independently reviewed upstream dependencies rather than treating the command above as a fully locked historical environment.

For already compatible embeddings, inference only needs the trusted scikit-learn pipeline; it does not require rerunning Evo2. Saved-feature evaluation previously reproduced AP/AUROC above. The fitted scaler has 47,137 samples, classes `[0,1]`, coefficients `(1,4096)`; these shapes alone are not an identity proof. `verification.json` records the probe checksum and feature evidence. Large embedding arrays are not included.

## Training data and limitations

The `T15_xs` cohort has **50,190 81-bp windows**, 25,095 per class. Train/validation/test: **44,085/3,052/3,053** rows and **24,001/547/547** positives. Recovered MMseqs2 memberships keep clusters within a split; retained identity recipe 0.6, original invocation/version not recovered.

| Positive source | Windows | Evidence |
|---|---:|---|
| Hocq et al. TSS | 5,506 | *C. acetobutylicum* ATCC 824; *C. beijerinckii* NCIMB 8052/DSM 6423; only held-out positive source (4,412/547/547) |
| Housekeeping-gene upstreams | 13,838 | In-genus, annotation/position-derived **silver** labels across 166 genomes; training-only |
| RegulonDB | 3,827 | Curated cross-species *E. coli* K-12 |
| *B. licheniformis* TSS | 888 | Cross-species; wrong-assembly extraction, below |
| *R. cellulolyticum* H10 TSS | 663 | Cross-species relative |
| DBTBS | 373 | Curated cross-species *B. subtilis* |

Silver labels are **not mapped TSS or experimentally measured promoters**. Their sample-confidence weight 0.3 affects fitting, not biological probability; silver and cross-species positives are training-only. There are 19,344 in-genus positives. Bakta 1.11.4/database 6.0 light annotations supplied housekeeping products; 22,760 selected loci reduced to 14,659 proximity-derived representatives among selected loci, not a full operon map. The public results index includes the 166-assembly manifest.

Putative negatives are CDS, intergenic, downstream and training-only SRS shuffles of silver cores. The custom shuffle has a fallback and no class is guaranteed promoter-free. **Retained error:** 888 *B. licheniformis* positives use GCF_000011645.1/NC_006270.3 rather than source-compatible GCF_000008425.1/NC_006322.1. They remain in the fitted probe's training history; this update does not retrain it. Target *C. butyricum* was excluded from supervised construction; backbone pretraining exposure is unknown.

Package files: `evo2_promoter_classifier.pkl`, `evo2_common.py`, `apply_probe.py`, `SOURCE_CODE_LICENSE`, `REGULONDB_LICENSE.md`, `verification.json` and `recovery_SHA256SUMS`. Recorded provenance aligns all 50,190 source rows/encodings and hashes 159 recovered files. The historical verification record is not a statement of current access status or a complete data archive.

## Licences and citations

- **Probe artifact:** CC-BY-4.0. Bundled project helper code: MIT (`SOURCE_CODE_LICENSE`). The separately downloaded [Evo2 base card](https://huggingface.co/arcinstitute/evo2_7b) declares Apache-2.0; backbone/runtime terms remain separate.
- **Data:** `REGULONDB_LICENSE.md` records [official April 2025 CC-BY-NC-4.0 terms](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md). Preserve attribution/licence links and disclose sanitising, extraction, deduplication/clustering and mixed-label processing. Academic/noncommercial sharing/adaptation is permitted; commercial data use needs an agreement. Probe/code declarations do not waive source-data restrictions. No raw database exports or Evo2 backbone weights are bundled.
- Evo 2: Brixi et al., Arc Institute, [paper](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1), [code](https://github.com/ArcInstitute/evo2). Cite the backbone and this HKUST iGEM 2026 project.
- Training TSS: Hocq et al. (2022), [doi:10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21); Xu et al. (2015), [doi:10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900), PRJNA247408; Wiegand et al. (2013), [doi:10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667), PRJNA188930.
- RegulonDB export 13.5 (2025-01-28), database paper [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072); [DBTBS](https://dbtbs.hgc.jp), export release unrecorded; NCBI RefSeq, Bakta and MMseqs2.
- Benchmarks: Al-Bassam et al. (2018), [doi:10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6); de Souza Pinto Lemgruber et al. (2019), [doi:10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549), PRJNA428323.
