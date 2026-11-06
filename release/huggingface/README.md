# Hugging Face packages and card publication

All four **CloBiT model repositories are public and ungated**. Downloads do not require an account, `HF_TOKEN` or project-authorised access. Uploads require a token with write permission for the target repository. These are downloadable local-inference packages, not a hosted prediction service or a single packaged ensemble.

The three genome-mining classifiers rank promoter-like DNA; the independent DH5α SVR predicts conditional reporter activity and is **not a fourth ensemble member**. Local card edits are not automatically synced to Hugging Face.

## Results

| Package | Purpose | Retained numerical result |
|---|---|---|
| [ProkBERT-mini](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter) | Fine-tuned promoter classifier | Deployed test AP **0.8249372466065321** |
| [CNN-v2 fusion](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter) | From-scratch promoter classifier | Deployed test AP **0.7255828423854476** |
| [Frozen-Evo2 probe](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe) | Logistic regression on 4,096-dimensional frozen embeddings; backbone separate | Saved-feature test AP/AUROC **0.70258010/0.89534660** |
| [DH5α SVR](https://huggingface.co/CloBiT/dh5alpha-promoter-strength-svr) | Separate 50-bp-core reporter-activity prediction | Selection-CV R² **0.266090**; leading candidate DL9 **32.8751 AU**, **474.57%** of the primary predicted `pptb_fU` core |

Classifier test cohort: 3,053 windows/547 positives; AP is unweighted (all Evo2 test weights equal 1). ProkBERT/CNN metrics are original deployed-run records, while Evo2 was evaluated on recovered saved embeddings, not fresh GPU extraction. The paired ensemble's restricted-genome precision proxy at recall 0.1/0.5 is **74.4/61.6%** for *C. ljungdahlii* and **87.0/76.6%** for *C. autoethanogenum*. This proxy uses incomplete published-TSS catalogues and is not measured biological purity; benchmark genomes supplied silver/putative-negative training windows. The known **888 wrong-assembly B. licheniformis training positives remain in deployed artifacts**.

Fresh CPU FP32 test AP was ProkBERT **0.8259149947673579** and CNN-v2 **0.7255402903111163**; CPU BF16 ProkBERT AP was **0.8248204288011242**. These precision-dependent remeasurements do not replace the GPU-era records; the original GPU runtime was not replayed. [Evaluation protocols](../../docs/RESULTS.md#regime-a--held-out-labelled-windows) retain weighting and leakage caveats.

The target ensemble mine contains **929 regions, 180 placed −10 hits and 45 both-hit candidates**. No native *C. butyricum* activity validation accompanies them. SVR ratios are same-model predictions, not measured fold changes: alternative reference framing puts DL9 at **93.67–474.57%**, a scenario range rather than a confidence interval. **Shortlist DL10 is `cand_00101` / `P_prfB`; `pptb_fU` is an independent reference**, despite an older notebook alias.

The Evo2 Hub `finetune` relationship is inferred metadata, not proof of updated backbone weights: the probe uses frozen embeddings. Preserve its valid base-model attribution; see the [probe card](evo2-logreg-clostridium-promoter-probe/README.md). Restriction-mask boundary behavior, sweep/CV/reporting defects and historical calibration risks are documented in [Pipeline](../../docs/PIPELINE.md), not repaired by these cards. Oxidative target tags `cbut_0001` / `FF104_RS17290` and `cbut_0021` / `FF104_RS17285` require annotation-version reconciliation despite verified stored DNA matches; [Results](../../docs/RESULTS.md#4-oxidative-stress-associated-candidates) gives the unresolved discrepancy.

### Downloads and reuse

Start with the [public results and downloads index](../../results/README.md) for benchmark CSVs, catalogues, named first-ten sequences, gene context, reporter/reference DNA, conditional strength predictions and the separate oxidative-stress outputs. The [model guide](../../docs/MODELS.md), [results interpretation](../../docs/RESULTS.md) and [DH5α method](../../docs/ECOLI_STRENGTH_PREDICTIONS.md) retain full evidence and limits. Each subdirectory's `README.md` is the standalone Hub-card source, with pinned downloads/loading examples.

## Publish these documentation changes only

The SRS correction was already merged in source PR #52 (`7cb586c58529db4f81c0e5e094b2d9d98147cb67`); the live cards inspected on 2026-11-03 matched that publication. It is not pending. Current files here are **local card sources**; subsequent edits do not automatically update live cards.

For any separately authorised future documentation-only publication, merge the reviewed source correction first so new source links resolve, then upload **only the changed standalone card README.md files**. No weight reupload, retraining, full artifact release or visibility change is required. The following are manual examples for the three classifier cards, not actions performed by editing this repository:

```bash
pip install -U huggingface_hub
hf auth login
hf auth whoami

hf upload CloBiT/prokbert-mini-clostridium-promoter \
  release/huggingface/prokbert-mini-clostridium-promoter/README.md README.md
hf upload CloBiT/cnn-fusion-clostridium-promoter \
  release/huggingface/cnn-fusion-clostridium-promoter/README.md README.md
hf upload CloBiT/evo2-logreg-clostridium-promoter-probe \
  release/huggingface/evo2-logreg-clostridium-promoter-probe/README.md README.md
```

Choose write-authorised CloBiT credentials only when publication is authorised. Each example updates **only `README.md`** in its existing model repository, not this guide or weights. Upload only cards whose prose changed; if the independent DH5α card changes, use the same README-only pattern for `CloBiT/dh5alpha-promoter-strength-svr`. Review resulting signed-out cards and source links and record the new card revisions. Keep inspected artifact/code revisions pinned rather than replacing them with documentation revisions. No current local correction is claimed uploaded here.

A public download failure is not evidence that an authorised account or duplicate model release is needed: check the exact repository/path/revision and upstream dependency conditions. Write authentication is for publication, not reading these public packages.

## Licences

| Model artifact | Declared licence | Separate terms to preserve |
|---|---|---|
| ProkBERT fine-tuned weights | **CC-BY-NC-4.0** | Base noncommercial terms; external tokenizer/model code declares **CC-BY-NC-ND-4.0** |
| CNN-v2 weights | **CC-BY-4.0** | Project source MIT; source-data restrictions remain separate |
| Evo2 logistic probe | **CC-BY-4.0** | Project helper source MIT; [Evo2 backbone card](https://huggingface.co/arcinstitute/evo2_7b) declares Apache-2.0 |
| DH5α fitted numeric SVR/project inference | **MIT** | Wang/Deep_promoter source-table/procedure MIT notice retained as `UPSTREAM_LICENSE` |

The CNN/Evo2 packages bundle project MIT source terms as `SOURCE_CODE_LICENSE`. ProkBERT's bundled `models.py` matches external code revision `366d9336b6b89d7c1feddf56d9b01be8065ccc43`; project MIT does not relicense it.

The three genome-mining packages carry the [RegulonDB notice](../../REGULONDB_LICENSE.md): academic/noncommercial CC-BY-NC-4.0 sharing/adaptation requires attribution, licence links and change disclosure (sanitising, extraction, deduplication/clustering, mixed-label processing); commercial data use needs an agreement. Weight/code declarations do not waive data terms. The DH5α package instead uses Wang's small published table, not RegulonDB exports, and retains both upstream/project MIT notices. The separately used GPL-3.0-or-later Promoter Calculator is not bundled or needed for cached SVR inference.

## Full artifact releases — only when intentionally needed

This is **not** the card-only route above. `upload_to_hf.py` uploads models/helpers, notices, verification records **and** cards. Root `uv sync` does not install `huggingface_hub` or inference stacks. Review source inputs, data rights and selected target repositories before an actual artifact release.

The uploader takes:

- ProkBERT weights/config/custom architecture from `PROKBERT_CKPT`.
- CNN weights/config/metrics/history from `CNN_CKPT`, plus project `pipeline/training/cnn/train_cnn_v2.py`.
- Evo2's committed probe/helper from `pipeline/training/Evo2` and the release `apply_probe.py`; no Evo2 backbone weights are uploaded.
- DH5α's committed `model.npz`, metadata, verification and inference requirements, plus the source predictor/table, exact reference, candidate catalogue and first-ten examples. This route needs no GPU checkpoints.

For all four, from the repository root:

```bash
pip install -U huggingface_hub
hf auth login
source env.sh
export PROKBERT_CKPT="$PROMOTER_RESULTS/T21_xs/prokbert/best_model"
export CNN_CKPT="$PROMOTER_RESULTS/T21_xs/cnn_v2"
python release/huggingface/upload_to_hf.py --dry-run
# Only after reviewing the complete payload and intending a full release:
python release/huggingface/upload_to_hf.py
```

Set both checkpoint paths explicitly if their locations differ; the uploader's fallback values are `/path/to/...` placeholders, not automatic checkpoint discovery. Select a single full package with `--only prokbert`, `--only cnn`, `--only evo2` or `--only dh5alpha`. For example:

```bash
uv run --no-project --with huggingface-hub python \
  release/huggingface/upload_to_hf.py --public --only dh5alpha --dry-run
uv run --no-project --with huggingface-hub python \
  release/huggingface/upload_to_hf.py --public --only dh5alpha
```

`--dry-run` authenticates with `whoami` and checks source inputs; it uploads nothing but is **not an offline inference test**. `HF_OWNER` defaults to `CloBiT`; changing it does not rewrite fixed CloBiT links in cards. New repositories default to private unless `--public` is supplied. **That flag only affects creation of a new repository; updating an existing repository does not change its visibility.** Existing four repositories are already public, so no visibility action is needed for the present documentation update.

## Artifact identity and evidence

| Package | Pinned inspected download revision |
|---|---|
| ProkBERT weights/custom architecture | `815f637e1a77ac25757b0add21a22668090cce31` |
| ProkBERT base tokenizer files / external code | `feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc` / `366d9336b6b89d7c1feddf56d9b01be8065ccc43` |
| CNN weights/feature code | `191fc88fffeb52ecbf4bdb27702a734365545815` |
| Evo2 project probe/helper package | `6e521c874dd61517df9b17a56f2a75847a054b87` |
| DH5α fitted package | `0c5202892f4255eb8eb17493bed8d51e04f8cfc4` |

These are retained artifact/package identities, not proof of every original training dependency. The exact historical Evo2 extraction code/layer remains unresolved; its current helper defaults and saved-feature evaluation do not audit a fresh CUDA extraction. ProkBERT is not limited to 81 bp: the chosen deployment frame tokenizes to 78 tokens, and the checkpoint supports 1,024 positions. Neither point establishes accuracy on a different input frame.

[verification.json](verification.json) records original alignment/historical metrics/saved-feature evidence; [recovery_SHA256SUMS](recovery_SHA256SUMS) is the immutable 159-file original relative-path receipt. Preserve their historical bytes/paths. Earlier access statements in these historical records describe their recording context, **not current repository visibility**. Recorded checks align all 50,190 source rows/encodings; large splits/features/tracks are not part of the model packages. No new scan, fit or assay is implied by this documentation release.

Any historical SRS-training-only statement in immutable verification/recovery records is superseded by the checksum-matched negative-source CSV and current cards: SRS is label 0 across all three splits, while silver/cross-species positive labels are training-only. Those historical records/receipts remain byte-for-byte unchanged; the correction changes no model, split, AP, benchmark metric, licence or provenance identity.

The DH5α package's historical read-back matched all 14 payload files and reproduced ten predictions with NumPy alone; fitted NPZ SHA-256: `153f41e1d051d432a2172b21b6e94f07799fa2229a45d795ae974fc8140130a8`. Its [card](dh5alpha-promoter-strength-svr/README.md) and [verification](dh5alpha-promoter-strength-svr/verification.json) retain the conditional model/evidence boundary.

For NumPy-only inference without a Hub download, follow the [local loading instructions](../../docs/ECOLI_STRENGTH_PREDICTIONS.md#loading-the-fitted-model). The repository copy contains the same retained fit and reference inputs.

### Historical publication identities

The authenticated **2026-10-11 19:32 UTC** card/verification update was read back byte-for-byte at the following revisions. These identify historical documentation/evidence, not the current local card sources or replacement model weights.

| Model | Historical card/verification revision | Preceding card/record revision |
|---|---|---|
| ProkBERT-mini | `ed99c7879f08e7956a12d55fbdd98d4777fbd5b4` | `12d92590a326b95ce2b97394eaa21057b39a4b56` |
| CNN-v2 | `b39daa476d18752677c438e8509cd33cb4ee9fd8` | `2cfa144141d1bead5ecceed2571d52af18a12374` |
| Evo2 probe | `6e521c874dd61517df9b17a56f2a75847a054b87` | `c965e3c7d47b048490bbf800b587a8eb11444698` |

The six card/verification files matched local sources **at that publication**, not the rewritten cards now. Its verification hash is `1b2549072e1bac6e62b19c0954fcfc0a856798fe4a2dd5561d820fbe32b9f0ef`. The preceding ten-file publication record remains at `~/cb-promoter-original-run/verification/huggingface_publication.json`; its verification hash is `6f7bbb5819ce54cec4f9ffa2f8b451ef7a7be2057e98755df7d59ac43cf89e02`. Recovery receipt hash: `da967bd0d64feb9b887dfeec43e3b64f0b66707e8aa81ebb7f0008c8e30eabef`. ProkBERT metrics hash: `63cd7ac371d00dfbaa7ed507e83403693fbb951eb81e9236ee1e4afa0b6bb153`. These historical identities are not evidence that the present corrections have been uploaded.
