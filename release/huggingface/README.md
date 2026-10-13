# Hugging Face packages

All four component packages exist in **private CloBiT repositories**: ProkBERT, CNN-v2, the frozen-Evo2 probe and the separate DH5α promoter-strength SVR. Downloads require an account/token authorised for the specific repository; signed-out or unauthorised links can return 404. The Hub profile is not a public prediction service or a packaged three-model ensemble. The DH5α fitted NPZ and predictor are also included in this source tree. Local card edits are not automatically synced to the Hub; use the publication commands below yourself.

## Licences

| Model | What it is | Declared artifact license | Upstream context |
|---|---|---|---|
| [`CloBiT/prokbert-mini-clostridium-promoter`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter) | fine-tune of `neuralbioinfo/prokbert-mini` | **CC-BY-NC-4.0** | inherited — base model is CC-BY-NC (non-commercial) |
| [`CloBiT/cnn-fusion-clostridium-promoter`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter) | multi-encoding fusion CNN, trained from scratch | **CC-BY-4.0** | team's artifact declaration; upstream data terms still apply |
| [`CloBiT/evo2-logreg-clostridium-promoter-probe`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe) | logreg probe on frozen Evo2-7B embeddings (probe only) | **CC-BY-4.0** | artifact declaration; [Evo2 base card](https://huggingface.co/arcinstitute/evo2_7b) declares Apache-2.0 |
| [`CloBiT/dh5alpha-promoter-strength-svr`](https://huggingface.co/CloBiT/dh5alpha-promoter-strength-svr) | reproduced fitted RBF SVR, numeric NPZ/JSON and NumPy inference | **MIT** | Wang/Deep_promoter procedure and small published table; upstream MIT notice retained |


Project helper source is MIT, separately bundled as SOURCE_CODE_LICENSE in CNN/Evo2 packages. ProkBERT external tokenizer/model code declares CC-BY-NC-ND-4.0; its recovered models.py matches revision 366d9336b6b89d7c1feddf56d9b01be8065ccc43. Artifact declarations do not waive data/backbone terms.

The three genome-mining packages carry the [RegulonDB notice](../../REGULONDB_LICENSE.md): academic/noncommercial CC-BY-NC-4.0 sharing/adaptation requires attribution, a licence link and change disclosure; commercial data use needs an agreement. The DH5α package does not use RegulonDB exports. It instead bundles Wang's 94-row source table, the exact reference and prediction examples, with the retained upstream notice and the project's MIT licence.

## Contents and recovery

Each subdirectory's README.md is its model-card source. `upload_to_hf.py` takes Evo2 probe/helper files from `pipeline/training/Evo2`, ProkBERT from `PROKBERT_CKPT` and CNN-v2 from `CNN_CKPT`. Their default local checkpoint root is `workflow/results/T21_xs`; these large weights are ignored by git. The DH5α entry takes the committed `model.npz`, `model_metadata.json`, verification and NumPy requirement from its release directory, plus the predictor, source table, reference and first-ten examples.

[verification.json](verification.json) records data alignment, historical CNN/ProkBERT metrics and saved-feature Evo2 test evaluation. [recovery_SHA256SUMS](recovery_SHA256SUMS) is the immutable original 159-file relative-path receipt; do not rewrite its historical paths after moving this checkout. Large splits/features/tracks remain under ~/cb-promoter-original-run/, not in these packages.

Recorded alignment covers all 50,190 source rows/encodings. Evo2 test AP/AUROC is 0.70258010/0.89534660 on recovered saved features. Its calculation supplied weights, but all test weights are 1, so weighted/unweighted test metrics coincide; this does not audit new GPU DNA inference. Calibration is not needed for deployed raw ranks. [Models](../../docs/MODELS.md) gives loading, local inventories and scientific limits.

### DH5α fitted-model publication

The separate [DH5α package](https://huggingface.co/CloBiT/dh5alpha-promoter-strength-svr/tree/0c5202892f4255eb8eb17493bed8d51e04f8cfc4) was published privately and downloaded at revision **`0c5202892f4255eb8eb17493bed8d51e04f8cfc4`**. All **14 payload files** matched their local sources byte-for-byte, and the downloaded predictor reproduced all ten candidates with NumPy alone. Fitted weights SHA-256: `153f41e1d051d432a2172b21b6e94f07799fa2229a45d795ae974fc8140130a8`.

The [card](dh5alpha-promoter-strength-svr/README.md) and [verification](dh5alpha-promoter-strength-svr/verification.json) describe the 88-record reproduction, reference normalization and uncertainty. This publication did not modify the existing three repositories or their visibility. Numerical reproduction is not a new assay or an independent accuracy estimate for the candidates.

Authenticated read-back still finds this revision and all 14 payload files. If the link appears missing, use `hf auth whoami` to check which account is active and confirm its access; uploading duplicate weights does not fix an unauthorised account. The repositories have not been made public. For NumPy-only inference without Hub access, follow the [local loading instructions](../../docs/ECOLI_STRENGTH_PREDICTIONS.md#loading-the-fitted-model).

## Publish

Run from the repository root with write-authorised credentials. Root uv sync does not install huggingface_hub or the model stack.

```bash
pip install -U huggingface_hub
hf auth login
source env.sh
export PROKBERT_CKPT="$PROMOTER_RESULTS/T21_xs/prokbert/best_model"
export CNN_CKPT="$PROMOTER_RESULTS/T21_xs/cnn_v2"
python release/huggingface/upload_to_hf.py --dry-run
python release/huggingface/upload_to_hf.py --only evo2
```

For the fitted DH5α release alone, no GPU checkpoints are needed:

```bash
uv run --no-project --with huggingface-hub python \
  release/huggingface/upload_to_hf.py --only dh5alpha --dry-run
uv run --no-project --with huggingface-hub python \
  release/huggingface/upload_to_hf.py --only dh5alpha
```

`--dry-run` authenticates via whoami and checks inputs; it is not an offline inference test. HF_OWNER defaults to CloBiT. New repositories are private unless `--public`; updating an existing repository does not change visibility. Verify actual package links/terms before publishing; changing HF_OWNER does not rewrite fixed CloBiT URLs in cards.

For a card-only update without weights:

```bash
hf upload CloBiT/prokbert-mini-clostridium-promoter \
  release/huggingface/prokbert-mini-clostridium-promoter/README.md README.md
```

## Historical publication

The authenticated **2026-10-11 19:32 UTC** card/verification update was read back byte-for-byte at:

| Model | Historical card/verification revision |
|---|---|
| ProkBERT-mini | [`ed99c7879f08e7956a12d55fbdd98d4777fbd5b4`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter/tree/ed99c7879f08e7956a12d55fbdd98d4777fbd5b4) |
| CNN-v2 | [`b39daa476d18752677c438e8509cd33cb4ee9fd8`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter/tree/b39daa476d18752677c438e8509cd33cb4ee9fd8) |
| Evo2 probe | [`6e521c874dd61517df9b17a56f2a75847a054b87`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe/tree/6e521c874dd61517df9b17a56f2a75847a054b87) |


Those six files matched local sources **at publication**, not the rewritten cards now. Weights/private visibility were unchanged. Its verification hash is 1b2549072e1bac6e62b19c0954fcfc0a856798fe4a2dd5561d820fbe32b9f0ef.

The preceding ten-file publication remains in ~/cb-promoter-original-run/verification/huggingface_publication.json:

| Model | Verified card/record revision |
|---|---|
| ProkBERT-mini | [`12d92590a326b95ce2b97394eaa21057b39a4b56`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter/tree/12d92590a326b95ce2b97394eaa21057b39a4b56) |
| CNN-v2 | [`2cfa144141d1bead5ecceed2571d52af18a12374`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter/tree/2cfa144141d1bead5ecceed2571d52af18a12374) |
| Evo2 probe | [`c965e3c7d47b048490bbf800b587a8eb11444698`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe/tree/c965e3c7d47b048490bbf800b587a8eb11444698) |


Earlier verification hash: 6f7bbb5819ce54cec4f9ffa2f8b451ef7a7be2057e98755df7d59ac43cf89e02. Recovery receipt hash: da967bd0d64feb9b887dfeec43e3b64f0b66707e8aa81ebb7f0008c8e30eabef. ProkBERT metrics hash: 63cd7ac371d00dfbaa7ed507e83403693fbb951eb81e9236ee1e4afa0b6bb153. These are publication evidence, not training-label repair or a full data archive.
