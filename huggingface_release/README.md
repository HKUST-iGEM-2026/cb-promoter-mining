# Hugging Face release — HKUST iGEM 2026 promoter models

Model cards + the upload script for the three deployment-ensemble models. They are published under the
**[CloBiT](https://huggingface.co/CloBiT)** organisation; each subfolder's `README.md` is the source of
that repo's model card, so edit it here and re-upload rather than editing on the Hub.

The repositories remain **private**. Authenticated access and artifact inventories have been verified.
**Hugging Face artifact recovery is complete (user-reported); the remaining reproducibility checks
are ongoing.** Recovery is not a completed training or inference rerun.

## The three models and their licenses

| Model | What it is | License | Why |
|---|---|---|---|
| [`CloBiT/prokbert-mini-clostridium-promoter`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter) | fine-tune of `neuralbioinfo/prokbert-mini` | **CC-BY-NC-4.0** | inherited — base model is CC-BY-NC (non-commercial) |
| [`CloBiT/cnn-fusion-clostridium-promoter`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter) | multi-encoding fusion CNN, trained from scratch | **CC-BY-4.0** | entirely our own weights; attribution only |
| [`CloBiT/evo2-logreg-clostridium-promoter-probe`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe) | logreg probe on frozen Evo2-7B embeddings (probe only) | **CC-BY-4.0** | probe is ours; Evo2 itself is Apache-2.0 and not re-uploaded |

**Bundling caveat:** the ProkBERT fine-tune is non-commercial. If you ever combine the three into one
ensemble repo that includes the ProkBERT weights, the whole thing becomes non-commercial. Publishing
three separate repos (as here) keeps the CNN and Evo2 probe permissive.

## RegulonDB data licensing

RegulonDB's [official April 2025 policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md)
permits academic/noncommercial sharing and adaptation under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**,
with attribution, license links and disclosure of changes. Commercial data use requires a separate agreement.
All three Hub packages include [`REGULONDB_LICENSE.md`](../REGULONDB_LICENSE.md), recording source and
processing provenance; they contain fitted parameters/helpers, not raw database exports or training tables.
The uploader retains this notice on future publication. CNN/Evo2 helpers also carry the repository's MIT
notice as `SOURCE_CODE_LICENSE`. These data and source-code notices do not change model-weight licenses.


## Files

```
huggingface_release/
├── upload_to_hf.py                              # run this to publish (private unless --public)
├── prokbert-mini-clostridium-promoter/README.md
├── cnn-fusion-clostridium-promoter/README.md
└── evo2-logreg-clostridium-promoter-probe/
    ├── README.md
    └── apply_probe.py                           # sequences -> Evo2 embeddings -> P(promoter)
```

The upload script takes the Evo2 probe files (`evo2_promoter_classifier.pkl`, `evo2_common.py`) from
`model-training/Evo2/`. The committed `.pkl` is the deployed `T15_xs`
probe. The ProkBERT and CNN **checkpoints** are not in git,
because they are too large. Train them (Phase C in [`docs/PIPELINE.md`](../docs/PIPELINE.md)) and
point the script at `results/T21_xs/prokbert/best_model` and `results/T21_xs/cnn_v2`.

## Re-publish or update

`HF_OWNER` defaults to `CloBiT`. You need a **write token** for that organisation, or set `HF_OWNER`
to your own account to publish copies. Give the two checkpoint folders as environment variables
(or edit the paths at the top of the script). Run it wherever the checkpoints and this checkout are;
it only needs internet access and `huggingface_hub`.

```bash
pip install -U huggingface_hub
hf auth login                         # write token for the CloBiT org
export PROKBERT_CKPT=/path/to/T21_xs/prokbert/best_model CNN_CKPT=/path/to/T21_xs/cnn_v2

python upload_to_hf.py --dry-run      # see exactly what would upload
python upload_to_hf.py --only evo2    # evo2 needs no checkpoint paths — its files are in-repo
python upload_to_hf.py --public       # push all three
```

To update a model card alone, without re-uploading weights (the cards contain no `<HF_OWNER>`
placeholder, so they upload as they are):

```bash
hf upload CloBiT/prokbert-mini-clostridium-promoter prokbert-mini-clostridium-promoter/README.md README.md
```

`upload_to_hf.py` creates repos **private** unless you pass `--public`, and `exist_ok=True` means
re-running it updates the existing repos in place rather than failing. `--public` only applies to a
repo the script creates: re-uploading never changes an existing repo's visibility. Change that in the
repo's Settings on the Hub, or with `HfApi().update_repo_settings(repo_id, private=False)`.
Uploading a model card alone is the common case: the script always re-uploads `README.md` with
`<HF_OWNER>` resolved.

## What is published, as of 2026-09-27

All three repos hold the **deployed** `T21_xs` artefacts and the current cards. They are **private**:
only CloBiT members can open the links above. To publish one, flip it in the repo's Settings on the
Hub, or:

```bash
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('CloBiT/<repo>', private=False)"
```

| Repo | Weights / artefact | Also shipped |
|---|---|---|
| `prokbert-mini-clostridium-promoter` | `model.safetensors` (20,490,627 params) | `config.json`, `models.py`, `training_args.bin` |
| `cnn-fusion-clostridium-promoter` | `best_model.pt` | `config.json`, `metrics.json`, `history.json`, `train_cnn_v2.py` |
| `evo2-logreg-clostridium-promoter-probe` | `evo2_promoter_classifier.pkl` (47,137 rows seen) | `evo2_common.py`, `apply_probe.py` |

ProkBERT ships **no tokenizer**: its card loads the unchanged one from `neuralbioinfo/prokbert-mini`.

## Before you push — checklist

- [ ] Write access to the `CloBiT` org; checkpoint paths filled in (or use `--only` for the ready ones)
- [ ] ProkBERT repo stays **CC-BY-NC-4.0** and cites the ProkBERT paper + base model
- [ ] Evo2 repo does **not** contain Evo2 weights (probe + helper only)
- [ ] `model-training/Evo2/evo2_promoter_classifier.pkl` is the `T15_xs` probe (its scaler reports 47,137 rows seen)
- [ ] Each card's data table, citations and licenses still match `docs/PIPELINE.md`
- [ ] scikit-learn **1.9.0** noted for the probe (pickle version match)

### Attribution status

Every data source is now cited: Hocq 2022, Al-Bassam 2018, Wiegand 2013, Xu 2015, Lemgruber 2019
and RegulonDB by DOI, DBTBS by URL (it has no DOI). The master list is the header of
`data-acquisition-and-curation/additional-published-tss/prep_new_tss.py`; keep it and the three cards
in step. The per-source row counts in the cards come from the deployed split grouped by `source`, not
from the Phase A outputs, which are larger because `cdhit_splits.py` drops exact duplicate sequences.
