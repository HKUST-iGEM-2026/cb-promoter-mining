# Hugging Face release — HKUST iGEM 2026 promoter models

Model cards + an upload script for publishing the three deployment-ensemble models to the
Hugging Face Hub. Each subfolder's `README.md` becomes that repo's model card.

## The three models and their licenses

| Model | What it is | License | Why |
|---|---|---|---|
| `prokbert-mini-clostridium-promoter` | fine-tune of `neuralbioinfo/prokbert-mini` | **CC-BY-NC-4.0** | inherited — base model is CC-BY-NC (non-commercial) |
| `cnn-fusion-clostridium-promoter` | multi-encoding fusion CNN, trained from scratch | **CC-BY-4.0** | entirely our own weights; attribution only |
| `evo2-logreg-clostridium-promoter-probe` | logreg probe on frozen Evo2-7B embeddings (probe only) | **CC-BY-4.0** | probe is ours; Evo2 itself is Apache-2.0 and not re-uploaded |

**Bundling caveat:** the ProkBERT fine-tune is non-commercial. If you ever combine the three into one
ensemble repo that includes the ProkBERT weights, the whole thing becomes non-commercial. Publishing
three separate repos (as here) keeps the CNN and Evo2 probe permissive.

## Files

```
huggingface_release/
├── upload_to_hf.py                              # run this to publish (starts PRIVATE)
├── prokbert-mini-clostridium-promoter/README.md
├── cnn-fusion-clostridium-promoter/README.md
└── evo2-logreg-clostridium-promoter-probe/
    ├── README.md
    └── apply_probe.py                           # sequences -> Evo2 embeddings -> P(promoter)
```

The Evo2 probe's data files (`evo2_promoter_classifier.pkl`, `evo2_common.py`) are already in the repo
under `model-training/Evo2/`; the upload script pulls them from there. Only the ProkBERT and CNN
**checkpoints** live on HPC4 — you point the script at them.

## Publish

```bash
pip install huggingface_hub
huggingface-cli login                 # write token

# edit the top of upload_to_hf.py: HF_OWNER + the two HPC4 checkpoint paths
python upload_to_hf.py --dry-run      # see exactly what would upload
python upload_to_hf.py --only evo2    # evo2 needs no HPC4 paths — its files are in-repo
python upload_to_hf.py                # push all three (private)
```

Repos are created **private** by default. Review them on the Hub, then make them public from the HF UI
or re-run with `--public`. iGEM values open science, so publishing (with model cards + the AT-rich
false-positive caveat) is encouraged — this is also project task **PT2**.

## Before you push — checklist

- [ ] `HF_OWNER` set; HPC4 paths filled in (or use `--only` for the ready ones)
- [ ] ProkBERT repo stays **CC-BY-NC-4.0** and cites the ProkBERT paper + base model
- [ ] Evo2 repo does **not** contain Evo2 weights (probe + helper only)
- [ ] Training-data attribution present (RegulonDB CC-BY, DBTBS, TSS papers)
- [ ] scikit-learn **1.9.0** noted for the probe (pickle version match)
