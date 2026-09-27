# The models

What each model is, how to load it, why we picked these three, and what we learned from the ones we
dropped.

---

## 1. The deployment ensemble

Three members, combined by **mean rank percentile**. A window only ranks highly if all three rank it
highly. Rank percentile is scale-invariant, so the members combine on raw scores with no calibration
step in between.

| Member | What it is | Role |
|---|---|---|
| **ProkBERT-mini**, fine-tuned | prokaryote-pretrained transformer, ~20 M params, full fine-tune | closest domain match to *Clostridium* |
| **CNN-v2**, from scratch | multi-encoding fusion CNN | cleanest motif detector; no pretraining bias |
| **Evo2-7B probe** | logistic regression on frozen embeddings | best at genome scale; fewest false positives |

All three are published at **[huggingface.co/CloBiT](https://huggingface.co/CloBiT)**.

Every model takes an **81 bp** A/C/G/T window with the transcription start site at 0-based offset 60
(1-based 61), and returns `P(promoter)`. The −10 box then falls around offset 50 and the −35 around
offset 25. Feed a different framing and the scores stop meaning what they meant in training.

---

## 2. Loading them

### From Hugging Face, no training needed

ProkBERT:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "CloBiT/prokbert-mini-clostridium-promoter"
# The tokenizer is unchanged from the base model, so load it from there.
tok = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(REPO, trust_remote_code=True).eval()

enc = tok(["A" * 81], padding="max_length", truncation=True, max_length=81, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)[:, 1]
```

Fusion CNN. The feature pipeline lives in `train_cnn_v2.py`, so import from the downloaded repo
rather than loading the weights alone:

```python
import json, sys, torch
from huggingface_hub import snapshot_download

path = snapshot_download("CloBiT/cnn-fusion-clostridium-promoter")
sys.path.insert(0, path)
from train_cnn_v2 import build_model_from_config, predict_proba

cfg = json.load(open(f"{path}/config.json"))
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(f"{path}/best_model.pt", map_location="cpu"))
print(predict_proba(model, ["A" * 81], device="cpu"))
```

Evo2 probe. Needs a GPU with Evo2 installed, and **scikit-learn 1.9.0** to match the pickle:

```bash
hf download CloBiT/evo2-logreg-clostridium-promoter-probe --local-dir evo2_probe
cd evo2_probe && python apply_probe.py --fasta windows.fna --out scores.csv
```

The probe expects mean-pooled embeddings from Evo2-7B layer `blocks.28.mlp.l3`, 4096-dimensional. A
different layer, pooling or window will not match it.

### From your own checkpoints

Trained checkpoints are too large for git, so this path starts from models you trained in Phase C
([PIPELINE.md](PIPELINE.md)). Phase C writes the deployed set to the scaffold's `results/T21_xs/`
(`env.sh` exports `results/` as `$PROMOTER_RESULTS`):

```
$PROMOTER_RESULTS/T21_xs/prokbert/best_model          # ProkBERT (HF model dir)
$PROMOTER_RESULTS/T21_xs/cnn_v2/best_model.pt         # CNN-v2 (+ config.json alongside)
$PROMOTER_RESULTS/regimeC/evo2/evo2_probe_xs.pkl      # Evo2 probe (needs the Evo2 env; see below)
```

`score_promoters.py` at the repo root is the single hand-off entry point for ProkBERT and CNN-v2:

```bash
source env.sh
conda activate "$ENV_MAIN"
python score_promoters.py --model prokbert --fasta seqs.fa --out prokbert_scores.csv --models-dir "$PROMOTER_RESULTS/T21_xs"
python score_promoters.py --model cnn_v2   --fasta seqs.fa --out cnn_v2_scores.csv --models-dir "$PROMOTER_RESULTS/T21_xs"
```

Without `--models-dir` it reads `$PROMOTER_RESULTS/models_deployed/`, which no script creates. To make
the deployed set the default, symlink it once:

```bash
cd "$PROMOTER_RESULTS" && mkdir -p models_deployed
ln -s ../T21_xs/prokbert models_deployed/prokbert
ln -s ../T21_xs/cnn_v2   models_deployed/cnn_v2
```

The other result folders (`T21_final`, `T21_final_06`, …) are earlier training rounds; see *Which
checkpoint folder is which* in [PIPELINE.md](PIPELINE.md). To score windows with the Evo2 probe (the deployed one is committed as
`model-training/Evo2/evo2_promoter_classifier.pkl`), run `apply_probe.py` in `$ENV_EVO2` — it
imports `evo2_common` from its own folder, which holds it only on the Hub, so point `PYTHONPATH`
at the in-repo copy:

```bash
PYTHONPATH="$PROMOTER_REPO/model-training/Evo2" python \
  huggingface_release/evo2-logreg-clostridium-promoter-probe/apply_probe.py \
  --fasta windows.fna --probe model-training/Evo2/evo2_promoter_classifier.pkl --out scores.csv
```

 `model-training/Evo2/evo2_scan_restricted.py` scans a whole genome with it.

`evaluation-and-calibration/genome_inference.py` already loads every model correctly and exposes a
`scorer(list_of_seqs) -> (probs, logits)` for each. Import it rather than reinventing the loading logic.

Raw scores are uncalibrated and CNN-v2 in particular runs optimistic. That is fine for ranking and
for the ensemble, since rank percentile is calibration-invariant. It matters only if you want a fixed
0.5 threshold, in which case apply the calibrators first.

---

## 3. Why three models

Not "a classifier plus a foundation model" — **three different inductive biases**, chosen so their
errors decorrelate.

- **ProkBERT** brings prokaryote-specific pretraining, the closest domain match.
- **CNN-v2** is trained from scratch with a strong locality prior, good for fixed-width −10/−35
  boxes, and carries no pretraining bias.
- **Evo2** brings broad multi-genome pretraining and far greater capacity, frozen.

Ensemble error falls with the decorrelation of member errors (Krogh & Vedelsby 1995; Ueda & Nakano
1996). CNNs bake in locality and translation-equivariance while transformers carry little spatial
bias (Dosovitskiy et al. 2021), so a CNN plus two differently-pretrained large models is about as
decorrelated as a realistic three-member set gets. Mean rank percentile is the right combiner because
the three emit incomparable score scales.

The honest caveat: diversity only helps if each member is individually decent (Kuncheva & Whitaker
2003). A model that is "different but wrong" hurts, which we confirmed empirically — see §5.

### Foundation model versus classifier

**ProkBERT is a foundation model, not a classifier.** It is pretrained by masked-language modelling
on raw DNA and emits embeddings. You turn it into a classifier by adding a small classification head
and training on labelled examples. A naming trap: `prokbert-mini-promoter` is a *separately*
fine-tuned classifier someone else uploaded. We fine-tune the **base** `prokbert-mini` and use the
`-promoter` variant only as an off-the-shelf baseline.

### Why ProkBERT gets a full fine-tune

ProkBERT-mini is small, so full fine-tuning is cheap and fits our data. The Nucleotide Transformer at
250M is roughly 12× bigger (20.5 M parameters against 250 M), so we used LoRA there (Hu et al. 2021): freeze the base, train under 1% of
parameters as low-rank adapters on the attention query and value projections. It regularises and fits
one GPU. Same end product, adaptation matched to size.

---

## 4. How the models were evaluated

- **Balanced test, primary metric AUPRC**, via `sklearn.metrics.average_precision_score`, reported
  against the prevalence baseline. Positives are rare at genome scale, so AUROC is misleadingly
  optimistic (Davis & Goadrich 2006; Saito & Rehmsmeier 2015). Never `auc(recall, precision)`, which
  inflates the no-skill baseline from prevalence toward 0.5. AUROC, MCC and F1 are secondary. The
  test split is cluster-disjoint: `StratifiedGroupKFold` over sequence-identity clusters, so
  near-duplicates never straddle train and test.
- **Composition-matched and cross-species external holdout** (Regime B), run on the earlier
  checkpoints trained on the plain 0.6 split.
- **Stratified evaluation** by source, species, GC band and sigma factor, run on an earlier round of
  ProkBERT, NT-LoRA and CNN-v1.
- **Genome-scale scanning** of the two validation genomes (Regime C), run on the deployed
  checkpoints. This is the one that predicts deployment.

Cluster-disjoint cross-validation scripts exist in `evaluation-and-calibration/cross-validation/`
but were not run. [RESULTS.md](RESULTS.md) gives the numbers and which checkpoints each one used.

### The frozen-embedding probe, as a diagnostic

Before or independent of fine-tuning, you can measure how much promoter signal a frozen foundation
model already encodes: freeze it, mean-pool the last hidden layer, fit a logistic regression, report
AUPRC. The probe-versus-fine-tune gap tells you what fine-tuning actually bought.

**The control that matters is real promoters versus SRS** — dinucleotide-preserving shuffles that
keep base composition but destroy the −10/−35 order. If the probe separates real from SRS, the signal
is in the *motif*, not in AT-richness. If it only separates promoters from random genomic sequence,
which also differs in composition, you may just be detecting AT content. That is shortcut learning
(Geirhos et al. 2020), and it is exactly the trap this project is built around.

Caveat: SRS should not be the *only* negative, since models can learn the shuffle's own signature.
That is why the training set carries four negative classes. Shuffle-only validation is necessary but
not sufficient: it controls composition away, so a model that keys on "AT-rich means promoter" still
scores well against it. Validate against the realistic four-class negatives too.

`evaluation-and-calibration/embedding_probe.py` runs the probe, the UMAP, and the real-versus-SRS
control for any model id, with no fine-tuning required.

---

## 5. What we dropped, and what it taught us

**Nucleotide Transformer v2 with LoRA.** Second-best balanced AUPRC (test 0.854) and near-worst at
genome scale on both validation genomes. Its in-genus composition-matched AUROC collapsed to 0.440,
which is chance (90% CI 0.36–0.53). Dropped.

**NTv3 with 384 bp context.** The *best* in-distribution AUPRC of any model, weighted val 0.9561 /
test 0.9598, from the `results/ntv3_xs_ctx_classifier` run. Its 384 bp context is built by padding the
81 bp core with real genomic flanks, which can repeat a training row's context verbatim, so the score
needs a leakage check: `dataset-construction/clustering-and-splits/context_leakage_check.py` finds the
high-recurrence rows and `model-training/nucleotide-transformer-v3/current/ntv3_rescore_clean.py`
re-scores without them. **No output of that re-score survives**, so treat the figure above as
un-adjusted for context leakage. Re-run those two scripts before quoting it anywhere load-bearing.

Either way it was the **worst** model at genome scale on both validation genomes, 35.7% and 58.5%
precision at recall 0.1 — and that is the number that decided it. A bigger backbone and mean-pooled
context buy curated-set accuracy and lose the spatial resolution that the genome base rate demands.
Its in-distribution score is also not apples-to-apples with the 81 bp models, since it reads 384 bp.

**CNN-v1.** The weaker, correlated cousin of CNN-v2. Adding it as a fourth member **diluted** the
ensemble on both validation genomes:

| Ensemble | *C. ljungdahlii* @0.1 / @0.5 | *C. autoethanogenum* @0.1 / @0.5 |
|---|---|---|
| 3 members (deployed) | **74.4% / 61.6%** | **87.0% / 76.6%** |
| + CNN-v1 | 71.4% / 58.3% | 86.5% / 74.5% |

That is the ensemble-diversity theory confirmed the hard way. The two runs are paired — same tracks,
same grid, same truth — and live in `results/regimeC/mine3_{clj,cau}/fp_*_ensemble_{3m,4m}_restrict/genome_fp_benchmark.md`
(cluster only, not committed).

### The off-the-shelf baseline, and why it looks near-random

Before training anything we slid a published promoter classifier, `prokbert-mini-promoter`, across
the *E. coli* genome. It recovered 24% of RegulonDB promoters within ±20 bp at 2.06% precision, an
enrichment of only **1.17× over random**. This is not a bug, and it is the bar the project set out to
beat.

The model demonstrably has real signal. Matching 756 known RegulonDB promoters inside that scan
against background gives median score 0.202 versus 0.008, with 36.2% of known promoters scoring ≥0.9
against 18.7% of background: AUC 0.624, Mann-Whitney p = 1.7×10⁻³⁷. Two compounding effects explain
the genome-wide result:

1. **Specificity collapse on real negatives.** It fires on 18.7% of *all* background windows. It was
   fine-tuned against easy balanced negatives, so real AT-rich intergenic and coding sequence is out
   of distribution and it lights up on composition alone.
2. **Base rate.** Promoters are about 0.21% of windows. Even a 95%/95% classifier yields under 4%
   genome-wide precision; reaching 50% precision needs specificity of 0.998.

So it is the right model for the wrong task: built and validated for balanced window classification,
where it works, then applied to an extreme-imbalance localisation problem. Threshold tuning does not
help, because the score distribution is saturated and the sweep is flat from 0.5 to 0.99. Realistic
negatives and calibration are the levers that do.

---

*Sources:* Hu et al. 2021 (LoRA); Krogh & Vedelsby 1995, Ueda & Nakano 1996, Dosovitskiy et al. 2021
(ensembles and inductive bias); Davis & Goadrich 2006, Saito & Rehmsmeier 2015 (AUPRC under
imbalance); Alain & Bengio 2016, Kumar et al. 2022 (probing); Geirhos et al. 2020 (shortcut
learning); Jiang et al. 2008 (uShuffle), Oubounyt et al. 2019 (DeePromoter, composition-matched
negatives); Ligeti et al. 2024 (ProkBERT); Dalla-Torre et al. 2023 (Nucleotide Transformer); Brixi et
al. (Evo2, Arc Institute).
