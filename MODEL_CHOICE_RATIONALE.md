# Model choice & validation rationale (for the iGEM engineering-cycle narrative)

Concrete, cited reasoning for three questions the team keeps hitting: why three models, why two of them are
"language models" adapted *differently*, and how each one — including Nucleotide Transformer (NT) — is
validated even though it "only returns embeddings."

## 1. Foundation model vs. classifier — the key clarification
**Both ProkBERT and NT are FOUNDATION models, not classifiers.** A foundation model is pretrained by
**masked-language-modelling** on raw DNA (predict masked bases); it learns general "DNA grammar" and emits
**embeddings** (a vector per token). Neither is born a promoter detector.

You turn a foundation model into a classifier by **adding a small classification head** (a linear layer:
pooled embedding → 2 logits = promoter / not) and training it on labelled examples:
- **ProkBERT (T18):** base `neuralbioinfo/prokbert-mini` (foundation) + fresh 2-class head, **full fine-tune**.
- **NT (T19):** base `…nucleotide-transformer-v2-250m-multi-species` (foundation) + 2-class head, **LoRA fine-tune**.
- **CNN (T20):** no pretraining — trained from scratch.

> **Naming trap that causes the confusion:** `prokbert-mini-promoter` is a *separately fine-tuned classifier*
> someone uploaded — NOT the base model. We fine-tune the **base** (`prokbert-mini`) for the ensemble and use
> the `-promoter` variant only as an off-the-shelf **baseline** (PT5/PT6). NT has **no** bacterial-promoter
> variant to download (its promoter benchmark is *human* TATA/non-TATA), so NT's head is trained by us — that's T19.

## 2. Why adapt them differently (full fine-tune vs LoRA)
- **ProkBERT-mini is small** → full fine-tuning is cheap and fits our data.
- **NT-250M is ~25× bigger** → full fine-tuning on our *small verified-positive set* would overfit and cost far
  more GPU. **LoRA** (Hu et al. 2021, arXiv:2106.09685) freezes the base and trains <1% of params (low-rank
  adapters on the attention query/value projections); it **regularises** (resists overfitting + catastrophic
  forgetting — Biderman et al. 2024) and fits one GPU. Same end product (a promoter classifier), adaptation
  matched to size.

## 3. Why THREE models — the ensemble argument (the real narrative)
Not "classifier + foundation model" — **three different inductive biases**, combined by **mean rank-percentile**,
chosen so their errors *decorrelate*:
- **ProkBERT** — *prokaryote-specific* pretraining (bacteria/archaea/viruses) → closest domain match to *Clostridium*.
- **NT v2 250M multispecies** — *broad multispecies* pretraining → different, more general priors + larger capacity.
- **CNN** — from scratch, strong **locality** prior → good at fixed-width −10/−35 boxes; no pretraining bias (the control).

Ensemble error falls with the **decorrelation** of member errors (Krogh & Vedelsby 1995, ambiguity decomposition;
bias–variance–covariance, Ueda & Nakano 1996). Different architectures + pretraining = different mistakes →
**agreement = confidence**. CNNs bake in locality/translation-equivariance while transformers carry little spatial
bias (Dosovitskiy et al. 2021), so a CNN + two differently-pretrained transformers is about as decorrelated as a
realistic 3-member set gets. **Mean rank-percentile** is the right combiner because the three emit incomparable
score scales — ranks are scale/calibration-invariant. *Honest caveat (not a yes-man point):* diversity only helps
if each member is individually decent (Kuncheva & Whitaker 2003) — a model that's "different but wrong" doesn't
help, so each must clear the baselines in §4.

## 4. How every model — including NT — is validated (one identical protocol)
Once each has a head it outputs a **promoter probability per window**, so all three go through the **same** held-out
evaluation — that's what makes them comparable:
- **Primary metric: AUPRC** (`sklearn.metrics.average_precision_score`) reported against the **prevalence
  baseline** — positives are rare, so AUROC is misleadingly optimistic (Davis & Goadrich 2006; Saito & Rehmsmeier
  2015). Use `average_precision_score`, never `auc(recall, precision)`.
- **AUROC / MCC / F1** secondary (T22).
- **Cluster-disjoint CV** (T23, `StratifiedGroupKFold` on CD-HIT clusters) — near-duplicate sequences must not leak
  train↔test or scores inflate.
- **Cross-species external holdout** (T24) + sigma-factor-stratified eval (T25).

So **"NT only returns embeddings" is a non-issue**: `AutoModelForSequenceClassification` adds the pooling + head, LoRA
keeps that head trainable, and NT is scored exactly like ProkBERT and the CNN.

## 5. The frozen-embedding linear probe (Ishi's idea — a great DIAGNOSTIC; run it as "Engineering Cycle 0")
Before/independent of fine-tuning, measure **how much promoter signal the frozen foundation model already encodes**:
freeze the model, **mean-pool** the last hidden layer to one vector per sequence, train a **logistic regression** on
those frozen features, report **AUPRC**. (Alain & Bengio 2016 = "linear probe"; the probe-vs-fine-tune *gap*, Kumar
et al. 2022, tells you how much fine-tuning adds on top.)
- **Visualise** with UMAP/PCA — but only to *explore*. A 2-D gap can be an artifact (t-SNE/UMAP caveats, Wattenberg
  et al. 2016); the quantitative claim is the held-out probe AUPRC, not the picture.
- **The control that matters (the AT-trap test):** probe **real promoters vs SRS** (dinucleotide-preserving shuffles
  — keep base/AT composition, destroy the −10/−35 order). If the probe separates real from SRS, the signal is in the
  **motif**, not just AT-richness. If it only separates promoters from random genome (which also differ in
  composition), you may be detecting AT-content = the shortcut (Geirhos et al. 2020, "shortcut learning"). This is
  exactly Ishi's "shuffle E. coli promoters into non-promoters" idea — and it's the repo's **SRS** negative (T14).
  *Caveat:* SRS shouldn't be the *only* negative (models can learn the shuffle's own signature — PLOS ONE 2020
  e0237412), which is why T14 has four negative classes.
- **Tooling:** `evaluation-and-calibration/embedding_probe.py` runs the probe + UMAP + the real-vs-SRS control for
  any model id (prokbert-mini, NT, …) on the T15 data, no fine-tuning required.

## 6. What you can / can't download (re: "download a classifier head")
- **ProkBERT — yes:** `neuralbioinfo/prokbert-mini-promoter` (RegulonDB-trained) exists; it's our PT5/PT6 baseline.
  Fine for a quick sliding-window baseline, NOT a substitute for the fine-tuned ensemble member.
- **NT — no:** no bacterial-promoter classifier to download (its promoter benchmark is human). For NT you train the
  head (T19). You *can* still run the **frozen-embedding probe** on NT today, training nothing.

---
*Sources:* Hu et al. 2021 (LoRA, arXiv:2106.09685); Krogh & Vedelsby 1995, Ueda & Nakano 1996, Dosovitskiy et al.
2021 (ensembles / inductive bias); Davis & Goadrich 2006, Saito & Rehmsmeier 2015 (AUPRC under imbalance); Alain &
Bengio 2016, Kumar et al. 2022, Hewitt & Liang 2019 (probing); Geirhos et al. 2020 (shortcut learning); uShuffle
(Jiang et al. 2008) & DeePromoter (Oubounyt et al. 2019) (composition-matched negatives); ProkBERT (Ligeti et al.,
Front. Microbiol. 2024); Nucleotide Transformer (Dalla-Torre et al., InstaDeep 2023).
