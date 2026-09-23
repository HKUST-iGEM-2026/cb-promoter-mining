# Contribution

> **Standard URL:** `2026.igem.wiki/hkust/contribution`
> **Medal relevance:** Bronze/Silver (Contribution to future teams)

We built our project to be **reused**. Everything a future team needs to mine promoters in their own
AT-rich, non-model organism is documented and, where licensing allows, openly published.

## 1. Three open promoter models on Hugging Face

We publish all three ensemble members as reusable models, each with a full model card (intended use,
training data, metrics, and limitations):

| Model | What it is | License |
|---|---|---|
| `prokbert-mini-clostridium-promoter` | Fine-tuned ProkBERT-mini promoter classifier | CC-BY-NC-4.0 *(inherited from the base model — non-commercial)* |
| `cnn-fusion-clostridium-promoter` | Multi-encoding fusion CNN, trained from scratch | CC-BY-4.0 |
| `evo2-logreg-clostridium-promoter-probe` | Logistic-regression probe on frozen Evo2-7B embeddings | CC-BY-4.0 |

We were careful with licensing: the ProkBERT fine-tune inherits its base model's **non-commercial**
terms, while the from-scratch CNN and the Evo2 probe are permissive. The Evo2 probe ships **only our
small classifier**, not the multi-gigabyte Evo2 weights (Evo2 is Apache-2.0 and installed separately).

## 2. A documented, reusable pipeline

The full five-phase pipeline — data acquisition → dataset construction → training → evaluation → mining
— is available as documented code with pinned software environments and a workflow definition. A future
team can point it at a new genome and a new set of known promoters and re-run it. Key reusable pieces:

- The **realistic-negative recipe** (SRS + four negative classes) that beats the AT-trap.
- **Cluster-aware splitting** and the **three-regime evaluation** (balanced / composition-matched /
  genome-scale) that keep results honest.
- The **rank-percentile ensemble**, **calibration**, and **empirical-FDR** steps.
- The **mine → peak-call → motif-annotate → candidate-table** workflow.

## 3. Curated datasets

We harmonized promoter/TSS data from several sources (*E. coli* RegulonDB, *B. subtilis* DBTBS,
*B. licheniformis*, *Ruminiclostridium*, and in-genus *Clostridium* TSS) into a single, consistently
windowed, leakage-controlled training set — a resource for any bacterial promoter-prediction effort.
*(Please cite the original data sources; see each model card.)*

## 4. The *C. butyricum* candidate catalogue

The mined candidate lists (912 high-confidence / 306 with a −10 box / 100 canonical) are themselves a
contribution: a starting genetic-parts catalogue for anyone wanting to engineer *C. butyricum*. See
[Results](results.md).

## 5. Methodological lessons (documented for the community)

Two findings we make explicit because they generalize:

1. **Balanced-test accuracy does not predict genome-scale performance.** Always evaluate promoter
   predictors on held-out whole genomes, and select deployment models on that metric.
2. **Realistic, promoter-derived negatives (SRS) are the single biggest lever** against false positives
   in AT-rich genomes.

## For the next team

If you are picking this up: start from the [Model](model.md) page for the method, the model cards on
Hugging Face for the trained weights, and the [Engineering](engineering.md) page for the mistakes we
already made so you don't have to. The clearest next improvement is **iterative hard-negative mining**
(feeding the model's confident false positives back as new negatives), which we scoped but leave as the
next refinement.
