# CLAUDE.md — validating NT (T19): the plan + the reasoning (for Ishi)

Answers Ishi's two questions: (1) how to validate NT if it's "just a foundation model that returns
embeddings," and (2) why we picked these models — with concrete reasoning for the engineering writeup.

## ⭐ The PT5/PT6-style test for NT (Oak's actual ask: "test NT like you tested prokbert")
**Why you can't do it identically:** PT5/PT6 ran `prokbert-mini-promoter` — an **already-fine-tuned
promoter classifier** — zero-shot, sliding-window across MG1655. **NT has no equivalent**: InstaDeep
released only the base foundation model + fine-tuning notebooks, so there is **no off-the-shelf NT promoter
head** (and certainly no bacterial one). Base NT returns embeddings, not promoter scores — so you must add a
small head before you can scan a genome.

**The faithful equivalent (two cheap steps — both scripts are in this folder):**
1. **Make a head:** `python nt_linear_probe.py --data-dir results/T16/nt --save-head nt_probe_head.joblib`
   — frozen NT embeddings + a logistic-regression head trained on the T16 promoter/negative splits. (The
   printed test AUPRC is *also* Tier-1 on its own: does NT's *frozen* embedding even carry promoter signal?)
2. **Scan the genome like PT5:** `python nt_sliding_window.py --fasta MG1655.fna --head nt_probe_head.joblib`
   — a **drop-in twin of `pre-tasks/PT5/sliding_window_inference.py`** (same 81 bp window, step 5, both
   strands, same output columns); the only change is the scorer = frozen-NT embedding → that head. Then run
   your **PT6 validation unchanged** (recall / precision / enrichment / bp-accuracy vs RegulonDB) and put
   NT's numbers next to the `prokbert-mini-promoter` PT5/PT6 numbers.

**Three honest caveats:**
- It's **not zero-shot** — you train a tiny head first (unavoidable; no NT promoter head is published). It's
  one logistic regression, so it's still a "minimal-effort" baseline that's fair to compare.
- **Cost:** NT-250M is far heavier than prokbert-mini; a full MG1655 scan (~1.9M windows × both strands) is
  many GPU-hours. Use a coarser `--step` (10–20) for the first pass, or scan a region.
- **Expect it to look weak genome-wide too** — the base-rate / AT-trap problem PT6 already found for
  prokbert. That's the informative result and the bar the T19 LoRA fine-tune must clear.

## 1. In our pipeline NT is NOT a frozen embedding model — T19 fine-tunes it into a classifier
The *base* NT returns embeddings, true. But **T19 wraps it in LoRA + a 2-class head**
(`AutoModelForSequenceClassification` + a LoRA adapter), so after training it **outputs promoter
probabilities** like any classifier. So NT's *primary* validation is identical to ProkBERT (T18) and the
CNN (T20): **AUPRC / AUROC / MCC on the held-out, cluster-disjoint test** (T22 / PT3), cluster-CV (T23),
and the cross-species external holdout (T24). The ensemble (T29) combines the three classifiers' scores —
no embedding-mapping needed for the headline metric.

This mirrors the **NT authors' own recipe**: they adapt NT with a PEFT method (IA3) + a task head, and
report fine-tuning beats frozen-embedding probing by ~20% on average (promoter task: ~0.966 probing →
~0.974 fine-tuned). We use LoRA instead of IA3 — same idea, same family of method.

## 2. Ishi's embedding-probe idea is good — as a BASELINE/ablation, not the main metric
"Map embeddings of known promoters vs non-promoters" = a **linear probe on frozen NT embeddings**, which is
literally the NT paper's probing methodology. Keep it — but as a **baseline tier**, because it answers a
different, useful question: *does NT's pretraining already encode promoter features before we train
anything?* (The paper says yes — embeddings capture promoters/enhancers with no supervision — but
fine-tuning still adds ~1–2 points.)
- **Run it:** frozen NT → mean-pool the last hidden state → logistic regression → AUPRC, on the **same T16
  NT splits** so it's apples-to-apples with the LoRA fine-tune. Script: **`nt_linear_probe.py`** (this folder).
- **The story it gives:** `frozen-probe AUPRC = X`, `LoRA AUPRC = X + Δ` → **Δ is exactly what fine-tuning
  buys** — a concrete, defensible point for the writeup.
- NT paper found **middle layers** probe best, so optionally sweep layers (`output_hidden_states=True`).

## 3. ⚠️ The negative set matters MORE than probe-vs-fine-tune (the non-obvious bit)
Ishi's "shuffle E. coli promoters into non-promoters and validate by mapping" = our **SRS** negative
(dinucleotide-preserving shuffle, T14). Good instinct — but **shuffle-only validation is necessary, not
sufficient**, and here's the trap:
- A shuffle null (real promoter vs same-composition shuffle) tests *"does the model use the −10/−35 motif,
  not just base composition?"* — useful, and it's also our **empirical-FDR null (T27)**.
- BUT our real failure mode (CLAUDE.md §1) is the **AT-trap**: *Clostridium* is AT-rich and promoter motifs
  are AT-rich, so the genuine risk is **false positives on AT-rich non-promoter genomic background**.
  Shuffle-only validation **hides** that risk (it controls composition away, so a model that just keys on
  "AT-rich = promoter" still scores well).
- **So validate against the realistic 4-class negatives** (CDS, intergenic, downstream, + SRS) from real
  *Clostridium* — that's what exposes the deployment failure mode. Use shuffle as the *motif* control and
  the FDR null, not as the whole evaluation.

## 4. Why ProkBERT + NT (+ CNN) — rationale for the narrative
Small framing fix first: **both ProkBERT and NT are foundation models we fine-tune** (both masked-LM
pretrained). The off-the-shelf `prokbert-mini-promoter` is a classifier, but **T18 fine-tunes the *base*
`prokbert-mini`**. So the design isn't "a classifier + a foundation model" — it's **two foundation models
chosen to be deliberately different, plus a from-scratch CNN control**:

| member | pretraining | size | adaptation | role in the ensemble |
|---|---|---|---|---|
| ProkBERT-mini (T18) | **prokaryote-specific** | small | full fine-tune | strong bacterial-promoter prior; cheap |
| NT v2 250M multispecies (T19) | **broad — 850 genomes** | large (250M) | LoRA (PEFT) | capacity + general features; tests whether broad pretraining transfers to AT-rich *Clostridium* |
| 1D CNN (T20) | **none (from scratch)** | small | trained | the **control** — isolates how much pretraining actually helps |

Ensemble logic (CLAUDE.md §1): models with **different pretraining + architectures make decorrelated
errors**, so their **agreement** (mean rank-percentile, T29) is a meaningful confidence signal and the
ensemble suppresses any single model's false positives. The choice also gives a **built-in ablation** —
domain-specific (ProkBERT) vs general-large (NT) vs no-pretraining (CNN) — a clean, publishable comparison.

## 5. On "download a trained head"
- **ProkBERT:** `neuralbioinfo/prokbert-mini-promoter` exists → we already use it as the **off-the-shelf
  baseline** (PT5 / PT6). Good as a "zero-effort" comparison tier.
- **NT:** there is **no off-the-shelf bacterial-promoter head** — InstaDeep's promoter benchmark is *human*
  TATA/non-TATA. So NT must be **trained on our data** = exactly T19 LoRA. Ishi's "we'll prob have to train
  it" is correct.

## Recommended validation tiers (the engineering-cycle story)
1. **Off-the-shelf baseline** — `prokbert-mini-promoter` on our test (already have it). *"What you get for free."*
2. **Frozen-NT linear probe** — `nt_linear_probe.py`. *"Is the signal already in the pretrained embeddings?"*
3. **LoRA fine-tune (T19)** — the real metric; Δ over the probe = the value of fine-tuning.
4. **Ensemble (T29)** — agreement of 3 decorrelated members = the final confidence.

All on the **same held-out, cluster-disjoint test**, scored against **realistic negatives** (not
shuffle-only), with shuffle reserved as the motif control + the FDR null (T27).

## Sources
- Nucleotide Transformer, *Nature Methods* 2024 (probing vs fine-tuning; promoter task): https://www.nature.com/articles/s41592-024-02523-z
- NT model card (two uses: embeddings or fine-tune with a task head): https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
- NT downstream-tasks datasets (the promoter benchmark is *human* TATA/non-TATA): https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
