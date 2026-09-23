# Promoter Mining in *Clostridium butyricum* — Progress Presentation

**HKUST iGEM 2026 · Dry Lab · Promoter Mining**
Audience: supervisors / PI · Style: results-forward, rigorous, honest caveats
Suggested length: ~14 slides, 15–20 min. Each slide below has **title · on-slide bullets · speaker notes.**

---

## Slide 1 — Title

**Mining the *Clostridium butyricum* Genome for Promoters with a Deep-Learning Ensemble**

- HKUST iGEM 2026 — Dry Lab (Promoter Mining module)
- Target: *C. butyricum* DSM 10702 · 4-model ensemble · end-to-end computational pipeline

*Speaker notes:* One line to set the frame — "Our goal was to hand the wet lab a shortlist of promoters in our chassis genome, found computationally, with evidence that they're real. I'll focus on results and how we validated them, not the plumbing."

---

## Slide 2 — The Goal, and Why It Matters

- **Promoters** are the DNA switches that turn genes on; a characterized set is foundational for any engineering in a new chassis.
- Finding them experimentally is slow; *C. butyricum* has almost none catalogued.
- **Deliverable:** a ranked, evidence-backed shortlist of candidate constitutive promoters in the target genome, ready for wet-lab testing.

*Speaker notes:* Emphasize the payoff — a prioritized shortlist means the wet lab tests tens of candidates, not thousands. This is the concrete output the whole project drives toward.

---

## Slide 3 — The Core Challenge (this shapes every result)

- *Clostridium* genomes are **AT-rich** (GC ~28–32%).
- The promoter signals themselves (−10 / −35 boxes) are **also AT-rich**.
- So a naïve model learns *"AT-rich = promoter"* → floods the wet lab with **false positives**.
- Everything in our design is built to beat this one trap.

*Speaker notes:* This is the single most important slide for understanding our choices. When the target and the noise look the same by composition, ordinary accuracy metrics lie. Foreshadow the three-regime evaluation here.

---

## Slide 4 — Approach at a Glance

- An **ensemble of 4 DNA models** scores every position; we combine them by **mean rank-percentile** (agreement = confidence).
- Scan the target's **promoter-plausible regions** → call **peaks** → **confirm with motif analysis**.
- Models: ProkBERT-mini, Nucleotide Transformer (LoRA), a from-scratch CNN, and a multi-encoding CNN-v2.

*Speaker notes:* Keep this light — the point is the *shape* of the pipeline (score → peak → confirm), not the internals. Rank-percentile matters because it's calibration-invariant, so combining raw model scores is valid.

---

## Slide 5 — How We Know the Results Are Real: Three Evaluation Regimes

- **Regime A** — balanced test set (standard ML accuracy, AUPRC).
- **Regime B** — vs **composition-matched** decoys (shuffled sequences with identical AT-content → tests real motif learning).
- **Regime C** — **genome-scale** false-positive count on *held-out genomes with known promoters* (the deployment-realistic test).
- **Central finding: Regime A does NOT predict Regime C.** Accuracy on a balanced set ≠ behaviour when scanning a real genome.

*Speaker notes:* This is our methodological backbone and the slide a PI will care about most. Any group can report a good AUPRC; the contribution is showing that metric is misleading here and building the genome-scale test that isn't.

---

## Slide 6 — Model Accuracy (Regime A, balanced)

- Weighted test AUPRC on the held-out balanced set:
  - **ProkBERT 0.88 · NT-LoRA 0.85 · CNN-v2 0.85 · CNN 0.80**
- All four are strong and comparable in the standard sense.
- **But this is the metric we just said can mislead** — hold this ranking in mind.

*Speaker notes:* Present these as "good but not the whole story." The deliberate setup: the balanced ranking looks clean, then Regime C reshuffles it. Note we use `average_precision_score`, not the inflated trapezoidal AUPRC.

---

## Slide 7 — Genome-Scale Validation: the Ranking Flips

- Tested on **two held-out genomes** with published promoters: *C. ljungdahlii* (1,245 TSS) and *C. autoethanogenum* (2,885 TSS) — **never seen in training.**
- **NT-LoRA:** 2nd-best balanced accuracy → **collapses** at genome scale (in-genus motif discrimination near chance).
- **The CNN family carries the ensemble** at genome scale, where it counts.
- Same models, opposite conclusion depending on the metric — exactly why Regime C exists.

*Speaker notes:* This is the rigor payoff. We caught a model that looks great on paper and fails in deployment. Two independent held-out genomes give the same answer, so it's not a fluke.

---

## Slide 8 — A Sharp Case: a Frozen Foundation Model (Evo2)

- **Evo2-7B** used *frozen* (no fine-tuning), with a simple probe: **worst balanced accuracy of all (AUPRC 0.70).**
- Yet at **genome scale it beats our best CNN at every recall on both held-out genomes** (e.g. 71% vs 63% precision at recall 0.1 on *ljungdahlii*).
- Fewer false-positive peaks for the same true promoters recovered.
- The clearest demonstration in the project that **balanced accuracy ≠ deployment value.**

*Speaker notes:* Optional but compelling — it shows methodological maturity. We audited the coordinate/strand code and replicated on two genomes before believing it. It reverses our earlier decision to drop Evo2; whether to deploy it is an open call (next-steps slide).

---

## Slide 9 — The Target Mine: End-to-End Result

- Ran the full pipeline on ***C. butyricum* DSM 10702** — the actual target (held out of everything by design).
- Scanned all promoter-plausible regions (300 bp upstream → 50 bp into each gene, both strands).
- Ensemble → peak-calling → **4,027 high-confidence promoter peaks** (each supported by many overlapping windows, not single spikes).

*Speaker notes:* This is the first time the whole pipeline ran on the real target. The peak-calling step collapses ~1.7M redundant scored windows into one call per locus. Stress: the target has no known promoters, so the next slide's independent evidence is how we validate.

---

## Slide 10 — Independent Evidence These Are Real Promoters

- We ran **de-novo motif discovery** on the top calls — *without telling it what to look for.*
- It independently rediscovered the canonical bacterial promoter grammar:
  - the **−10 box** (extended `TATGnTATAAT`) in ~half the top calls,
  - the **−35 box** (`TTGACA`),
  - and the **ribosome-binding site** (`AGGAGG`).
- Recovering real promoter architecture — against an AT-rich background model — is strong evidence we are **not** just picking AT-rich noise.

*Speaker notes:* The single most persuasive slide for the "did it work?" question on a genome with no ground truth. The motif model's background is set to the genome's AT-content, so a `TATAAT` hit is meaningful, not free. This directly answers the Slide 3 trap.

---

## Slide 11 — The Candidate Shortlist (the deliverable)

- Final table: every call + coordinates + confidence + motif evidence, in a locked schema (TSV/JSONL).
- **Tiered for the wet lab:**
  - **82 candidates** with full −10 **and** −35 architecture — maximum certainty.
  - **601 candidates** with a confirmed, correctly-placed −10 — includes extended-−10 promoters (a real class that lacks a −35).
- Ranked by −10 motif significance; the top ~20–30 are the strongest promoters in the genome.

*Speaker notes:* Frame the two tiers as a choice of how many to test, not "real vs fake." Extended-−10 promoters are legitimate — which is why the 601 is the honest high-confidence pool and 82 is the conservative core.

---

## Slide 12 — Honest Caveats (what we're careful about)

- The **target has no ground-truth promoters** — motif evidence is our validation, not precision/recall.
- Published promoter catalogues are **incomplete** (mostly primary, intergenic-sense), so some "false positives" may be real, unannotated promoters — absolute precision is a lower bound.
- Cross-species transfer to the more AT-rich target is **imperfect**; scores are ranks, not calibrated probabilities yet (calibration is in progress).
- Every genome-scale claim is on **held-out** genomes; nothing is evaluated on data it trained on.

*Speaker notes:* Say these unprompted — a PI trusts work that states its own limits. None of them undercut the deliverable; they scope it correctly as an *enriched, evidence-ranked* list, not a perfect map.

---

## Slide 13 — Status & Next Steps

- **Done:** data pipeline, 4-model ensemble trained & validated across three regimes, target mine complete, candidate shortlist produced.
- **In progress:** genome-scale validation of the *full* pipeline on the two held-out genomes; score **calibration** + empirical **false-discovery-rate** control.
- **Planned:** motif/sigma-factor annotation on the full list; **hard-negative mining** to further cut false positives.
- **Open decision:** adopt the frozen foundation model (Evo2) as a 5th ensemble caller for the final target list.

*Speaker notes:* Shows the project is near completion with a clear, honest runway. The Evo2 decision is the one place we'd welcome PI input.

---

## Slide 14 — Takeaways

- Built a 4-model ensemble that beats the AT-rich false-positive trap, and **a genome-scale test that reveals which models actually deploy well.**
- Delivered a ranked, motif-validated promoter shortlist in *C. butyricum* — **82 high-certainty + 601 high-confidence candidates.**
- Independently recovered real promoter architecture from the target's own top calls.
- Honest about scope: an **enriched, evidence-backed candidate list** for the wet lab, with validation on held-out genomes.

*Speaker notes:* Close on the deliverable and the methodological contribution together. One sentence: "We didn't just train models — we built the test that tells us which ones to trust, and used it to hand the wet lab a shortlist we can defend."

---

### Appendix slides (optional, if asked)

- **A1 — Regime C numbers table:** Evo2 vs cnn_v2 precision @ recall 0.1/0.5/0.7/0.9 on both held-out genomes.
- **A2 — Training data composition:** 25,095 promoter positives — **19,344 in-genus (77%) + 5,751 out-of-genus (23%)**. Out-of-genus = *E. coli* 3,827 (RegulonDB) + *B. licheniformis* 888 + *Ruminiclostridium cellulolyticum* 663 + *B. subtilis* 373 — AT-rich Gram-positive relatives + *E. coli* for generalization, all **pinned to the training split** (never in validation/test). In-genus positives are empirical TSS from *C. acetobutylicum* / *C. beijerinckii*. Point: cross-species AT-rich relatives teach real promoter signal without leaking into the deployment metric.
- **A3 — The four negative classes** (coding, intergenic, downstream/3′, shuffled-real) and why shuffled-real is the key AT-trap control.
- **A4 — Candidate table schema** and a worked example row (coordinates, TSS, −10 p-value, spacer).
- **A5 — Why rank-percentile ensembling** is calibration-invariant and lets us combine raw model scores.
