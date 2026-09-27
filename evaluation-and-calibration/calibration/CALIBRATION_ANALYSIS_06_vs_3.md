# T26 — Calibration Analysis: `calib_06` vs `calib_3` (comparative)

Comparative read of the two consolidated reports:

- [`calib_06/calibration_analysis.md`](calib_06/calibration_analysis.md) — calibrators fit on
  **`results/T21_final_06/<model>`**: the **0.6-hyperparameter-swept** final models (the set the project
  adopted for ProkBERT).
- [`calib_3/calibration_analysis.md`](calib_3/calibration_analysis.md) — calibrators fit on
  **`results/T21_final/<model>`**: the earlier final models carrying **0.9-swept hyperparameters**.

**Both runs share everything except the model checkpoint.** Same 4 models (cnn, cnn_v2, nt_lora, prokbert),
same AT-aware Platt calibrator (`StandardScaler → LogisticRegression([logit, AT])`, unregularized chosen by
5×5 stratified CV in every case), and the **same 279-sequence eval set** — 93 T9 positives + 186 genomic
negatives (130 convergent-intergenic + 56 CDS-interior), 81 bp windows. Both model sets were **trained on the
0.6-clustered data** (`T21_final` and `T21_final_06` differ only in which sweep tuned their
hyperparameters). So this is a clean A/B of the **0.6 hyperparameter re-sweep**, viewed through calibration.

> The two takeaways sections in the source files are identical boilerplate; this document is the part that
> actually compares the runs. Numbers are lifted verbatim from the two reports' §2/§3/§4 tables.

---

## 1. TL;DR — who won what

| Model | Best checkpoint (by calibrated discrimination) | One-line verdict |
|---|---|---|
| **cnn_v2** | **`calib_3` (0.9-swept)** — cal AUROC **0.872**, AUPRC **0.769** | Best model in the whole comparison. The 0.6 re-sweep *regressed* it (→0.834/0.692). **Keep the T21_final checkpoint.** |
| **prokbert** | **`calib_06` (0.6-swept)** — cal AUROC 0.799, AUPRC 0.643 | Effectively a tie (0.796/0.630 in calib_3); 0.6 wins on raw too. **Adopt T21_final_06**, consistent with the genome-scale finding. |
| **nt_lora** | **`calib_06` (0.6-swept)** — cal AUROC 0.769, AUPRC 0.617 | The 0.9-swept nt raw ranker **collapsed below chance** (raw AUROC 0.470). 0.6 is decisively healthier, but nt is still the weakest member either way. |
| **cnn** | ~tie (calib_06 0.833/0.701 vs calib_3 0.829/0.687) | The most *honest* calibrator (fewest rows harmed) in both runs; middling discrimination. Stable, not a leader. |

**Headline:** the 0.6 re-sweep is **not uniformly better**. It clearly rescues **nt_lora** (from a sub-chance
raw ranker) and marginally helps **prokbert**, but it **hurts cnn_v2** and barely moves **cnn** — which is
exactly the mixed picture the 0.6 re-sweep produced: cnn_v2 and NT slipped slightly, which reads as proxy
noise, so they keep their current versions while ProkBERT adopts the 0.6-swept checkpoint. The calibration evidence independently corroborates a **mixed-checkpoint
deployment**, not a blanket switch to `T21_final_06`.

---

## 2. Raw discrimination — before calibration

| Metric | model | `calib_06` raw (0.6-swept) | `calib_3` raw (0.9-swept) | Δ (06 − 3) |
|---|---|---|---|---|
| AUROC | cnn | 0.750 | 0.715 | +0.035 |
| AUROC | cnn_v2 | 0.720 | **0.775** | −0.055 |
| AUROC | nt_lora | 0.683 | **0.470** ⚠️ | +0.213 |
| AUROC | prokbert | 0.708 | 0.691 | +0.017 |
| AUPRC | cnn | 0.596 | 0.591 | +0.005 |
| AUPRC | cnn_v2 | 0.586 | **0.633** | −0.047 |
| AUPRC | nt_lora | 0.504 | **0.337** | +0.167 |
| AUPRC | prokbert | 0.572 | 0.560 | +0.012 |

**What this says.** Discrimination (AUROC/AUPRC) is a property of the *raw model*, not the calibrator — the
Platt map is monotonic in the score alone, so these raw columns are the honest read on the checkpoints.

- **nt_lora is the story.** With 0.9-swept knobs its raw AUROC is **0.470 — below the 0.5 coin-flip line**,
  i.e. on this genomic eval set it ranks negatives *above* positives. The 0.6 re-sweep pulls it back to a
  weak-but-real 0.683. This is the same instability seen at genome scale (NT's Regime-B in-genus
  AUROC "collapsed to chance, 0.440") — the calibration set reproduces it independently. Whatever the sweep is
  doing to NT, the 0.9 hyperparameters left it fundamentally broken on out-of-training composition.
- **cnn_v2 goes the other way.** Its raw ranker is *stronger* with the old 0.9-swept knobs (0.775 vs 0.720
  AUROC, 0.633 vs 0.586 AUPRC). The 0.6 re-sweep is a small but real regression here — consistent with
  treating cnn_v2's re-sweep drop as "proxy noise" and keeping the current version.
- **cnn and prokbert are nearly sweep-invariant** (Δ ≤ 0.035). Their raw rankers don't care much which sweep
  tuned them; for these two the 0.6 side is marginally ahead, which is enough (together with the genome-scale
  evidence) to prefer `T21_final_06` for prokbert.

---

## 3. Calibrated discrimination + reliability — after the AT-aware Platt map

Calibrated = out-of-fold `calibrated_logit_at_oof` (the deployed behaviour). ECE/Brier/log-loss lower is better.

| Metric | model | `calib_06` (0.6-swept) | `calib_3` (0.9-swept) | better run |
|---|---|---|---|---|
| **cal AUROC** | cnn | 0.833 | 0.829 | ~tie |
| | cnn_v2 | 0.834 | **0.872** | calib_3 |
| | nt_lora | **0.769** | 0.712 | calib_06 |
| | prokbert | 0.799 | 0.796 | ~tie |
| **cal AUPRC** | cnn | **0.701** | 0.687 | calib_06 |
| | cnn_v2 | 0.692 | **0.769** | calib_3 |
| | nt_lora | **0.617** | 0.562 | calib_06 |
| | prokbert | **0.643** | 0.630 | calib_06 |
| **ECE** | cnn | 0.064 | **0.029** | calib_3 |
| | cnn_v2 | 0.081 | **0.059** | calib_3 |
| | nt_lora | 0.063 | **0.044** | calib_3 |
| | prokbert | 0.065 | **0.056** | calib_3 |
| **log-loss** | cnn | 0.469 | 0.480 | calib_06 |
| | cnn_v2 | 0.472 | **0.421** | calib_3 |
| | nt_lora | **0.536** | 0.583 | calib_06 |
| | prokbert | 0.517 | 0.516 | ~tie |

**What this says.**

- **Calibration works in every cell.** Compare against §2's raw numbers: every model in both runs gains AUROC
  and AUPRC and sheds ECE/Brier/log-loss after the AT-aware Platt map. Nothing here is a question of *whether*
  to calibrate — it's which checkpoint feeds the calibrator.
- **`calib_3` shows lower ECE across all four models.** Post-hoc reliability is slightly tighter for the
  0.9-swept set. But ECE is a *reliability* score, not a *ranking* score — a model can be beautifully calibrated
  and still rank poorly (nt_lora in calib_3 is the poster child: ECE 0.044, yet its raw ranker is sub-chance).
  For a mining pipeline that thresholds/ranks candidates, **AUPRC and raw AUROC matter more than ECE**, so the
  lower-ECE calib_3 is not automatically the deployment winner.
- **cnn_v2 @ calib_3 is the single best cell in the whole table** (cal AUROC 0.872, AUPRC 0.769, log-loss
  0.421). It is the only model that is *both* a strong raw ranker *and* well-calibrated — the profile you
  actually want. This is the concrete reason to keep cnn_v2 on its `T21_final` checkpoint.
- **The biggest headline metric T26 buys is log-loss, not AUROC.** Raw log-loss for nt_lora/prokbert is
  ~1.2–1.6 (see the source §2 raw columns) because they score real promoters near 0.00; calibration cuts that
  to ~0.52–0.58. That ~60–65% log-loss reduction — a proper-scoring-rule collapse of pathological
  overconfidence — is the main deliverable of T26 for the transformer members.

---

## 4. How much does each model lean on the AT crutch?

The calibrated score is a 2-feature logistic on `[raw_logit, AT]`, so it can **re-rank** relative to the raw
logit alone. The size of the raw→calibrated AUROC jump measures **how much separation the AT covariate adds on
top of the model's own score** — a large jump means the model isn't reading motif well and AT is carrying it.

| Model | raw→cal AUROC, `calib_06` | raw→cal AUROC, `calib_3` | AT coef (06 / 3) |
|---|---|---|---|
| cnn | 0.750 → 0.833 (**+0.083**) | 0.715 → 0.829 (**+0.114**) | +1.076 / +1.127 |
| cnn_v2 | 0.720 → 0.834 (**+0.114**) | 0.775 → 0.872 (**+0.097**) | +1.230 / +1.351 |
| nt_lora | 0.683 → 0.769 (**+0.086**) | 0.470 → 0.712 (**+0.242** ⚠️) | +0.994 / +0.878 |
| prokbert | 0.708 → 0.799 (**+0.091**) | 0.691 → 0.796 (**+0.105**) | +0.959 / +0.994 |

**What this says.**

- **The AT covariate earns its place in all 8 model×run cells** (source §3: every `+AT` lowers OOF log-loss,
  every AT coef is positive → high-AT sequences pushed up, AT-rich promoters rescued). That's a robust,
  checkpoint-independent effect and the central justification for the AT term.
- **The +0.242 jump for nt_lora @ calib_3 is a red flag, not a success.** When the raw ranker sits at 0.470,
  a calibrated 0.712 means **the AT-content feature — not NT's learned representation — is doing almost all the
  discrimination.** The "calibration" is really composition-matching in disguise: it happens to separate this
  eval set only because the T9 positives skew AT-rich. That will not transfer to the target *C. butyricum*
  scan the way genuine motif signal would. The 0.6 re-sweep (jump only +0.086 off a healthier 0.683 raw) is
  the far safer nt_lora.
- **cnn_v2 has the ideal profile in calib_3:** a strong raw ranker (0.775) that *still* benefits from AT
  (+0.097) — AT is a top-up, not a crutch. cnn is similar but off a lower base.

---

## 5. The "calibrator made the majority of rows worse" paradox

Both source reports flag ⚠️ that for **nt_lora and prokbert the calibrator made most rows *worse* than raw**
(§4): calib_06 nt 192/279, prokbert 197/279; calib_3 nt 171/279, prokbert 187/279 — yet §2 shows every headline
metric *improved*. These are not in conflict, and the resolution is the same in both runs.

**Mechanism.** The transformers are **conservative**: they score the 186 genomic negatives almost perfectly
(prokbert raw CDS err 0.06–0.12, convergent 0.11–0.22) but score the 93 real promoters near **zero** (T9
positive raw err 0.56–0.86 — the model calls true promoters non-promoters with high confidence). A globally-fit
calibrator pulls everything toward the base rate: it **rescues the catastrophic positives** (large per-row
gains) and **slightly degrades the already-excellent negatives** (small per-row losses). Because negatives
outnumber positives 2:1, more rows tick "worse" — but the *magnitude* of the fixes on the positives dwarfs the
magnitude of the regressions on the negatives. Log-loss/Brier/AUPRC weight the egregious errors heavily, so they
all improve; the unweighted "rows made worse" count does not, and is therefore **misleading here.**

**Contrast with the CNNs**, where far fewer rows are harmed (cnn 84–102/279, cnn_v2 84–110/279). The CNNs are
**liberal** — they over-score AT-rich genomic negatives — so the calibrator's downward pull helps a *majority*
of rows. Fewer rows harmed is not "better calibration," it's a different raw error geometry (see §6).

**Takeaway:** ignore the raw "rows made worse" count for the transformers; it is an artifact of class imbalance
× conservative scoring, and §2's proper scores confirm calibration is doing the right thing.

---

## 6. Complementary error directions — why the ensemble is more than the sum

The per-class error tables (source §4) reveal the two model families **fail in opposite directions**, in both
runs:

| Family | on genomic **negatives** (CDS + convergent) | on real **positives** (T9) |
|---|---|---|
| **CNN / cnn_v2** (liberal) | raw err **high** (CDS ~0.33–0.36, conv ~0.41–0.44) → over-calls AT-rich junk | raw err **low** (~0.31–0.40) |
| **nt_lora / prokbert** (conservative) | raw err **low** (CDS ~0.06–0.18, conv ~0.10–0.22) → clean on negatives | raw err **high** (~0.56–0.86) → misses real promoters |

**What this says.** The CNNs are exactly the AT-trap failure mode the project is built to avoid (score AT-rich
non-promoters as promoters); the transformers are the opposite (so wary of the AT-trap they reject genuine
AT-rich promoters). Calibration nudges each family toward the middle — for CNNs it fixes the negatives (CDS err
0.33→~0.10 after calibration), for transformers it fixes the positives (T9 err 0.86→0.58 for nt). **Because the
raw errors are anti-correlated across families, the T29 mean-rank-percentile ensemble should cancel a chunk of
both** — a liberal CNN vote and a conservative transformer vote average toward the truth. This is direct,
per-class evidence that the four-model ensemble is well-motivated and not just redundancy (the two CNNs *are*
correlated with each other, which is why they are flagged for T29 down-weighting — but the CNN-vs-transformer axis is the
real diversity).

---

## 7. Failure mode to watch: low-AT true promoters

The "worst 8 rows" tables expose one sharp, reproducible cost of the AT covariate. In **both** runs, the worst
calibration flips for nt_lora and prokbert are **POS rows the raw transformer got right and the +AT calibrator
destroyed**:

- `calib_3` nt_lora: POS, AT **0.47**, raw **0.977** → calib **0.036** (Δ **+0.942**); POS, AT 0.51, raw 0.991 → calib 0.069 (Δ +0.922).
- `calib_3` prokbert: POS, AT 0.47, raw 0.996 → calib 0.080; POS, AT 0.51, raw 0.996 → calib 0.154.
- `calib_06` shows the same pattern (cnn_v2: POS, AT 0.47, raw 0.982 → calib 0.085, Δ +0.898).

**Mechanism.** These are the minority **low-AT promoters** (AT ≈ 0.47–0.53, below the promoter mean). A positive
AT coefficient pushes low-AT sequences *down* — so a genuine promoter the raw model scored ~0.98 gets collapsed
below 0.1 **purely because its composition is atypical.** The AT term rescues the many AT-rich positives at the
explicit cost of the few AT-poor ones. On this 93-positive set the trade is net-positive (§3), but for the
final target scan it means: **calibrated scores will systematically under-rank low-AT promoters.** If any
*C. butyricum* promoters of interest are GC-leaning, prefer the raw score or the rank-percentile ensemble
(§8) over the calibrated probability for those. Worth a small held-out check stratified by AT before T27/T32.

---

## 8. Implications for deployment (T27 / T28 / T29 / T32)

1. **The T29 ensemble is rank-percentile → calibration-invariant.** Per-model Platt maps do **not** change the
   ensemble ranking, only per-model probabilities and fixed thresholds. So the `calib_06`-vs-`calib_3`
   *calibration* differences are moot for the ensemble ranking — what matters there is **raw AUROC/AUPRC**
   (§2), which is why the mixed-checkpoint call is made on the raw rankers, not the ECE.
2. **Where calibration DOES bind: T27 empirical FDR and any per-model threshold / reported probability.** The
   SRS-null q-values in T27 are computed on a score distribution; if a model scores real promoters near 0
   (transformers, §5) the null and signal overlap badly until calibrated. **Calibrate before T27**, and prefer
   the checkpoints with healthier raw rankers so the AT term is a top-up not a crutch (§4).
3. **Recommended calibrated checkpoints (calibration-evidence view):**
   - **prokbert → `T21_final_06` (calib_06)** — better raw + marginally better calibrated; matches the genome-scale result.
   - **nt_lora → `T21_final_06` (calib_06)** — the 0.9-swept nt is sub-chance raw; 0.6 is the only defensible
     nt. (It remains the weakest member on every axis, the round's "weak link" — lean on the CNN family.)
   - **cnn_v2 → `T21_final` (calib_3)** — best raw *and* calibrated discrimination; the 0.6 re-sweep regressed
     it. Keep the current checkpoint.
   - **cnn → either** (sweep-invariant); take whichever checkpoint is already staged.
4. **Do not read ECE as the deployment ranking.** `calib_3` has lower ECE everywhere but that does not make its
   checkpoints better for mining — nt_lora @ calib_3 is well-calibrated *and* sub-chance. Rank on AUPRC + raw
   AUROC; use ECE/Brier only to confirm the probabilities are trustworthy once a checkpoint is chosen.
5. **Small-eval caveat.** 93 positives, all from T9 (synthetic/literature, incl. out-of-genus *C.
   phytofermentans*), drive every "worst row" and the AT-coef fits. Treat the absolute numbers as directional;
   the *cross-run consistency* (nt collapse, CNN-vs-transformer error split, low-AT failure) is what's robust,
   because it reproduces across two independent checkpoint sets on the same data.

---

<sub>Source tables: `calib_06/calibration_analysis.md` and `calib_3/calibration_analysis.md` (§2 metrics, §3 AT
ablation, §4 per-class + worst-row breakdowns). Checkpoint provenance from each report's §1. `calib_06 =
T21_final_06` (0.6-swept), `calib_3 = T21_final` (0.9-swept); both trained on 0.6-clustered data, calibrated on
the identical 279-sequence set. Deployment decisions and the AT-trap rationale are summarised in the repo
[`README.md`](../../README.md).</sub>
