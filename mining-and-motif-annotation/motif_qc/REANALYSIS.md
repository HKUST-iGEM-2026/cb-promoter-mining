# g131795 −10/−35 motif QC — reanalysis on real inputs

Re-run of the "cross-element contamination" check from R. Swastika's report (2026-09-19),
rebuilt on the actual files in this repo rather than reconstructed PWMs:

- **motifs:** `mining-and-motif-annotation/motif-annotation/sigma_a_promoter.meme` — the pipeline's own MEME
  model (canonical `TATAAT` −10 / `TTGACA` −35, width 6), with an **AT-rich background baked in**
  (A/T = 0.35, C/G = 0.15).
- **candidates:** `g131795_ranked_seq.tsv` (929 rows, the delivered table with `minus10_seq` /
  `minus35_seq` / `spacer_bp`).
- **code:** `reanalyze_motif_calls.py` (stdlib + numpy/matplotlib, fixed seed). Outputs
  `reanalysis_results.json` and `fig1–4`.

## Bottom line

The original computation is **correct and reproduces exactly**. Its *cause* (a fixable −35
windowing/offset/extraction bug) is **not** supported. What the data actually shows is that the
−35 boxes carry **almost no canonical `TTGACA` signal above random AT-rich DNA**, while the −10
boxes are strong and clean — which is the *expected* promoter architecture for an AT-rich
*Clostridium* genome, not a pipeline defect. So the practical recommendation (lean on the −10
evidence, treat the −35-dependent "canonical" label with caution) is right; the mechanism is
biology + a weak per-run −35 motif, not a coordinate bug, and it does **not** touch the ranked
929 / peak-score deliverable (which never uses the −35 call).

## Numbers

| Check | Result | Reading |
|---|---|---|
| Coverage (929) | both 45 · −10-only 135 · −35-only 288 · neither 461 | matches the report exactly |
| −35 "contamination" (prefers −10 motif) | **86.5 %** uniform bg · **86.2 %** AT-rich bg | reproduces the report's 86.8 % |
| −10 "contamination" (prefers −35 motif) | 5.0 % · 5.6 % | reproduces the report's 5.0 % |
| Random AT-rich control (non-promoter) | 61 % uniform · 53 % AT-rich "prefer −10" | a majority of *random* AT-rich DNA also flags |
| **Signal vs matched null — −10 boxes** | **median pct 98 · 79 % above null-95th** | positive control: real, strong −10 signal |
| **Signal vs matched null — −35 boxes** | **median pct ~51 · ~12 % above null-95th** | ≈ noise: little real `TTGACA` signal |
| Spacer (`spacer_bp`, n=45) | min 5 · median **18** · max 34 | canonical-centered; rules out a gross offset |
| AT content | −10 88 % · −35 85 % · summit 79 % | boxes are very AT-rich |

## What this changes vs the score-only report

1. **Background is not the driver.** The report scored against a uniform 0.25 background; the
   motif file itself declares an AT-rich background. Swapping to the correct background moves the
   −35 number by 0.3 points (86.5 → 86.2 %). *(This also corrects my own first-pass guess that the
   uniform background was doing the damage — on the real motifs it isn't.)* The −35 motif's
   information actually **rises** slightly under the AT-rich background, because its discriminating
   bases are the G (pos 3) and C (pos 5) that are rare in this genome.

2. **A high flag rate is partly just what AT-rich sequence does.** Feeding random,
   non-promoter AT-rich 14-mers through the identical test, **53–61 % also "prefer the −10 motif."**
   The two reference motifs are not on a comparable scale (raw log-odds from a peaked `TATAAT` vs a
   flatter `TTGACA` can't be compared directly), so "scored higher against −10" is a weak signal on
   its own. The real −35 boxes flag *higher still* (86 %) — see #3.

3. **The decisive test is signal-vs-null, and it splits the two arms cleanly** (`fig2`). Scored
   against a matched AT-rich null, the **−10 boxes are real** (median percentile 98; 79 % clear the
   null's 95th percentile — the positive control proving the method detects signal when it exists),
   whereas the **−35 boxes are indistinguishable from random AT-rich DNA** (median percentile ~51;
   only ~12 % clear the 95th). So the −35 windows are, as a population, AT-rich stretches in the
   right position that resemble the AT-rich `TATAAT` more than the G/C-containing `TTGACA` — not
   mislabeled −10 boxes, just not strong −35 boxes.

4. **Not an offset/coordinate bug.** The −10 boxes are cleanly `TATAAT`, the −35 box sits upstream
   of the −10 in 44/45, and `spacer_bp` is centered at the textbook 18 bp. A windowing/offset error
   would break these; re-extracting won't manufacture a `TTGACA` that isn't there. (Caveat: the
   positional gating enforces *some* spacer range, so median-18 rules out a gross offset but is only
   weak positive evidence.)

## Biology

A strong −10 with a weak or absent −35 is the **expected** architecture in AT-rich Firmicutes such
as *Clostridium* — many promoters rely on a strong (often extended) −10 and lack a conserved
`TTGACA` −35. So scoring these −35 windows against *E. coli*'s `TTGACA` is somewhat unfair, and a
low −35 signal is a feature of the organism, not evidence the pipeline mis-cut the window. (A quick
extended-−10 `TG` proxy on the 45 canonical was inconclusive — 3/45 — so that specific mechanism is
plausible but unconfirmed here.)

## Recommendations

- **Demote "canonical" to a soft annotation, not a quality gate.** Rank the wet-lab shortlist by
  ensemble `peak_score` + −10 strength; the 929 confident set and its ranking come from the
  ProkBERT+cnn_v2+Evo2 ensemble and are unaffected by the −35 call. This matches the existing
  `CLAUDE.md` note that "canonical is a motif-convenience subset."
- **Do not re-window / re-extract the −35 boxes to "fix" 86.8 %** (the report's follow-up #1) — the
  signal isn't there to recover, so a re-extraction would chase noise.
- **If an architecture call is wanted for this organism,** learn a *Clostridium*-specific −35 (or an
  extended-−10) model instead of using the *E. coli* `TTGACA` reference, and report −35 strength as
  a FIMO p-value/percentile-vs-null (as here), not a raw-bits comparison between two motifs.
- The report's other follow-ups stand: the −10 arm is trustworthy, and separately characterizing
  the 461 "neither" candidates is worthwhile.
