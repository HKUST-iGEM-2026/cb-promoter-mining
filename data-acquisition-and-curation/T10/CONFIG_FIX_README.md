# T10 — TSSpredator config fix (2026-06)

## What was wrong
The previous run (`fuchs_config_hpc.conf`, old version) produced **170,766 TSS** in
`result_hpc/superTSS.gff` — for a ~4 Mb *C. difficile* genome the literature reports a few
**thousand** primary TSS, so it was ~50× over-called and unusable. Causes, in order of impact:

1. **No replicate reproducibility filter.** The 9 Fuchs libraries are **3 conditions × 3
   biological replicates** (LE-TY, ST-TY, LE-BHI), but the config declared
   `numberOfDatasets = 10`, `numReplicates = 1`, `minNumRepMatches = 1` — i.e. every replicate
   was its own single-replicate dataset, so a position seen in **one** library survived into the
   super table. No cross-replicate agreement was ever required.
2. **`minNormalHeight = 0.0`** — no expression floor, so noise was eligible to be called a TSS.
3. **Two studies mixed in one run.** Dataset 1 was the Soutourina **TAP** pair (a different
   enrichment method, single replicate) flattened in with the Fuchs **TEX** data. TSSpredator
   requires an **equal replicate count across all datasets**, so TAP (1 rep) cannot share a run
   with the 3-replicate Fuchs conditions.

The enrichment libraries themselves were fine (TEX+/TAP+ `fivePrime*` were correctly assigned).

## What changed
Split into two corrected, internally-validated configs (every indexed key present; SRR mapping
verified against the original):

| file | datasets × reps | minNumRepMatches | content |
|---|---|---|---|
| `fuchs_config_hpc.conf` (rewritten) | **3 × 3** | **2** | Fuchs TEX: LE-TY, ST-TY, LE-BHI, 3 bioreps each → `result_fuchs/` |
| `soutourina_tap_config.conf` (new) | **1 × 1** | 1 | Soutourina TAP pair (SRR11573852/851) → `result_tap/` |

Also `minNormalHeight 0.0 → 0.2` (a non-zero floor; **tune to the Fuchs 2021 Methods / a
TSSpredator preset** — this and `minCliffHeight` are the main sensitivity knobs). All other
thresholds carried over unchanged.

## ⚠️ The old output is stale
`result_hpc/` was produced by the broken config — **do not use it**. Regenerate on HPC4:

```bash
# on SuperPOD, in a CPU job, with the TSSpredator JAR (it is NOT a conda package):
java -jar TSSpredator.jar fuchs_config_hpc.conf        # -> result_fuchs/
java -jar TSSpredator.jar soutourina_tap_config.conf   # -> result_tap/
```

**Sanity check after re-running:** expect **low thousands** of primary TSS, not 100k+. If
`superTSS.gff` is still enormous, raise `minNormalHeight`/`minCliffHeight` and confirm
`minNumRepMatches = 2`.

## Taxonomy caveat (unchanged by this fix)
This data is ***Clostridioides difficile*** — **out of genus *Clostridium*** (taxon 1485). Even
once re-run cleanly, per CLAUDE.md §7/§8 the **in-genus** empirical positives should come from the
published in-genus TSS tables (the T7b path), so treat the corrected T10 output as
cross-species-adjacent support, not the primary in-genus signal.
