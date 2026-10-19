# DH5α promoter-core strength estimates

Promoter discovery and reporter activity answer different questions. After mining the Cb catalogue, we used a **DH5α-trained SVR** to estimate activity for the first ten promoters with both −10 and −35 motifs, scoring their inferred 50-bp cores. We also retain an older **generic σ70 initiation comparison** below, but its arbitrary units are not interchangeable with the SVR's reporter-activity scale. Neither calculation measures these candidates' expression.

## DH5α-trained SVR strength estimates

We reproduce the SVR procedure from Wang et al., *Nucleic Acids Research* 48:6403–6412 (2020), [doi:10.1093/nar/gkaa325](https://doi.org/10.1093/nar/gkaa325), rather than the paper's whole generative workflow. The [published methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7337522/) distinguish reporter activities measured in DH5α from the same paper's MG1655 transcriptome-trained CNN/generator. The reporter assay used modified pFAB217/p15A, sfGFP, fixed B0030 5′UTR and LB/kanamycin; a different construct can change activity. DH5α is this external assay/model context, not a verified identity for our team's separately reported *E. coli* reporter stock.

### Input and scale

Each of the first ten already-oriented 81-bp records with both −10 and −35 motifs supplies `summit_seq[10:60]`, the 50 bases upstream of the assumed TSS offset 60. No padding or second reverse complement is added. Actual DH5α starts are not mapped. Legacy artifact filenames and the machine-readable `canonical` tier value are unchanged.

The reference is the exact supplied **149-bp `pptb_fU` sequence**, independently supplied and historically aliased DL10 in notebooks. **The current shortlist DL10 is instead `cand_00101` / `P_prfB`**; these are not the same sequence. Its same-model primary core prediction is **6.927284194385597 training-table AU**, set to 1×/100%. No notebook fluorescence or measured J23119 value enters the denominator.

The [results/download index](../results/README.md) collects the [named first-ten TSV](../results/candidates/g131795_first10_canonical_named.tsv), [oriented 81-bp FASTA](../results/candidates/g131795_first10_canonical_named.fasta), [reporter/reference DNA](../results/candidates/sfGFP_pptb_reporter_sequences.fasta) and gene-context records. Naming does not change candidate ranking, activity predictions or annotation confidence. The shared 81-bp discovery frame is a dataset convention, not a ProkBERT input limit; see [input rationale](MODELS.md#why-the-discovery-workflow-uses-81-bp).

```text
activity = 2 ** predicted_log2_activity
fraction_of_predicted_pptb_fU = activity / predicted_pptb_fU_core_activity
percent_of_predicted_pptb_fU = 100 * fraction_of_predicted_pptb_fU
```

### DH5α results

| DH5α SVR rank | Candidate | Predicted activity, training-table AU | Estimated % of predicted pptb_fU core | % across five reference-TSS scenarios |
|---:|---|---:|---:|---:|
| 1 | `cand_00098` | 32.8751 | 474.57 | 93.67–474.57 |
| 2 | `cand_00101` | 17.9027 | 258.44 | 51.01–258.44 |
| 3 | `cand_00029` | 12.0296 | 173.66 | 34.28–173.66 |
| 4 | `cand_00089` | 11.3169 | 163.37 | 32.24–163.37 |
| 5 | `cand_00072` | 10.1352 | 146.31 | 28.88–146.31 |
| 6 | `cand_00035` | 9.92639 | 143.29 | 28.28–143.29 |
| 7 | `cand_00094` | 9.79309 | 141.37 | 27.90–141.37 |
| 8 | `cand_00070` | 7.16201 | 103.39 | 20.41–103.39 |
| 9 | `cand_00032` | 5.91377 | 85.37 | 16.85–85.37 |
| 10 | `cand_00048` | 3.97501 | 57.38 | 11.33–57.38 |


[TSV](../results/candidates/g131795_first10_canonical_dh5alpha_strength.tsv), [JSON](../results/candidates/g131795_first10_canonical_dh5alpha_strength.json) and [CSV](../results/candidates/g131795_first10_canonical_dh5alpha_strength.csv) preserve original DNA, cores, anchors, model settings and full-precision ratios. CSV separates Primary_relative_strength_pptb_fU_equals_1 from the displayed range and full-precision bounds. AU are the upstream table's unspecified activity units, not RFU, physical rates or validated RPU. Percentages are not capped at 100. cand_00048 lies below the fitted labels' minimum; this does not establish inactivity.

### pptb_fU reference framing and uncertainty

The [reference artifact](../results/candidates/pptb_fU_reference.json) preserves DNA and provenance. Promoter Calculator 1.2.4 enumerates 72 contained configurations on the forward reference, choosing offset 113/position 114 at minimum energy −3.4439275934806064. The core is reference[63:113]:

```text
TTCTAAAAAACTTAACTTCATGTGAAAAGTTTGTTAAAATATAAATGCGT
```

The calculator's MG1655-labelled setting is not DH5α-calibrated; it selects a conditional location only. Its initiation-rate AU do not enter the SVR scale. The five lowest-energy starts, selected before SVR scoring, give:

| Reference TSS offset, zero-based | Predicted reference activity, training-table AU |
|---:|---:|
| 113 (primary) | 6.927284 |
| 109 | 30.008978 |
| 111 | 12.589095 |
| 112 | 21.106966 |
| 110 | 35.097146 |


Holding candidate starts at 60, cand_00098 ranges from 0.936689–4.745743× reference (93.67–474.57%). This is a framing scenario range, not a confidence interval or robust improvement claim. Earlier notebook inspection reported DL1–DL4 and the historically DL10-aliased reference in separate notebooks, including saturation in two reference replicates. Underlying numeric assay CSVs are not retained with these predictions, so no notebook fluorescence enters the SVR denominator and no measured fold-change is inferred. Keep that historical reference alias separate from current shortlist DL10 / `P_prfB`.

### Reproduced model and data

Pinned [predictor.py](https://github.com/HaochenW/Deep_promoter/blob/c0e16c8218a983fbb07bf46d59c44109d4371b0e/predictor.py), commit c0e16c8218a983fbb07bf46d59c44109d4371b0e, and the [94-row table](../pipeline/mining/strength-prediction/wang2020-dh5alpha/seq_exp_94.txt) supply the procedure/data. Exclude six short controls and strip cloning overhangs with [5:-1], leaving 83 generated promoters/five random sequences.

Features are float32 T/C/G/A one-hot, flattened to 200 values; target is log2 activity. NumPy permutation seed 2 precedes 5-fold R² grid search over C=[1,10,100,1000], gamma=[0.01,0.1,1,10,100]. RBF SVR uses epsilon 0.1 and refits all 88 rows. Selected C=10/gamma=0.01 gives selection-CV R² **0.266090**, not independent candidate accuracy. The paper's 66/22 split reports held-out Pearson **0.57**.

Our adapter uses raw SVR output/inverse log2, not the upstream maximum-normalized files or swapped CNN/SVR assignments. Data SHA-256 is e81b9316059886846c4e367dd6ae1a0bb196953fcd1a1bc6ac634ed4a1394976; predictor SHA-256 is 37912a418ec1e98b8f935c38731f5365712f6c1e8fe26159c4a366d28d834e68; input for the both-motif subset is 0c6a0e5764ceff7b6ab5ad1e2d45de85df62d1de82b26c3f60a55caa47c232f2. [Upstream MIT notice](../pipeline/mining/strength-prediction/wang2020-dh5alpha/LICENSE) is retained.

### Reproduction

From the repository root:

```bash
uv venv --python 3.13.5 /tmp/cb-dh5alpha-strength
uv pip install --python /tmp/cb-dh5alpha-strength/bin/python \
  -r pipeline/mining/strength-prediction/requirements-dh5alpha.txt
/tmp/cb-dh5alpha-strength/bin/python \
  pipeline/mining/strength-prediction/predict_dh5alpha_strength.py \
  --input results/candidates/g131795_tier1_canonical_seq.tsv \
  --top-n 10 --tss-offset 60 \
  --out-prefix results/candidates/g131795_first10_canonical_dh5alpha_strength
```

The runner writes TSV/JSON/CSV, checks bundled hashes and records dependencies. Observed stack: Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1. No GPU, classifier weights or prediction service is needed. `--tss-offset` controls candidate anchors; the reference artifact supplies offset 113.

### Loading the fitted model

The committed local [fitted artifact](../release/huggingface/dh5alpha-promoter-strength-svr/) contains numeric support vectors, coefficients, intercept and gamma in `model.npz`, plus checksum/provenance metadata. Inference uses NumPy with no pickle execution or repeat fitting. The same fit is available from the [public, ungated Hub package](https://huggingface.co/CloBiT/dh5alpha-promoter-strength-svr): the recorded download at revision `0c5202892f4255eb8eb17493bed8d51e04f8cfc4` reproduced all ten results. Downloads need no account/token; uploading requires write authorisation. This SVR is separate from the three mining classifiers, not a hosted service or packaged ensemble.

The local inference route does not depend on Hub access:

```bash
uv run --no-project --python 3.13.5 --with numpy==2.5.4 python \
  pipeline/mining/strength-prediction/predict_dh5alpha_strength.py \
  --model release/huggingface/dh5alpha-promoter-strength-svr \
  --reference results/candidates/pptb_fU_reference.json \
  --input results/candidates/g131795_tier1_canonical_seq.tsv \
  --top-n 10 --tss-offset 60 --out-prefix /tmp/dh5alpha_first10
```

The current audit exercised this local NumPy-only route without refitting or uploading. All ten saved log₂ activities, linear activities, primary reference ratios and five-reference-start scenario bounds agreed within floating-point tolerance. The maximum absolute activity difference was **2.6645352591003757 × 10⁻¹⁴ AU**; the maximum percentage difference was **1.2505552149377763 × 10⁻¹² percentage points**. All five reference-core activities and the candidate DNA/crops/order also matched. Observed inference environment: Python 3.13.5/NumPy 2.5.4. Fitted weights SHA-256: `153f41e1d051d432a2172b21b6e94f07799fa2229a45d795ae974fc8140130a8`. These are computational checks, not biological validation or verification of a remote package.

To export another reproduction of the unchanged procedure, add `--export-model` with a new output directory to the fitting command above. Do not overwrite the retained fitted artifact when only running inference.

The same retained fit can instead be downloaded anonymously at its recorded revision:

```bash
hf download CloBiT/dh5alpha-promoter-strength-svr \
  --revision 0c5202892f4255eb8eb17493bed8d51e04f8cfc4 --local-dir dh5alpha_svr
uv run --no-project --with numpy==2.5.3 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --model dh5alpha_svr --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv --top-n 10 --tss-offset 60 \
  --out-prefix dh5alpha_first10
```

To rederive the conditional reference frame:

```bash
uv pip install --python /tmp/cb-dh5alpha-strength/bin/python \
  -r pipeline/mining/strength-prediction/requirements.txt
/tmp/cb-dh5alpha-strength/bin/python - <<'PY'
import json
from pathlib import Path
from promoter_calculator.promoter_calculator import Promoter_Calculator
ref = json.loads(Path("results/candidates/pptb_fU_reference.json").read_text())
calculator = Promoter_Calculator(organism=ref["framing_organism"], threads=1, verbosity=0)
sequence = ref["sequence_5to3"]
forward, _ = calculator.predict(sequence, [0, len(sequence)], callback=lambda: None)
best = min(forward.values(), key=lambda p: (float(p["dG_total"]), int(p["TSS"])))
print(len(forward), int(best["TSS"]), float(best["dG_total"]))
PY
```

### Verification and limits

Recorded checks matched raw log2/inverse predictions against the pinned source and all 98 training/candidate feature rows, preserved input DNA/crops, rejected invalid offsets 49/81 and checked TSV/JSON/CSV agreement. The reference stayed fixed when cand_00029 was scored alone. Candidate-offset sensitivity 58–62 is [saved separately](../results/candidates/g131795_first10_canonical_dh5alpha_tss_sensitivity.json): cand_00048 leads at 58, cand_00098 at 59–62. Do not pick each shifted maximum as its primary result.

The small fit, inferred starts, leaders, flanks, copy number and actual conditions limit interpretation. These are core predictions, not whole-81-bp candidate or whole-149-bp reference expression. No biological strong/weak/inactive threshold or candidate-specific error interval is supported.

## Earlier generic σ70 initiation comparison

Promoter Calculator v1.0 from LaFleur, Hossain and Salis, [doi:10.1038/s41467-022-32829-5](https://doi.org/10.1038/s41467-022-32829-5), is packaged by [Barrick Lab](https://github.com/barricklab/promoter-calculator/tree/0fe5f6f0fb2befc087556afa575f7241afd98fbf) as 1.2.4, commit 0fe5f6f0fb2befc087556afa575f7241afd98fbf. Parameter files match the original Salis implementation. It combines fitted −35/−10, extended −10, spacer, UP, discriminator and transcribed-region terms, choosing minimum-energy configurations per TSS.

DH5α remains the target host; the **MG1655-labelled in-vivo constants are an unvalidated proxy**. Other organism strings reuse them, so changing the label adds no strain correction. The paper's DH10B characterization/NEB 5-alpha library construction does not validate these ten candidates.

```text
Tx_rate = 42 × exp(−1.636217004872062 × dG_total)
```

### Thermodynamic comparison results

Primary rate is the largest forward-strand estimate among fully contained configurations in each exact oriented 81-bp fragment. Max=100 is a within-ten display normalization, not a reference standard or probability.

| E. coli rank | Candidate | Best-site rate, model AU | Relative strength, max = 100 | Model TSS within fragment, 1-based | Rate at original anchor, model AU |
|---:|---|---:|---:|---:|---:|
| 1 | `cand_00094` | 18,659.9 | 100.00 | 62 | 1,249.94 |
| 2 | `cand_00098` | 8,721.08 | 46.74 | 62 | 896.738 |
| 3 | `cand_00035` | 6,944.23 | 37.21 | 62 | 967.413 |
| 4 | `cand_00089` | 4,734.29 | 25.37 | 61 | 4,734.29 |
| 5 | `cand_00029` | 4,066.81 | 21.79 | 62 | 3,639.43 |
| 6 | `cand_00032` | 2,254.98 | 12.08 | 62 | 511.131 |
| 7 | `cand_00072` | 1,874.29 | 10.04 | 61 | 1,874.29 |
| 8 | `cand_00070` | 640.455 | 3.43 | 61 | 640.455 |
| 9 | `cand_00048` | 236.960 | 1.27 | 60 | 220.325 |
| 10 | `cand_00101` | 166.535 | 0.89 | 61 | 166.535 |


The predicted range is about 112-fold. At fixed offset 60, cand_00089 leads instead of cand_00094; the latter changes 14.93-fold after a one-base TSS shift. [TSV](../results/candidates/g131795_first10_canonical_ecoli_strength.tsv) and [JSON](../results/candidates/g131795_first10_canonical_ecoli_strength.json) retain full precision and provenance.

### Context boundary

A complete model geometry can need 67 upstream/20 transcribed bases. These fragments permit only TSS offsets 58–61, with 1/3/6/10 geometries respectively: 20 per fragment, not all geometries of a full construct. Offset 60 permits six of the nominal 30. No vector sequence or invented neutral flanks are added; the old motif contexts/center-separation fields are not model inputs. Actual construct context may alter rates and ranking.

### Thermodynamic comparison reproduction

```bash
uv venv --python 3.13.5 /tmp/cb-ecoli-strength
uv pip install --python /tmp/cb-ecoli-strength/bin/python \
  -r pipeline/mining/strength-prediction/requirements.txt
/tmp/cb-ecoli-strength/bin/python \
  pipeline/mining/strength-prediction/predict_ecoli_strength.py \
  --input results/candidates/g131795_tier1_canonical_seq.tsv \
  --top-n 10 \
  --out-prefix results/candidates/g131795_first10_canonical_ecoli_strength
```

The runner verifies package version/source and distinguishes target_host from model_organism/host_transfer_status. Observed dependencies also include Biopython 1.88. Source-ranked input SHA-256 is 316940e6e2abb4ed77d4eeeef3cbbf9d1b2c30706da0f1288580d00f75f1f829; the reproducible command uses the matching committed both-motif subset file. Parameter hashes: free_energy_coeffs.npy fddf16b38bf29799481e8a45890b669111e16cf2e3448e99d1934d19a8f0b8e9, model_intercept.npy f7c66a8d838da71a6b2ab17936cd6ff2564e446af8c0676fba95de2116e99a7b.

Recorded exhaustive checks covered all 200 contained configurations, element/DNA round trips, exponential conversion and unchanged rates after separating host metadata. They verify computation, not expression, sigma use, regulation or cotranscription. Reporter/translation and cellular effects are outside the model. The external package remains [GPL-3.0-or-later](https://github.com/barricklab/promoter-calculator/blob/0fe5f6f0fb2befc087556afa575f7241afd98fbf/LICENSE).
