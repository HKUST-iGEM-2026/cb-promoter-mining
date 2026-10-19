# Models

Three complementary sequence representations feed the genome-mining ranking. A separate DH5α-trained SVR estimates reporter activity for 50-bp cores from the first ten promoters with both −10 and −35 motifs; it is not a fourth ensemble member. [Results](RESULTS.md) explains how the benchmark measurements and conditional strength predictions should be interpreted.

## The deployment ensemble

| Member | Implementation | Deployment role |
|---|---|---|
| **ProkBERT-mini** | full fine-tune of `neuralbioinfo/prokbert-mini` with a two-class head | prokaryote-pretrained transformer |
| **CNN-v2 fusion** | from-scratch one-hot/ENAC convolution branches plus dinucleotide/trinucleotide frequency branches | local and composition features |
| **Evo2-7B probe** | logistic regression on frozen 4096-dimensional embeddings | strongest standalone member in the reported restricted benchmarks |


Members read **81-bp transcription-oriented DNA**, nominal TSS offset 60. Silver windows are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped starts or experimentally measured promoters; curated length normalization does not locate a start. The mine intersects step-10 tracks on contig/start/strand, converts each member's scores to average-tie percentiles in that common pool, then takes their mean. This is neither probability, strength nor unanimity. No calibrator is used. Only strictly increasing score-only transforms preserve ranks without changing ties.

### Why the discovery workflow uses 81 bp

Published-TSS extraction takes 60 upstream bases, the TSS base and 20 downstream bases, giving zero-based TSS offset 60. This shared source convention keeps labels, encodings and scan coordinates aligned; normalising curated sequences does not map their starts. **81 bp is not a ProkBERT architectural limit or a demonstrated optimal window length.** Its recovered config supports 1024 token positions. Overlapping six-mers turn 81 bp into 76 sequence tokens plus CLS/SEP, or **78 tokens**; the example below pads to width 81 tokens.

The deployed Evo2 extraction job selected the per-row `sequence` column: all retained inputs are 81 bp, with no `sequence_context` column. Examples were not stitched together or extended with flanks. Longer backbone capacity does not imply that the fitted probe was trained on longer inputs.

### Architectures

ProkBERT has six layers/six heads, hidden size 384, overlapping six-base tokens and padding-masked learned attention pooling before its two-class head. The recovered config records float32 and Transformers 5.3.0. Saved training arguments confirm learning rate 4.675283456079137e-05, train/eval batches 16/64, 10 maximum epochs, weight decay 0.07894770663194649 and seed 42. Warmup ratio 0.11090738634337446 is stored in float warmup_steps; warmup_ratio is None. Saved training uses fused AdamW, linear schedule, epoch evaluation/saving and bf16; CPU inference uses float32.

CNN-v2 combines one-hot/ENAC convolution branches (kernels 7/5/3, widths 64/128/256, batch normalization/pooling) with 16-d dinucleotide and 64-d trinucleotide projections. Concatenation order is onehot,dinuc,trinuc,enac; fusion dimension is 256. A/C/G/T channels, all-position k-mer denominators and clipped centered ENAC width 5 define the features. N adds no base-channel count; inputs are center-cropped or N-padded to 81 bp. Recorded training uses learning rate 0.0003748224111698842, batch 32, dropout 0.10382857322260322, weight decay 0.00024777455052193346, 30 epochs, patience 8 and seed 42.

Evo2 uses a trusted StandardScaler→LogisticRegression probe with 4096 features, classes [0,1] and scaler count 47,137. Current DNA inference mean-pools blocks.28.mlp.l3 from evo2_7b; the exact historical extraction revision/layer is not established. Backbone weights stay frozen and are obtained separately.

## Loading

### Public Hugging Face repositories

CNN-v2, ProkBERT, the Evo2 probe and the DH5α SVR are public and ungated. Project-package downloads require no account or HF_TOKEN; uploads require write-authorised credentials. Released weight hashes match recovered local copies. Review executable remote code before trusting it, and follow separate upstream-backbone access and licence conditions. Never load untrusted pickle/joblib files: they can execute code. Cards do not sync automatically from GitHub. The fitted DH5α NPZ is also available in `release/huggingface/dh5alpha-promoter-strength-svr`. See [compatible setup](SETUP.md) and the [public results/download index](../results/README.md).

| Repository | Model-card license |
|---|---|
| [`CloBiT/prokbert-mini-clostridium-promoter`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter) | CC-BY-NC-4.0 |
| [`CloBiT/cnn-fusion-clostridium-promoter`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter) | CC-BY-4.0 |
| [`CloBiT/evo2-logreg-clostridium-promoter-probe`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe) | CC-BY-4.0 for the probe; Evo2 backbone separately licensed |
| [`CloBiT/dh5alpha-promoter-strength-svr`](https://huggingface.co/CloBiT/dh5alpha-promoter-strength-svr) | MIT for the reproduced fitted artifact/project inference; upstream Wang MIT notice retained |


ProkBERT's external tokenizer/model code separately declares CC-BY-NC-ND-4.0. [RegulonDB data terms](../REGULONDB_LICENSE.md) apply to the genome-mining packages, which contain fitted artifacts/helpers rather than large training exports or Evo2 backbone weights. The DH5α package instead uses Wang's small published table and its retained notice. [Release licences](../release/huggingface/README.md#licences) distinguish these sources.

**ProkBERT:** pin tokenizer files and code revisions separately; the inspected revisions below are not proven historical-training revisions. max_length counts tokens, not bases.

```python
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "CloBiT/prokbert-mini-clostridium-promoter"
tok = AutoTokenizer.from_pretrained(
    "neuralbioinfo/prokbert-mini", trust_remote_code=True,
    revision="feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc",
    code_revision="366d9336b6b89d7c1feddf56d9b01be8065ccc43")
model = AutoModelForSequenceClassification.from_pretrained(
    REPO, trust_remote_code=True, revision="815f637e1a77ac25757b0add21a22668090cce31").eval()
with open("results/candidates/g131795_tier1_canonical_seq.tsv") as source:
    seqs = [row["summit_seq"] for row in csv.DictReader(source, delimiter="\t")][:10]
enc = tok(seqs, padding="max_length", truncation=True, max_length=81, return_tensors="pt")
with torch.no_grad():
    scores = torch.softmax(model(**enc).logits.float(), dim=1)[:, 1]
print(scores)  # uncalibrated positive-class scores
```

**CNN-v2:** reconstruct the recorded features and ENAC width.

```python
import csv, json, sys, torch
from huggingface_hub import snapshot_download

path = snapshot_download("CloBiT/cnn-fusion-clostridium-promoter",
                         revision="191fc88fffeb52ecbf4bdb27702a734365545815")
sys.path.insert(0, path)
from train_cnn_v2 import build_model_from_config, predict_proba

with open(f"{path}/config.json") as fh:
    cfg = json.load(fh)
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(f"{path}/best_model.pt", map_location="cpu", weights_only=True))
with open("results/candidates/g131795_tier1_canonical_seq.tsv") as source:
    seqs = [row["summit_seq"] for row in csv.DictReader(source, delimiter="\t")][:10]
scores = predict_proba(model, seqs, device="cpu",
                       enac_window=int(cfg.get("enac_window", 5)))
print(scores)  # uncalibrated positive-class scores, not measured promoter probabilities
```

**Evo2:** DNA inference needs the compatible CUDA stack; saved-feature evaluation needs only the numerical probe stack. The committed trusted pickle records scikit-learn 1.9.0.

```bash
INPUT_FASTA="$PWD/results/candidates/g131795_first10_canonical_named.fasta"
hf download CloBiT/evo2-logreg-clostridium-promoter-probe \
  --revision 6e521c874dd61517df9b17a56f2a75847a054b87 --local-dir evo2_probe
cd evo2_probe
python apply_probe.py --fasta "$INPUT_FASTA" --out scores.csv
```


### DH5α promoter-core activity

We reproduced Wang et al.'s 88-record RBF SVR and packaged the fitted support vectors, coefficients, intercept and gamma as safe NPZ/JSON. Its input is 50-bp position-major T/C/G/A one-hot DNA, and its output is log₂ activity converted back to training-table AU. The first-ten analysis crops `summit_seq[10:60]` from already-oriented 81-bp candidates and compares their predictions with the same-model `pptb_fU` core.

```bash
hf download CloBiT/dh5alpha-promoter-strength-svr \
  --revision 0c5202892f4255eb8eb17493bed8d51e04f8cfc4 --local-dir dh5alpha_svr
uv run --no-project --with numpy==2.5.3 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --model dh5alpha_svr --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv --top-n 10 --tss-offset 60 \
  --out-prefix dh5alpha_first10
```

The public package includes the exact reference, the input of promoters with both −10 and −35 motifs, first-ten predictions and numerical verification. Cached inference needs no scikit-learn, GPU or service. Its selection-CV R² of 0.266090 is not candidate accuracy, and its 6.927284–35.097146 AU reference-start scenarios are not confidence intervals. DL10 in the current shortlist means `cand_00101` / `P_prfB`, not the independent `pptb_fU` reference. [The strength guide](ECOLI_STRENGTH_PREDICTIONS.md) provides complete results and source assay context.
### Local checkpoints

Default root is `workflow/results/`. The mining jobs use this location directly; PROMOTER_RESULTS overrides alone do not relocate every output.

```text
$PROMOTER_RESULTS/T21_xs/prokbert/best_model
$PROMOTER_RESULTS/T21_xs/cnn_v2/best_model.pt       # config.json alongside
$PROMOTER_RESULTS/regimeC/evo2/evo2_probe_xs.pkl
```

| Member | Recovered local evidence |
|---|---|
| ProkBERT | `best_model/{model.safetensors,config.json,training_args.bin,models.py,README.md}` |
| CNN-v2 | `best_model.pt`, `config.json`, `metrics.json`, `history.json`, `train_cnn_v2.py`, `README.md` |


These CNN/ProkBERT files are ignored by git. The deployed Evo2 probe is committed at `pipeline/training/Evo2/evo2_promoter_classifier.pkl`.

To stage public package contents into these CLI paths, run from the repository root after sourcing `env.sh`:

```bash
hf download CloBiT/prokbert-mini-clostridium-promoter \
  --revision 815f637e1a77ac25757b0add21a22668090cce31 \
  --local-dir "$PROMOTER_RESULTS/T21_xs/prokbert/best_model"
hf download CloBiT/cnn-fusion-clostridium-promoter \
  --revision 191fc88fffeb52ecbf4bdb27702a734365545815 \
  --local-dir "$PROMOTER_RESULTS/T21_xs/cnn_v2"
```

These are existing public revisions, not retrained or repaired weights. Inspect downloaded executable code before loading; tokenizer files/code still use the separate pins above.

The local FASTA CLI scores one neural classifier, not Evo2 or the ensemble. Its legacy id,prob CSV contains raw positive-class scores:

```bash
source env.sh
conda activate "$ENV_MAIN"
python pipeline/mining/score_promoters.py --model prokbert --fasta results/candidates/g131795_first10_canonical_named.fasta --out /tmp/prokbert_scores.csv --models-dir "$PROMOTER_RESULTS/T21_xs"
python pipeline/mining/score_promoters.py --model cnn_v2 --fasta results/candidates/g131795_first10_canonical_named.fasta --out /tmp/cnn_v2_scores.csv --models-dir "$PROMOTER_RESULTS/T21_xs"
```

Without `--models-dir` it expects models_deployed; prefer the explicit flag. If intentionally creating that default, preserve unrelated existing files:

```bash
mkdir -p "$PROMOTER_RESULTS/models_deployed"
ln -s ../T21_xs/prokbert "$PROMOTER_RESULTS/models_deployed/prokbert"
ln -s ../T21_xs/cnn_v2 "$PROMOTER_RESULTS/models_deployed/cnn_v2"
```

The local release directory lacks evo2_common.py, so add its source directory when using the probe helper:

```bash
conda activate "$ENV_EVO2"
PYTHONPATH="$PROMOTER_REPO/pipeline/training/Evo2${PYTHONPATH:+:$PYTHONPATH}" python \
  "$PROMOTER_REPO/release/huggingface/evo2-logreg-clostridium-promoter-probe/apply_probe.py" \
  --fasta windows.fna --probe "$PROMOTER_REPO/pipeline/training/Evo2/evo2_promoter_classifier.pkl" \
  --out scores.csv
```

### Verified original data and encodings

Original recovery under `~/cb-promoter-original-run/` is separate from Hub checkpoint recovery. The recorded alignment check covers all 50,190 rows:

| Split | Rows | Positive labels |
|---|---:|---:|
| Train | 44,085 | 24,001 |
| Validation | 3,052 | 547 |
| Test | 3,053 | 547 |


T16_xs preserves row order and all ten metadata fields. ProkBERT/NT IDs and masks regenerate exactly, with stored widths 78/82. CNN (N,81,4) arrays reproduce source DNA; labels/weights/clusters match. Split cluster-ID intersections are 0/0/0. Verification used Python 3.12.12, Transformers 5.3.0, NumPy 2.5.4, pandas 3.0.6 and PyArrow 26.0.0; NT tokenizer revision c0f0359229f36ff6bc3a021247eefe0a9c344bd1 and ProkBERT revisions from the example above.

The 159-file SuperPod transfer passed source hashes:

| Recovered input | Path under the local recovery root |
|---|---|
| Current-target three-member tracks | `results/regimeC/mine3_g131795/aligned_3m/` |
| Paired benchmark three-member tracks | `results/regimeC/mine3_clj/aligned/`, `results/regimeC/mine3_cau/aligned_3m/` |
| Evo2 train/validation/test features, labels and weights | `evo2_embeddings/{train,val,test}_{X,y,w}.npy` |
| Source manifest and extraction log/script snapshots | `verification/superpod-source/` |
| Machine-readable provenance, hashes and verification results | `verification/recovery.json` |


Saved float32 Evo2 features have 4096 columns and matching labels/weights. The untouched test cohort has 3,053 rows/547 positives; the committed probe reproduced test AP 0.70258010/AUROC 0.89534660 with NumPy 2.5.4, scikit-learn 1.9.0 and joblib 1.5.3. Recovered test weights are all 1, so weighted and unweighted test AP/AUROC coincide; evaluation origins differ. Validation is part of the final refit, not held out. Aligned ordered window counts are target 179,172, C. ljungdahlii 176,961 and C. autoethanogenum 169,343. Extraction records identify job 516257, Evo2-7B and mean pooling, not the exact historical code/layer.

Recovered original-run ProkBERT validation/test AP is 0.8080683612857302/0.8249372466065321; CNN-v2 is 0.7380835045313733/0.7255828423854476. Their losses use weights, whereas these AP values are unweighted; evaluation origins differ. These are historical metrics, not remeasurements. Saved-feature evaluation does not audit new GPU extraction or benchmark scanning. Original T22_xs predictions and optional calib_xs maps/cohort remain unavailable; raw scoring/ranks do not need them.

### Verified recovered-checkpoint check

Both classifiers previously scored all 45 promoters with both −10 and −35 motifs on CPU with Python 3.13.5, torch 2.11.0, Transformers 5.3.0, NumPy 2.5.3 and scikit-learn 1.9.0. IDs/order were preserved; helper-versus-six-decimal-CSV difference was at most 4.92e-7. ProkBERT fixed (45,81) versus longest (45,78) padding gave identical scores on those inputs, not a general padding guarantee.

From the repository root:

```bash
source env.sh
# Use an already activated compatible Python environment.
CHECK_DIR="$PROMOTER_RESULTS/checks/canonical45"
mkdir -p "$CHECK_DIR"
python - "$CHECK_DIR/canonical45.fasta" <<'PY'
import csv
import sys
with open("results/candidates/g131795_tier1_canonical_seq.tsv") as source:
    rows = list(csv.DictReader(source, delimiter="\t"))
with open(sys.argv[1], "w") as output:
    for row in rows:
        output.write(f">{row['candidate_id']}\n{row['summit_seq']}\n")
PY
python pipeline/mining/score_promoters.py --model prokbert --models-dir "$PROMOTER_RESULTS/T21_xs" \
  --fasta "$CHECK_DIR/canonical45.fasta" --out "$CHECK_DIR/prokbert_scores.csv"
python pipeline/mining/score_promoters.py --model cnn_v2 --models-dir "$PROMOTER_RESULTS/T21_xs" \
  --fasta "$CHECK_DIR/canonical45.fasta" --out "$CHECK_DIR/cnn_v2_scores.csv"
```

This checks loading, framing and finite scores. It does not recompute held-out AP, establish promoter activity or repair labels.

## Model selection

| Ensemble | *C. ljungdahlii* @0.1 / @0.5 | *C. autoethanogenum* @0.1 / @0.5 |
|---|---|---|
| 3 members | 74.4% / 61.6% | 87.0% / 76.6% |
| + CNN-v1 | 71.4% / 58.3% | 86.5% / 74.5% |


The trio led at recall 0.5 on both restricted benchmarks; paired Evo2 led slightly at recall 0.1 in C. autoethanogenum. Complementary representations motivated testing, not a measured claim of maximally decorrelated errors.

Historical alternatives were one-hot CNN-v1, NT-v2 query/value LoRA and NTv3 384-bp-context full fine-tuning. NTv3's high reported AP has unresolved context-overlap/provenance limits; an unmasked token-reconstruction or constant-only probe record is not classifier evidence. The off-the-shelf prokbert-mini-promoter baseline is separate from our prokbert-mini training base. Cluster-disjoint CV code exists, but completed CV output is not supplied.

### NTv3 context-length requirement and extraction

The provider's [architecture guide](https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer_v3.md) requires multiples of 128 bp for the main seven-downsample models. The supervised trial chose **384 bp = 3×128**, not the only supported length. The [launcher](../pipeline/training/nucleotide-transformer-v3/current/finetune_ntv3.sbatch) selects `sequence_context`; [context construction](../pipeline/datasets/positives-with-context/build_context_sources.py) uses published coordinates or locates source DNA in its matched genome, then extracts contiguous strand-oriented flanks around the same 81-bp core (151/152 flanking positions). Missing flanks and sequence-only sources use N-padding, which supplies length, not biological context.

This construction method does not certify all unavailable historical context rows or resolve context-overlap leakage. NTv3's scanner reports 81-bp core coordinates despite reading longer context, so common coordinates—not identical input lengths—are the ensemble alignment requirement. A longer-context Evo2 experiment would require a compatible probe and renewed evaluation; no evidence establishes longer inputs as better here.

The dataset retains 888 incompatible-assembly B. licheniformis positives; all deployed weights remain affected. [Pipeline provenance](PIPELINE.md#genome-assemblies-and-coordinate-provenance) explains the correction/retraining boundary. Target exclusion is supervised only; backbone pretraining exposure is unknown. Motif/rank outputs require biological testing.
