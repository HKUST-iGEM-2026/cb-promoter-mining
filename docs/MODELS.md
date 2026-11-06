# Models

The models answer two questions in sequence. First, three classifiers rank promoter-like DNA so the miner can select genomic regions worth testing. Second, an independent DH5α-trained SVR estimates reporter activity from selected 50-bp cores. Recognition and expression are not the same target, so the SVR is not a fourth ensemble member or a transformation of classifier scores. [Results](RESULTS.md) follows the evidence from held-out classification through genomic ranking to conditional activity predictions.

## The deployment ensemble

| Member | Implementation | Deployment role |
|---|---|---|
| **ProkBERT-mini** | full fine-tune of `neuralbioinfo/prokbert-mini` with a two-class head | prokaryote-pretrained transformer |
| **CNN-v2 fusion** | from-scratch one-hot/ENAC convolution branches plus dinucleotide/trinucleotide frequency branches | local and composition features |
| **Evo2-7B probe** | logistic regression on frozen 4096-dimensional embeddings | strongest standalone member in the reported restricted benchmarks |


The three deployed members score the same **81-bp transcription-oriented DNA**. Common framing makes their tracks coordinate-compatible, but their raw score scales differ. The miner therefore intersects step-10 tracks on contig/start/strand, converts each member's scores to average-tie percentiles in that shared pool, and takes their mean. This measures relative ranking, not probability, strength or unanimity. No calibrator is used; only strictly increasing score-only transforms preserve ranks without changing ties.

### Why the discovery workflow uses 81 bp

Published-TSS extraction takes **60 upstream bases, the TSS base and 20 downstream bases**, placing the start at zero-based offset 60. This 81-bp convention covers the upstream recognition region and immediate start-site context and keeps data sources, classifiers and scans on one frame. The [extractor](../pipeline/acquisition/tss-literature/extract_published_tss.py) explicitly matches the existing reference-promoter framing; the [splitter](../pipeline/datasets/clustering-and-splits/cdhit_splits.py) normalises other sources to the same length. Silver windows remain gene-anchored putative positives, and cropping/padding curated sequences does not locate their starts.

**ProkBERT is not restricted to 81 bp.** The [original paper](https://doi.org/10.3389/fmicb.2023.1331233) uses 81-bp promoter examples as a dataset convention and describes longer sequence inputs; the [base-model card](https://huggingface.co/neuralbioinfo/prokbert-mini) advertises approximately 1-kb context. The recovered checkpoint config records `max_position_embeddings=1024`, `kmer=6` and `shift=1`. Overlapping six-mers turn 81 bases into **76 sequence tokens plus CLS/SEP: 78 tokens**, not 81.

A CPU smoke run of that checkpoint accepted untruncated 81-, 161- and 257-bp sequences, producing 78, 158 and 254 tokens and completing classification loss/backpropagation for each. No optimiser step or checkpoint write was performed. This disproves an 81-bp architectural/training-path limit; it does not establish accuracy on longer sequences. No retained window-length comparison establishes 81 bp as optimal, and changing the deployed frame requires compatible data, encodings and evaluation.

### Architectures

Each representation offers a different input to the same classification task. ProkBERT learns token relationships; CNN-v2 combines local patterns and composition; Evo2 supplies a frozen embedding to a small fitted probe. The settings below identify the retained fits, rather than implying that a larger backbone or a more elaborate representation must perform better.

ProkBERT has six layers/six heads, hidden size 384, overlapping six-base tokens and padding-masked learned attention pooling before its two-class head. The recovered config records float32 and Transformers 5.3.0. Saved training arguments confirm learning rate 4.675283456079137e-05, train/eval batches 16/64, 10 maximum epochs, weight decay 0.07894770663194649 and seed 42. Warmup ratio 0.11090738634337446 is stored in float warmup_steps; warmup_ratio is None. Saved training uses fused AdamW, linear schedule, epoch evaluation/saving and bf16; CPU inference uses float32.

CNN-v2 combines one-hot/ENAC convolution branches (kernels 7/5/3, widths 64/128/256, batch normalization/pooling) with 16-d dinucleotide and 64-d trinucleotide projections. Concatenation order is onehot,dinuc,trinuc,enac; fusion dimension is 256. A/C/G/T channels, all-position k-mer denominators and clipped centered ENAC width 5 define the features. N adds no base-channel count; inputs are center-cropped or N-padded to 81 bp. Recorded training uses learning rate 0.0003748224111698842, batch 32, dropout 0.10382857322260322, weight decay 0.00024777455052193346, 30 epochs, patience 8 and seed 42.

Evo2 uses a trusted StandardScaler→LogisticRegression probe with 4096 features, classes [0,1] and scaler count 47,137. Current DNA inference mean-pools blocks.28.mlp.l3 from evo2_7b; the exact historical extraction revision/layer is not established. Backbone weights stay frozen and are obtained separately.

**Evo2 input length:** the retained completed extraction job `516257` explicitly logs `col='sequence'` for 44,085 training, 3,052 validation and 3,053 test rows. Inspection of the actual three input Parquets finds every `sequence` is 81 bp and no `sequence_context` column. The deployed frozen-embedding probe therefore did **not** stitch examples together or append longer genomic flanks. The extractor's optional `--sequence-col sequence_context` supports a different experiment; its existence does not show it was used. The 384-bp context trial was NTv3. This establishes the deployed input frame, not every historical extraction revision or the backbone's pretraining context.

## Loading

### Public Hugging Face repositories

CNN-v2, ProkBERT, the Evo2 probe and the DH5α SVR are available from public Hugging Face repositories; project-package downloads do not require an authorised account or HF_TOKEN. Released weight hashes match the recovered local copies. Review executable remote code before trusting it, and follow any separate access or licence conditions for upstream backbones. The fitted DH5α NPZ is also available in `release/huggingface/dh5alpha-promoter-strength-svr` in this source tree. See [compatible setup](SETUP.md).

| Repository | Model-card license |
|---|---|
| [`CloBiT/prokbert-mini-clostridium-promoter`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter) | CC-BY-NC-4.0 |
| [`CloBiT/cnn-fusion-clostridium-promoter`](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter) | CC-BY-4.0 |
| [`CloBiT/evo2-logreg-clostridium-promoter-probe`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe) | CC-BY-4.0 for the probe; Evo2 backbone separately licensed |
| [`CloBiT/dh5alpha-promoter-strength-svr`](https://huggingface.co/CloBiT/dh5alpha-promoter-strength-svr) | MIT for the reproduced fitted artifact/project inference; upstream Wang MIT notice retained |


ProkBERT's external tokenizer/model code separately declares CC-BY-NC-ND-4.0. [RegulonDB data terms](../REGULONDB_LICENSE.md) apply to the genome-mining packages, which contain fitted artifacts/helpers rather than large training exports or Evo2 backbone weights. The DH5α package instead uses Wang's small published table and its retained notice. [Release licences](../release/huggingface/README.md#licences) distinguish these sources.

**ProkBERT:** pin tokenizer files and code revisions separately; the inspected revisions below are not proven historical-training revisions. `max_length` counts **tokens**, not DNA bases: this example pads a clean 81-bp input's 78 tokens to width 81, with three masked padding positions. It does not impose an 81-bp limit on the backbone. Keep the deployed input frame even though the architecture can accept longer sequences.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "CloBiT/prokbert-mini-clostridium-promoter"
tok = AutoTokenizer.from_pretrained(
    "neuralbioinfo/prokbert-mini", trust_remote_code=True,
    revision="feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc",
    code_revision="366d9336b6b89d7c1feddf56d9b01be8065ccc43")
model = AutoModelForSequenceClassification.from_pretrained(
    REPO, trust_remote_code=True, revision="815f637e1a77ac25757b0add21a22668090cce31").eval()
seqs = ["A" * 81]  # API illustration only; replace with correctly oriented 81 bp DNA
enc = tok(seqs, padding="max_length", truncation=True, max_length=81, return_tensors="pt")
with torch.no_grad():
    scores = torch.softmax(model(**enc).logits.float(), dim=1)[:, 1]
print(scores)  # uncalibrated positive-class scores
```

**CNN-v2:** reconstruct the recorded features and ENAC width.

```python
import json, sys, torch
from huggingface_hub import snapshot_download

path = snapshot_download("CloBiT/cnn-fusion-clostridium-promoter",
                         revision="191fc88fffeb52ecbf4bdb27702a734365545815")
sys.path.insert(0, path)
from train_cnn_v2 import build_model_from_config, predict_proba

with open(f"{path}/config.json") as fh:
    cfg = json.load(fh)
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(f"{path}/best_model.pt", map_location="cpu", weights_only=True))
scores = predict_proba(model, ["A" * 81], device="cpu",
                       enac_window=int(cfg.get("enac_window", 5)))
print(scores)  # uncalibrated positive-class scores, not measured promoter probabilities
```

**Evo2:** DNA inference needs the compatible CUDA stack; saved-feature evaluation needs only the numerical probe stack. The committed trusted pickle records scikit-learn 1.9.0.

```bash
INPUT_FASTA="$PWD/windows.fna"  # existing oriented input, captured before changing directory
hf download CloBiT/evo2-logreg-clostridium-promoter-probe --local-dir evo2_probe
cd evo2_probe
python apply_probe.py --fasta "$INPUT_FASTA" --out scores.csv
```


### DH5α promoter-core activity

We reproduced Wang et al.'s 88-record RBF SVR and packaged the fitted support vectors, coefficients, intercept and gamma as safe NPZ/JSON. Its input is 50-bp position-major T/C/G/A one-hot DNA, and its output is log₂ activity converted back to **arbitrary units (AU)** on the training table's unspecified numerical activity scale. The first-ten analysis crops `summit_seq[10:60]` from already-oriented 81-bp candidates and compares their predictions with the same-model `pptb_fU` core.

```bash
hf download CloBiT/dh5alpha-promoter-strength-svr \
  --revision 0c5202892f4255eb8eb17493bed8d51e04f8cfc4 --local-dir dh5alpha_svr
uv run --no-project --with numpy==2.5.3 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --model dh5alpha_svr --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv --top-n 10 --tss-offset 60 \
  --out-prefix dh5alpha_first10
```

The public package includes the exact reference, the input of promoters with both −10 and −35 motifs, first-ten predictions and numerical verification. Cached inference needs no scikit-learn, GPU or service. Its selection-CV R² of 0.266090 is not candidate accuracy, and its 6.927284–35.097146 AU reference-start scenarios are not confidence intervals. [The strength guide](ECOLI_STRENGTH_PREDICTIONS.md) provides the complete results and source assay context.
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

The local FASTA CLI scores one neural classifier, not Evo2 or the ensemble. Its legacy id,prob CSV contains raw positive-class scores:

```bash
source env.sh
conda activate "$ENV_MAIN"
python pipeline/mining/score_promoters.py --model prokbert --fasta seqs.fa --out prokbert_scores.csv --models-dir "$PROMOTER_RESULTS/T21_xs"
python pipeline/mining/score_promoters.py --model cnn_v2 --fasta seqs.fa --out cnn_v2_scores.csv --models-dir "$PROMOTER_RESULTS/T21_xs"
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

Model selection depends on the intended use. Historical classification scores alone would favour different choices from the restricted genomic ranking. The table compares the selected trio with adding CNN-v1, so readers can see why the extra member was not retained:

| Ensemble | *C. ljungdahlii* @0.1 / @0.5 | *C. autoethanogenum* @0.1 / @0.5 |
|---|---|---|
| 3 members | 74.4% / 61.6% | 87.0% / 76.6% |
| + CNN-v1 | 71.4% / 58.3% | 86.5% / 74.5% |


The trio led at recall 0.5 on both restricted benchmarks; paired Evo2 led slightly at recall 0.1 in C. autoethanogenum. Complementary representations motivated testing, not a measured claim of maximally decorrelated errors.

Historical alternatives were one-hot CNN-v1, NT-v2 query/value LoRA and NTv3 384-bp-context full fine-tuning. NTv3's high reported AP has unresolved context-overlap/provenance limits; an unmasked token-reconstruction or constant-only probe record is not classifier evidence. The off-the-shelf prokbert-mini-promoter baseline is separate from our prokbert-mini training base. Cluster-disjoint CV code exists, but completed CV output is not supplied.

### NTv3 context-length requirement and extraction

The provider's [NTv3 architecture guide](https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer_v3.md) requires input lengths divisible by 128 for the main seven-downsample models. The supervised context run selected 384 bp (3 × 128), not because 384 is the model's only valid length. Its retained preflight records `T15_xs_ctx` and `max_length=384`; the [classifier launcher](../pipeline/training/nucleotide-transformer-v3/current/finetune_ntv3.sbatch) selects `sequence_context`.

For recoverable loci, [context construction](../pipeline/datasets/positives-with-context/build_context_sources.py) uses published-TSS coordinates or searches the source promoter/background region in its matched genome. It extracts a contiguous, strand-oriented 384-bp window. An 81-bp core has 151/152 flanking positions, reversed in order after reverse complementation; the current extraction function was exercised on actual forward/reverse reference loci and a contig-boundary case. Unavailable flanks are represented by N, including [sequence-only source padding](../pipeline/datasets/clustering-and-splits/npad_context.py). N-padding supplies length, not biological context. This describes the construction method; it does not certify every row of an unavailable original context split or repair the separately identified flawed intermediate.

The deployed Evo2 probe instead selected the 81-bp core column. Keeping the deployed members on one frame controls an input difference, but it is not required for score-track combination: the [NTv3 scanner](../pipeline/training/nucleotide-transformer-v3/current/ntv3_genome_scan.py) itself reads 384 bp and reports 81-bp core coordinates. A longer-input Evo2 experiment would require a compatible probe and renewed ensemble evaluation, not automatically retraining every other member. Neither the current evidence nor model architecture establishes 81 bp as an accuracy optimum.


The dataset retains 888 incompatible-assembly B. licheniformis positives; all deployed weights remain affected. [Pipeline provenance](PIPELINE.md#genome-assemblies-and-coordinate-provenance) explains the correction/retraining boundary. Target exclusion is supervised only; backbone pretraining exposure is unknown. Motif/rank outputs require biological testing.
