# Evo2 frozen-embedding probe

Evo2 is deployed alongside ProkBERT-mini and CNN-v2. The backbone stays frozen; we fit a weighted scaler/logistic classifier on 81-bp embeddings. Current extraction mean-pools blocks.28.mlp.l3 from evo2_7b into 4096 float32 features. That recipe does not prove the exact historical revision/layer.

## Training and result

Weighted validation AP selected logistic regression (0.7007 versus RF about 0.61), then the winner was refit on 44,085 train + 3,052 validation rows. Test has 3,053 rows/547 positives. Matching recovered features reproduced weighted **test AP 0.70258010/AUROC 0.89534660**, not new GPU inference. Validation is not held out from the final probe.

| Checkpoint / split | Test AUPRC |
|---|---|
| ProkBERT / NT / CNN-v2 / CNN-v1 (`T21_final`, earlier split) | 0.884 / 0.854 / 0.848 / 0.804 |
| **Evo2 frozen + logreg** | **0.703** |


Earlier task-trained scores use a different plain-0.6 split; they are not an all-model matched comparison. [Historical report](EVO2_GENOME_SCALE_REPORT.md) gives standalone restricted-scan tables. Paired deployment Evo2 values are 69.4/58.4% and 87.6/72.5%, distinct from the standalone run.

## Files

| File | Role |
|---|---|
| extract_embeddings_evo2.py / extract_embeddings.sbatch | Frozen embeddings |
| evo2_promoter_classifier.py | Weighted logreg/RF selection and new refit |
| evo2_promoter_classifier.pkl | Trusted deployed probe, scikit-learn 1.9.0, scaler count 47,137 |
| evo2_common.py | Batched embedding and orientation |
| evo2_scan_restricted.py | Candidate-region scanner |
| C_evo2_ljung_scan.sbatch / C_evo2_cau_scan.sbatch | Historical standalone benchmark jobs |

From the root, [pipeline recipes](../../../docs/PIPELINE.md#the-evo2-probe) construct embeddings, fit a new probe or evaluate the saved one. Create `--out`'s parent directory. Missing weight arrays trigger unweighted fitting; that is not the documented weighted run. Rough train CV is not cluster-aware.

Use `--align-grid` when joining with genome_inference.py; equal stride alone does not align reverse-strand starts. Deployment intersects coordinates before ranking. Step 10 is a cost choice, not an Evo2 limit or lossless-peak guarantee. Scores and mean ranks are not probability, strength or unanimity.

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters. Putative-negative windows are not experimentally established promoter-free DNA.

The benchmark genomes supplied silver positives and putative-negative training windows and have incomplete TSS truth. Deployed training retains 888 wrong-assembly B. licheniformis positives. [Model evidence](../../../docs/MODELS.md#verified-original-data-and-encodings) documents recovered inputs and [setup](../../../docs/SETUP.md) documents CUDA/FlashAttention requirements. New DNA needs a compatible backbone; matching saved features can be evaluated on CPU.
