# REORGANISE_CLAUDE.md — repo restructure spec

Instructions for a Claude Code instance. Goal: make `cb-promoter-mining` legible to someone
with no knowledge of our task numbering, by renaming the `T*`/`PT*` directories to
descriptive names. Read this whole file before running anything.

**Owner to consult on judgement calls: Oak.**

---

## 0. Hard rules

1. **Work on a new branch `reorganise`. Never commit to `main`.**
2. **Do not create new files and do not delete any file.** Allowed operations: create
   directories, move/rename directories and files, remove directories that are left empty.
3. **Do not change file contents except path strings** that point at a directory you moved.
   Rewriting a path is allowed and required. Rewriting anything else (logic, prose, numbers,
   task-table rows) is not.
4. **Prose task IDs stay.** `CLAUDE.md` and the run-books legitimately refer to tasks as
   "T31", "PT5", "T27 FDR". Only change strings that are *filesystem paths*
   (i.e. contain a `/`). `mining-and-motif-annotation/T31/sigma_a_promoter.meme` → rewrite.
   "T31 Motif annotation FIMO/MEME" in the §6 task table → leave exactly as is.
5. Use `git mv` for tracked files so history follows (`git log --follow` keeps working).
   Use plain `mv` only for untracked files.
6. If any step fails or looks ambiguous, **stop and report** rather than improvising.

---

## 1. Facts established before you start (don't re-derive)

- **490 path references** to task-numbered directories exist across **93 tracked files**.
  Biggest: `evaluation-and-calibration/T26` (156), `infrastructure/T3` (98),
  `model-training/T20` (30), `data-acquisition-and-curation/T9` (29).
- Heaviest files to edit: `CLAUDE.md` (33), `infrastructure/T3/snakemake_scaffold/Snakefile`
  (31), `mining-and-motif-annotation/REMINE_g131795_CLAUDE.md` (19), and the
  `evaluation-and-calibration/T26/*_calib.py` + `run_calibration_*.sh` set.
- Root `.gitignore`: no task-numbered paths. Safe, leave alone.
- `data-acquisition-and-curation/T5/.gitattributes` uses a **relative** pattern
  (`Clostridium_genomes_v2_166.zip`), so it survives its parent being renamed. Leave alone.
- No `.github/` CI exists. Nothing to update there.

---

## 2. DO NOT TOUCH

| Thing | Why |
|---|---|
| `results/` anywhere, and every `results/T*` string in code (`results/T15`, `results/T16`, `results/T21_xs`, `results/T21_final_06`, `results/T13`, `results/T22`, `results/T23`, `results/T30`, …) | These are **output artifact directories that also exist on HPC4**, not source. Renaming them desyncs the repo from the cluster and from every cached model checkpoint. ~250 references. Leave every one of them. |
| `venv/`, `.venv/`, `__pycache__/`, `.DS_Store` | Local/ignored junk. |
| `wiki/`, `huggingface_release/`, `positives_with_context/`, `phylogenetic_tree/`, `ntv3_training/` | Untracked or already task-number-free. Out of scope. |
| `bakta_outputs/` | Lives outside this repo entirely. |
| Dataset/model *names* in prose: `T15_xs`, `T21_final_06`, `T16/cnn` | Identifiers, not paths. |

The distinction that matters: `dataset-construction/T15` is **source** → rename it.
`results/T15_xs` is **data** → never touch it.

---

## 3. Preconditions

The working tree is currently dirty: 9 modified tracked files (including `CLAUDE.md`, the
Git-LFS `.zip`/`.xlsx` binaries, and `evaluation-and-calibration/T27/*`) plus many untracked
files. `main` is also 1 commit ahead of `origin/main`.

Do this so the restructure lands as one reviewable commit:

```bash
cd /Users/oak23/Documents/GitHub/cb-promoter-mining
git status                      # confirm what's dirty
git checkout -b reorganise      # dirty tree carries over; main stays untouched
git add -A && git commit -m "wip: in-flight work before restructure"
```

**Do not use `git stash`** here. Popping a stash across ~30 directory renames produces
conflicts on every stashed path.

Rollback at any point: `git reset --hard HEAD` (within the branch) or
`git checkout main && git branch -D reorganise` (abandon entirely).

---

## 4. Phase 1 — rename the directories

Rename parents *before* their children so no empty parent is left behind.

```bash
# pre-tasks -> prototypes  (parent first)
git mv pre-tasks prototypes
git mv prototypes/PT1 prototypes/prokbert-finetune
git mv prototypes/PT3 prototypes/metrics-and-cv
git mv prototypes/PT4 prototypes/calibration-fdr
git mv prototypes/PT5 prototypes/ecoli-sliding-window
git mv prototypes/PT6 prototypes/pipeline-validation

# data acquisition
git mv data-acquisition-and-curation/T5  data-acquisition-and-curation/clostridium-genomes
git mv data-acquisition-and-curation/T6  data-acquisition-and-curation/bakta-annotations
git mv data-acquisition-and-curation/T7  data-acquisition-and-curation/tss-literature
git mv data-acquisition-and-curation/T8  data-acquisition-and-curation/curated-promoters
git mv data-acquisition-and-curation/T9  data-acquisition-and-curation/synthetic-promoters
git mv data-acquisition-and-curation/T10 data-acquisition-and-curation/cdifficile-drnaseq

# dataset construction
git mv dataset-construction/T11 dataset-construction/housekeeping-genes
git mv dataset-construction/T12 dataset-construction/operon-detection
git mv dataset-construction/T13 dataset-construction/upstream-extraction
git mv dataset-construction/T14 dataset-construction/negative-sets
git mv dataset-construction/T15 dataset-construction/clustering-and-splits
git mv dataset-construction/T16 dataset-construction/tokenisation

# model training
git mv model-training/T18 model-training/prokbert
git mv model-training/T19 model-training/nucleotide-transformer
git mv model-training/T20 model-training/cnn
git mv model-training/T21 model-training/hyperparameter-sweep

# evaluation
git mv evaluation-and-calibration/T22 evaluation-and-calibration/metrics
git mv evaluation-and-calibration/T23 evaluation-and-calibration/cross-validation
git mv evaluation-and-calibration/T24 evaluation-and-calibration/external-holdout
git mv evaluation-and-calibration/T25 evaluation-and-calibration/sigma-factor-stratified
git mv evaluation-and-calibration/T26 evaluation-and-calibration/calibration
git mv evaluation-and-calibration/T27 evaluation-and-calibration/empirical-fdr

# mining
git mv mining-and-motif-annotation/T28 mining-and-motif-annotation/genome-scan
git mv mining-and-motif-annotation/T29 mining-and-motif-annotation/ensemble-aggregation
git mv mining-and-motif-annotation/T30 mining-and-motif-annotation/peak-calling
git mv mining-and-motif-annotation/T31 mining-and-motif-annotation/motif-annotation
git mv mining-and-motif-annotation/T32 mining-and-motif-annotation/candidate-table

# infrastructure (T3 is handled separately in Phase 4 — do NOT move it here)
git mv infrastructure/T4 infrastructure/conda-environments
```

Leave alone (already free of task numbers): `data-acquisition-and-curation/TSS_results`,
`dataset-construction/negative-tokenization`, `model-training/Evo2`.
Do **not** case-rename `Evo2` — macOS is case-insensitive and it will corrupt the index.

---

## 5. Phase 2 — fix Git-LFS patterns (CRITICAL, do immediately after Phase 1)

The root `.gitattributes` has **path-anchored** LFS patterns. Two of them break the moment
Phase 1 lands, and if you commit without fixing them, Git-LFS stops intercepting those files
and a **242 MB CSV gets committed as a raw blob into git history** — effectively
unrecoverable without a history rewrite.

Edit `.gitattributes` and change exactly these two lines (this is a path rewrite, so it is
permitted under rule 3):

```diff
-pre-tasks/PT5/*.csv filter=lfs diff=lfs merge=lfs -text
+prototypes/ecoli-sliding-window/*.csv filter=lfs diff=lfs merge=lfs -text
-data-acquisition-and-curation/T7/published_tss/*.xls filter=lfs diff=lfs merge=lfs -text
-data-acquisition-and-curation/T7/published_tss/*.xlsx filter=lfs diff=lfs merge=lfs -text
+data-acquisition-and-curation/tss-literature/published_tss/*.xls filter=lfs diff=lfs merge=lfs -text
+data-acquisition-and-curation/tss-literature/published_tss/*.xlsx filter=lfs diff=lfs merge=lfs -text
```

Leave `*.zip` and `MasterTable.tsv` unchanged — they are not path-anchored.

Verify **before committing**. Each command must print `filter: lfs`:

```bash
git check-attr filter -- prototypes/ecoli-sliding-window/mg1655_promoter_scores.csv
git check-attr filter -- data-acquisition-and-curation/tss-literature/published_tss/hocq2022_cac_tss.xlsx
git check-attr filter -- data-acquisition-and-curation/clostridium-genomes/Clostridium_genomes_v2_166.zip
```

Also record the LFS inventory before and after and confirm the count is unchanged:

```bash
git lfs ls-files | wc -l
```

If any check prints `filter: unspecified`, **stop** — do not commit.

---

## 6. Phase 3 — rewrite the path references

Use `perl`, not `sed`. macOS `sed` lacks `\b`, and the word boundary is what stops
`infrastructure/T3` from also matching `T30`/`T31`/`T32` and `dataset-construction/T1`
from matching `T11`–`T16`.

Apply to **tracked text files only**. Skip binaries, `results/`, and `.git/`.

```bash
cd /Users/oak23/Documents/GitHub/cb-promoter-mining

FILES=$(git ls-files | grep -Ev '\.(zip|xlsx|xls|parquet|png|pdf|pyc|fna|wig|tree)$')

perl -pi -e '
  s{pre-tasks/PT1\b}{prototypes/prokbert-finetune}g;
  s{pre-tasks/PT3\b}{prototypes/metrics-and-cv}g;
  s{pre-tasks/PT4\b}{prototypes/calibration-fdr}g;
  s{pre-tasks/PT5\b}{prototypes/ecoli-sliding-window}g;
  s{pre-tasks/PT6\b}{prototypes/pipeline-validation}g;
  s{data-acquisition-and-curation/T5\b}{data-acquisition-and-curation/clostridium-genomes}g;
  s{data-acquisition-and-curation/T6\b}{data-acquisition-and-curation/bakta-annotations}g;
  s{data-acquisition-and-curation/T7\b}{data-acquisition-and-curation/tss-literature}g;
  s{data-acquisition-and-curation/T8\b}{data-acquisition-and-curation/curated-promoters}g;
  s{data-acquisition-and-curation/T9\b}{data-acquisition-and-curation/synthetic-promoters}g;
  s{data-acquisition-and-curation/T10\b}{data-acquisition-and-curation/cdifficile-drnaseq}g;
  s{dataset-construction/T11\b}{dataset-construction/housekeeping-genes}g;
  s{dataset-construction/T12\b}{dataset-construction/operon-detection}g;
  s{dataset-construction/T13\b}{dataset-construction/upstream-extraction}g;
  s{dataset-construction/T14\b}{dataset-construction/negative-sets}g;
  s{dataset-construction/T15\b}{dataset-construction/clustering-and-splits}g;
  s{dataset-construction/T16\b}{dataset-construction/tokenisation}g;
  s{model-training/T18\b}{model-training/prokbert}g;
  s{model-training/T19\b}{model-training/nucleotide-transformer}g;
  s{model-training/T20\b}{model-training/cnn}g;
  s{model-training/T21\b}{model-training/hyperparameter-sweep}g;
  s{evaluation-and-calibration/T22\b}{evaluation-and-calibration/metrics}g;
  s{evaluation-and-calibration/T23\b}{evaluation-and-calibration/cross-validation}g;
  s{evaluation-and-calibration/T24\b}{evaluation-and-calibration/external-holdout}g;
  s{evaluation-and-calibration/T25\b}{evaluation-and-calibration/sigma-factor-stratified}g;
  s{evaluation-and-calibration/T26\b}{evaluation-and-calibration/calibration}g;
  s{evaluation-and-calibration/T27\b}{evaluation-and-calibration/empirical-fdr}g;
  s{mining-and-motif-annotation/T28\b}{mining-and-motif-annotation/genome-scan}g;
  s{mining-and-motif-annotation/T29\b}{mining-and-motif-annotation/ensemble-aggregation}g;
  s{mining-and-motif-annotation/T30\b}{mining-and-motif-annotation/peak-calling}g;
  s{mining-and-motif-annotation/T31\b}{mining-and-motif-annotation/motif-annotation}g;
  s{mining-and-motif-annotation/T32\b}{mining-and-motif-annotation/candidate-table}g;
  s{infrastructure/T4\b}{infrastructure/conda-environments}g;
' $FILES
```

Then handle bare parent references: search for remaining `pre-tasks/` (not followed by `PT`)
and rewrite to `prototypes/`.

```bash
grep -rn 'pre-tasks' $FILES | grep -v 'prototypes'
```

Fix any hits by hand. Note `prototypes/PT*` must not appear anywhere afterwards.

---

## 7. Phase 4 — `infrastructure/T3` (HIGH BLAST RADIUS — ask Oak first)

`infrastructure/T3/snakemake_scaffold/results/` is the root of **every trained model
checkpoint and output on HPC4**. Paths like
`infrastructure/T3/snakemake_scaffold/results/T21_xs/prokbert/best_model` are consumed by the
mining jobs, the FDR jobs, and the inducible-promoter scripts. 98 references.

Renaming it in the repo is only half the job — the physical directory on the cluster must move
too, or every job breaks with a missing-model error.

**Default: SKIP this phase.** Leave `infrastructure/T3` as-is and note it in the commit
message as deliberately deferred. One opaque directory name is a far smaller cost than an
HPC4 outage mid-sprint.

**If Oak explicitly approves**, then:

```bash
git mv infrastructure/T3 infrastructure/snakemake-pipeline
perl -pi -e 's{infrastructure/T3\b}{infrastructure/snakemake-pipeline}g;' $FILES
```

and the **mandatory** follow-up, run by Oak on the cluster:

```bash
ssh osoo@superpod.ust.hk
cd /project/igem2026/cb-promoter-mining
mv infrastructure/T3 infrastructure/snakemake-pipeline
```

Every teammate with a clone must then re-pull before submitting jobs. Confirm no SLURM jobs
are queued or running first (`squeue -A igem2026`).

---

## 8. Phase 5 — clean up empty directories

Only directories, never files. Scope it tightly so it cannot wander into `.git` or `venv`:

```bash
find . -type d -empty \
  -not -path './.git/*' -not -path './venv/*' -not -path './.venv/*' \
  -not -path './results/*' -print
```

Review the list, then re-run with `-delete` if and only if every entry is genuinely an empty
leftover of a move.

---

## 9. Verification (all must pass before you commit)

```bash
# 1. No task-numbered SOURCE paths remain (should print nothing)
git ls-files | xargs grep -InE '(pre-tasks/PT[0-9]|infrastructure/T4|data-acquisition-and-curation/T[0-9]|dataset-construction/T[0-9]|model-training/T[0-9]|evaluation-and-calibration/T[0-9]|mining-and-motif-annotation/T[0-9])' 2>/dev/null

# 2. results/T* references are UNCHANGED (should still be ~250)
git ls-files | xargs grep -ohE 'results/T[0-9]+[A-Za-z_0-9]*' 2>/dev/null | wc -l

# 3. No file was lost — counts must match pre-restructure
git ls-files | wc -l

# 4. Renames detected as renames, not delete+add
git status --find-renames

# 5. LFS intact (see Phase 2 checks)
git lfs ls-files | wc -l

# 6. Every referenced path actually exists. Extract path-like strings from the
#    shell/python entry points and stat them.
for f in $(git ls-files '*.sh' '*.sbatch'); do
  grep -ohE '(infrastructure|evaluation-and-calibration|model-training|mining-and-motif-annotation|dataset-construction|data-acquisition-and-curation|prototypes)/[A-Za-z0-9_./-]+' "$f"
done | sort -u | while read p; do
  [ -e "$p" ] || echo "MISSING: $p"
done

# 7. Snakemake DAG still wires up — the strongest single check (31 refs live in the Snakefile)
cd infrastructure/T3/snakemake_scaffold && snakemake -n ; cd -
```

Check 7 is the real test. If `snakemake -n` builds the 31-job DAG as before, the path
rewriting is sound. If it fails, diff the failure against the pre-change behaviour before
assuming you broke it — several rules already point at unwritten scripts by design.

---

## 10. Commit

One commit, on `reorganise` only:

```bash
git add -A
git commit -m "Reorganise: rename task-numbered directories to descriptive names

Renames T*/PT* source directories to self-describing names so the repo is
legible without the task brief. Updates the 490 path references that pointed
at them, and the two path-anchored Git-LFS patterns in .gitattributes.

No file contents changed except path strings. Output artifact directories
(results/T15, results/T21_xs, ...) are deliberately untouched: they exist on
HPC4 and renaming them would desync the cluster.

infrastructure/T3 deferred - it is the HPC4 model/results root and renaming it
requires a coordinated move on the cluster."
```

Do **not** push. Leave that to Oak.

---

## 11. Open questions for Oak — report these, don't act on them

1. **`scripts/` at the repo root looks like an older duplicate of `ntv3_training/`.** Seven of
   its twelve files share names with that directory (`evaluate_ntv3.py`,
   `extract_embeddings.py`, `fine_tune_ntv3.py`, `run_eval.sh`, `run_extract.sh`,
   `run_finetune.sbatch`, `train_logistic_regression.py`), and the extras
   (`fine_tune_nt3.py`, `run_nt3.sbatch`, `test_nt3_smoke.sbatch`, `train_classifier.py`,
   `train_nt3_promoters.py`) look like an earlier NT-v3 iteration. They are **not identical**,
   so nothing was deleted. Suggestion: consolidate both under
   `model-training/nucleotide-transformer-v3/` with `current/` and `earlier-iteration/`
   subdirectories — but confirm which is live first. NTv3 was benched, so both may be dead code.
2. **~25 loose files sit at the repo root** — the `g131795_*` / `g409755_*` candidate tables and
   wet-lab spreadsheets, plus `prioritize_candidates.py`, `split_candidate_tiers.py`,
   `filter_and_rank_wetlab.py`, `score_promoters.py`, `main.py`. Gathering the tables into a
   `final-candidates/` directory and the ranking scripts into `mining-and-motif-annotation/`
   would tidy the root considerably. Not done here because several are untracked and it widens
   the diff. Say the word and it's a small follow-up.
3. **`data-acquisition-and-curation/TSS_results`** has no task number but is still vague
   (it holds the B. licheniformis / Ruminiclostridium published-TSS prep). Rename to
   `additional-published-tss`?
4. **`CLAUDE.md` §3 contains a repo-layout tree** that this restructure makes stale. Updating it
   is a content change beyond path strings, so it was left alone. It needs a follow-up edit.
