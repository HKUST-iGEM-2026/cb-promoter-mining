# T3 — Snakemake + SLURM scaffold (mock pipeline)

A runnable, end-to-end **mock** of the promoter-mining pipeline. Every task is a
placeholder rule that prints a message and `touch`es its (real-named) output file.
The point is to prove the dependency graph wires up and submits to SLURM correctly
**before** any real code exists. Teammates later replace the `shell:` line of their
own rule with the real command.

## What's in here (and what isn't)

This Snakefile is the **data pipeline only**: Phase B (dataset construction) → C
(training) → D (evaluation) → E (mining). These are the steps where files flow from
one rule's output into the next.

Deliberately *not* modelled as rules — they're real project tasks, but not
data-pipeline steps, so they live in the project plan, not this automation:

- **Infrastructure T1–T4** (HPC4 access, repo, this scaffold, conda env) — one-time
  setup / preconditions.
- **Pre-tasks PT1–PT6** — learning + reusable code (metrics, calibration harness,
  schema). Downstream rules *import* this code; they don't read it as a pipeline file.
- **Phase A T5–T10** — acquisition + curation, done locally one-shot (`local_prep.smk`).

**Preconditions** before running on HPC4: the conda/mamba env (the work of T4) is
deployed and active, and the Phase A starting state is uploaded to `inputs/`.

## Two stages (see context §3)

- **Phase A — produced beforehand, uploaded as the `inputs/` starting state.**
  - **T5–T9 (local, one-shot)** — `local_prep.smk`, run on your own machine.
  - **T10 (TSSpredator) on HPC4** — `tss_prep.smk`, run with the SLURM profile; it is
    CPU-heavy (pulls ~36 GB of reads onto the cluster + aligns), so it does NOT run locally.
- **Phase B onwards — runs on HPC4.** The main `Snakefile` starts at Phase B and
  consumes the Phase A outputs as pre-staged files under `inputs/`.

## Layout

```
snakemake_scaffold/
├── Snakefile                  # HPC4 pipeline: Phase B–E (consumes inputs/)
├── local_prep.smk             # Phase A (T5–T9) mock: run locally, produces inputs/
├── tss_prep.smk               # Phase A (T10/TSSpredator) mock: runs on HPC4 (SLURM profile)
├── config/config.yaml         # placeholder paths/params
└── profiles/slurm/config.yaml # SLURM submission profile (one place for cluster settings)
```

## Run it

1. Produce the Phase A starting state (mocked here):

```bash
cd snakemake_scaffold
snakemake -s local_prep.smk --cores 4                        # T5–T9 (local)
# upload inputs/ to HPC4, then on the cluster:
snakemake -s tss_prep.smk --workflow-profile profiles/slurm  # T10 (HPC4, TSSpredator)
```

2. Run the HPC4 pipeline. Locally to test the wiring:

```bash
snakemake --cores 4            # Phase B–E
snakemake -n                   # dry run: see the plan without executing
```

On HPC4, submit the GPU rules as SLURM jobs:

```bash
snakemake --workflow-profile profiles/slurm
```

If `inputs/` is missing, Snakemake stops with `MissingInputException` — that's the
reminder to upload the Phase A starting state first.

Requires (in the deployed conda env — the precondition from T4):

```bash
pip install snakemake snakemake-executor-plugin-slurm
```

## How the conventions are enforced

All cluster settings live in **one file**, `profiles/slurm/config.yaml`, so nobody
hand-writes `sbatch` scripts:

- **preempt partition** (`slurm_partition: preempt`) — 5× cheaper.
- **preempt hygiene on every job** — `--requeue --signal=B:USR1@120` (the SIGUSR1
  checkpoint trap itself goes in each training script).
- **per-GPU billing** — GPU rules add `--gpus-per-node=1`; we never use `--exclusive`
  (whole-node = 8× cost).

A rule only needs a `resources:` block when it differs from the defaults. GPU rules
(T18–T21, T23, T24, T28) set `slurm_extra=GPU_EXTRA`; everything else inherits the
CPU defaults from the profile.

**HPC4 for necessary tasks only.** Even with the SLURM profile active, only the 7
GPU tasks are submitted as cluster jobs. The other 15 (light tasks) are declared in
the `localrules:` line of the `Snakefile` and run locally on the login/submit node —
they never hit the queue. To move a hybrid task (e.g. T15, T22, T29) onto the
cluster, remove it from that `localrules:` list.

## Filling in a real task

Find your rule in the `Snakefile`. The output filenames are already realistic — keep
them so the next rule's input still matches — and just replace the placeholder
`shell:` line with the real command (the `# real:` comment under each rule shows an
example). E.g. T13:

```python
rule t13_upstream_extraction:
    input:  "results/housekeeping_genes.tsv", "results/operons.tsv"
    output: "results/upstream_regions.fasta"
    shell:  "echo '[T13] upstream region extraction (mock)' && touch {output}"
```

becomes:

```python
rule t13_upstream_extraction:
    input:  housekeeping="results/housekeeping_genes.tsv", operons="results/operons.tsv"
    output: "results/upstream_regions.fasta"
    shell:  "python scripts/extract_upstream.py {input.housekeeping} {input.operons} {output}"
```

Keep your inputs pointing at the upstream rule's real outputs (or, for Phase B rules
that need Phase A data, the `inputs/` files) so the DAG stays connected.
