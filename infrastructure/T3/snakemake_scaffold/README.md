# T3 — Snakemake + SLURM scaffold (mock pipeline)

A runnable, end-to-end **mock** of the promoter-mining pipeline. Every task is a
placeholder rule that prints a message and `touch`es a `results/<task>.done` sentinel.
The point is to prove the dependency graph wires up and submits to SLURM correctly
**before** any real code exists. Teammates later replace the `shell:` line of their
own rule with the real command.

## What's in here (and what isn't)

This `Snakefile` is the **data pipeline**: Phase B (dataset construction) → C
(training) → D (evaluation) → E (mining), plus the **pre-task** rules (PT1–PT6) those
phases depend on. Files flow from one rule's output into the next.

**Not** in this `Snakefile` — real project tasks, but not data-pipeline steps:

- **Infrastructure T1–T4** (HPC4 access, repo, this scaffold, conda env) — one-time
  setup / preconditions; tracked in the project plan, not this automation.
- **Phase A T5–T10** — acquisition + curation, produced beforehand and consumed as
  pre-staged `inputs/`: T5–T9 locally (`local_prep.smk`), T10/TSSpredator on HPC4
  (`tss_prep.smk`).

(PT1–PT6 are conceptually reusable code that downstream rules *import*, but they're kept
here as mock rules anyway, to show the cross-phase dependencies — PT1/PT5 run on GPU.)

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
(PT1, PT5, T18–T21, T23, T24, T28) set `slurm_extra=GPU_EXTRA`; everything else
inherits the CPU defaults from the profile.

**HPC4 for necessary tasks only.** Even with the SLURM profile active, only the 9
GPU/cluster tasks are submitted as SLURM jobs. The other 19 (light tasks) are declared
in the `localrules:` line of the `Snakefile` and run locally on the login/submit node —
they never hit the queue. To move a hybrid task (e.g. T15, T22, T29) onto the
cluster, remove it from that `localrules:` list.

## Filling in a real task

Find your rule in the `Snakefile`. The mock wires the DAG with `results/<task>.done`
sentinels, and the placeholder `shell:` just `touch`es them. To implement, replace the
`shell:` with the `# real:` command shown under the rule. For a rule that passes real
data downstream, also give it a real `output` filename and point the next rule's
`input` at it. E.g. T13:

```python
rule t13_upstream_extraction:
    input:  "results/T11.done", "results/T12.done"
    output: "results/T13.done"
    shell:  "echo '[T13] upstream region extraction (mock)' && touch {output}"
    # real: python scripts/extract_upstream.py {input} {output}   (100-300 bp upstream; gffutils + samtools faidx)
```

becomes:

```python
rule t13_upstream_extraction:
    input:  "results/T11.done", "results/T12.done"
    output: "results/upstream_regions.fasta"
    shell:  "python scripts/extract_upstream.py {input} {output}"
```

(then update the rules that read T13 — e.g. T14, T15 — to consume
`results/upstream_regions.fasta`.) Keep each rule's inputs pointing at its upstream
rule's output (or, for Phase B rules that need Phase A data, the `inputs/` files) so
the DAG stays connected.
