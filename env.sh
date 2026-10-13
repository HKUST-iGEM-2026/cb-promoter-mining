# Environment contract for the promoter-mining pipeline.
#
# Source this once per session or at the top of a job script, then activate the
# conda environment you need:
#
#   source /path/to/cb-promoter-mining/env.sh
#   conda activate "$ENV_MAIN"        # or $ENV_NT / $ENV_EVO2 / $ENV_MEME
#
# Everything below is overridable: export a variable before sourcing and that
# value wins. Nothing here is specific to any one machine or cluster.

# --- Repo root -------------------------------------------------------------
# Resolved from this file's own location, so any clone works.
_env_sh="${BASH_SOURCE[0]:-$0}"
export PROMOTER_REPO="${PROMOTER_REPO:-$(cd "$(dirname "$_env_sh")" && pwd)}"
unset _env_sh

# --- Conda environment names ----------------------------------------------
# Plain names by default, so `conda activate "$ENV_MAIN"` works anywhere. Set
# these to absolute prefixes instead if you install envs in a shared location:
#   export ENV_MAIN=/shared/envs/promoter-mining
export ENV_MAIN="${ENV_MAIN:-promoter-mining}"    # everything except the cases below
export ENV_NT="${ENV_NT:-promoter-nt}"            # Nucleotide Transformer only (not deployed)
export ENV_EVO2="${ENV_EVO2:-promoter-evo2}"      # the Evo2 probe
export ENV_MEME="${ENV_MEME:-promoter-meme}"      # MEME suite / FIMO

# --- Where outputs go ------------------------------------------------------
# Pipeline artifacts (checkpoints, score tracks, candidate tables). Point this
# at scratch or a shared volume if the clone is on a small disk.
export PROMOTER_RESULTS="${PROMOTER_RESULTS:-$PROMOTER_REPO/workflow/results}"

# --- Model cache -----------------------------------------------------------
# Left to the Hugging Face default (~/.cache/huggingface) unless you set it.
# On a cluster whose compute nodes have no internet, point HF_HOME at a shared
# directory, pre-download there from a login node, then set HF_HUB_OFFLINE=1 in
# the job so a cache miss fails fast instead of hanging.
#   export HF_HOME=/shared/hf_cache
#   export HF_HUB_OFFLINE=1
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi

# --- Scheduler (optional) --------------------------------------------------
# Only used if you submit the .sbatch scripts. Leave unset on a cluster that
# needs no account, or if you are not using Slurm at all. The scripts carry no
# site-specific #SBATCH directives; pass what your site requires:
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" job.sbatch

# --- Stop a stray user-site install from shadowing the env -----------------
# A leftover ~/.local numpy or torch takes precedence over the conda env and
# fails with errors like "_ARRAY_API not found". This is the fix.
export PYTHONNOUSERSITE=1
