# Shared HPC4 environment for the promoter-mining pipeline.
# Source this at the top of every job / interactive session, THEN conda-activate the
# right env (promoter-mining-hpc4 for everything except NT; nt-tf4x for NT):
#
#   source /project/igem2026/cb-promoter-mining/env.sh
#   conda activate /project/igem2026/envs/promoter-mining-hpc4   # or .../nt-tf4x for NT
#
# One source of truth so nobody hand-copies (and fat-fingers) the HF variables.

# Shared, group-readable HuggingFace cache (models + trust_remote_code modules), offline.
# NOTE: /project/igem2026/hf_cache is a teammate's PRIVATE (mode-700) dir — do NOT use it.
# hf_shared is the group-readable one; rebuild with:  cp -a ~/hf_cache/. /project/igem2026/hf_shared/
export HF_HOME=/project/igem2026/hf_shared
export HF_HUB_OFFLINE=1
export HF_TOKEN_PATH=/tmp/hf_notoken_$USER   # steer HF away from any token file (avoids the shared-token PermissionError)
