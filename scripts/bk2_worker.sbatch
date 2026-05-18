#!/bin/bash
#SBATCH --job-name=mariha-bk2
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/bk2_%x_%A_%a.out
#SBATCH --error=logs/bk2_%x_%A_%a.err

# One array task = one run_id. JOB_LIST, SUBJECT, AGENT, CL_METHOD, RUN_PREFIX
# are exported by submit_bk2_combo.sh via --export. Modules, venv and the
# MARIHA_* paths all come from the setup_hpc.sh env file. CPU-only: BK2
# generation is greedy-policy inference, no GPU needed.
MARIHA_ENV_FILE="${MARIHA_ENV_FILE:-$HOME/.config/mariha/hpc_env.sh}"
if [[ ! -f "$MARIHA_ENV_FILE" ]]; then
    echo "ERROR: MariHA HPC env file not found at $MARIHA_ENV_FILE" >&2
    exit 1
fi
source "$MARIHA_ENV_FILE"

RUN_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST")

echo "=== Task ${SLURM_ARRAY_TASK_ID}: ${AGENT}${CL_METHOD:+_$CL_METHOD} | $SUBJECT | $RUN_PREFIX | $RUN_ID ==="

ARGS=(
    --subject "$SUBJECT"
    --agent "$AGENT"
    --run_prefix "$RUN_PREFIX"
    --run_id "$RUN_ID"
    --experiment_dir "$MARIHA_EXPERIMENT_DIR"
)
if [[ -n "${CL_METHOD:-}" ]]; then
    ARGS+=(--cl_method "$CL_METHOD")
fi

# generate_model_bk2.py writes into $MARIHA_DATA_ROOT/mario.scenes.{combo}/.
python "$MARIHA_REPO/scripts/generate_model_bk2.py" "${ARGS[@]}"
