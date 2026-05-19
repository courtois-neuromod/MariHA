#!/bin/bash
#SBATCH --job-name=mariha-bk2
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/bk2_%x_%A_%a.out
#SBATCH --error=logs/bk2_%x_%A_%a.err

# One array task = one BIDS session (it replays every run of that session and
# writes the session's gamelogs.tar — so a task is the sole writer of its file
# and there are no concurrent writers). JOB_LIST, SUBJECT, AGENT, CL_METHOD,
# RUN_PREFIX are exported by submit_bk2_combo.sh via --export. Modules, venv
# and the MARIHA_* paths all come from the setup_hpc.sh env file. CPU-only:
# BK2 generation is greedy-policy inference, no GPU needed. --time is sized
# for a whole session; bump it for subjects with long sessions.
MARIHA_ENV_FILE="${MARIHA_ENV_FILE:-$HOME/.config/mariha/hpc_env.sh}"
if [[ ! -f "$MARIHA_ENV_FILE" ]]; then
    echo "ERROR: MariHA HPC env file not found at $MARIHA_ENV_FILE" >&2
    exit 1
fi
source "$MARIHA_ENV_FILE"

SESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST")

echo "=== Task ${SLURM_ARRAY_TASK_ID}: ${AGENT}${CL_METHOD:+_$CL_METHOD} | $SUBJECT | $RUN_PREFIX | $SESSION ==="

ARGS=(
    --subject "$SUBJECT"
    --agent "$AGENT"
    --run_prefix "$RUN_PREFIX"
    --session "$SESSION"
    --experiment_dir "$MARIHA_EXPERIMENT_DIR"
)
if [[ -n "${CL_METHOD:-}" ]]; then
    ARGS+=(--cl_method "$CL_METHOD")
fi

# generate_model_bk2.py writes into $MARIHA_DATA_ROOT/mario.scenes.{combo}/.
python "$MARIHA_REPO/scripts/generate_model_bk2.py" "${ARGS[@]}"
