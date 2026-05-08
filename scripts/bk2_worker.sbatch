#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --job-name=mariha-bk2
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/bk2_%A_%a.out
#SBATCH --error=logs/bk2_%A_%a.err

# Each array task reads its run_id from the job list (one run_id per line).
# RUN_LABEL, AGENT, CL_METHOD, RUN_PREFIX, SUBJECT, EXPERIMENT_DIR are exported
# by submit_bk2_all.sh via --export.
RUN_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST")

echo "=== Task ${SLURM_ARRAY_TASK_ID}: $RUN_LABEL | $RUN_PREFIX | $RUN_ID ==="

module load StdEnv/2023 python/3.12 cmake gcc cuda/12.2 opencv/4.13.0
export PYTHONPATH="$EBROOTOPENCV/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

ARGS=(
    --subject "$SUBJECT"
    --agent "$AGENT"
    --run_prefix "$RUN_PREFIX"
    --run_id "$RUN_ID"
    --experiment_dir "$EXPERIMENT_DIR"
)

if [[ -n "$CL_METHOD" ]]; then
    ARGS+=(--cl_method "$CL_METHOD")
fi

python "$REPO/scripts/generate_model_bk2.py" "${ARGS[@]}"
