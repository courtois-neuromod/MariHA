#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --job-name=mariha-bk2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/bk2_%x_%A_%a.out
#SBATCH --error=logs/bk2_%x_%A_%a.err

# Each array task processes a chunk of run_ids from the job list.
# CHUNK_SIZE, JOB_LIST, RUN_LABEL, AGENT, CL_METHOD, RUN_PREFIX,
# SUBJECT, EXPERIMENT_DIR are exported by submit_bk2_all.sh via --export.
CHUNK_SIZE=${CHUNK_SIZE:-12}
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

mapfile -t CHUNK_RUN_IDS < <(sed -n "${START},${END}p" "$JOB_LIST")

echo "=== Task ${SLURM_ARRAY_TASK_ID}: $RUN_LABEL | $RUN_PREFIX | ${#CHUNK_RUN_IDS[@]} run_ids (lines ${START}-${END}) ==="

module load StdEnv/2023 python/3.12 cmake gcc opencv/4.13.0
export PYTHONPATH="$EBROOTOPENCV/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

FAILED=()
for RUN_ID in "${CHUNK_RUN_IDS[@]}"; do
    echo "--- run_id: $RUN_ID ---"

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

    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
        ARGS+=(--output_root "$OUTPUT_ROOT")
    fi

    if ! python "$REPO/scripts/generate_model_bk2.py" "${ARGS[@]}"; then
        echo "ERROR: generate_model_bk2.py failed for run_id=$RUN_ID" >&2
        FAILED+=("$RUN_ID")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED (${#FAILED[@]}/${#CHUNK_RUN_IDS[@]}): ${FAILED[*]}" >&2
    exit 1
fi

echo "=== Done: all ${#CHUNK_RUN_IDS[@]} run_ids completed successfully ==="
