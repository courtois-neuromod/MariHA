#!/bin/bash
# Submit one SLURM array job per (run_label, run_prefix), where each array
# task handles a chunk of run_ids so that multiple subjects can be queued
# simultaneously without hitting CC's per-user task limit.
#
# Usage:
#   ./scripts/submit_bk2_all.sh --agent dqn [options]
#
# Options:
#   --agent        <dqn|ppo|sac>        (required)
#   --subject      <sub-01>             (default: sub-01)
#   --run_prefix   <prefix>             only this prefix (default: all found)
#   --experiment_dir <path>             (default: $SCRATCH/MariHA/experiments)
#   --output_root  <path>               output root for BK2 files (default: same as experiment_dir)
#   --cl_method    <method>             only submit for this CL method (default: all found)
#   --chunk_size   <N>                  run_ids per array task (default: 12)
#   --dry-run                           print job list, do not submit

set -euo pipefail

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
AGENT=""
SUBJECT="sub-01"
FILTER_PREFIX=""
FILTER_CL_METHOD=""
DRY_RUN=false
CHUNK_SIZE=12
EXPERIMENT_DIR="${MARIHA_EXPERIMENT_DIR:-${SCRATCH:-$HOME}/MariHA/experiments}"
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)         AGENT="$2";            shift 2 ;;
        --subject)       SUBJECT="$2";          shift 2 ;;
        --run_prefix)    FILTER_PREFIX="$2";    shift 2 ;;
        --cl_method)     FILTER_CL_METHOD="$2"; shift 2 ;;
        --experiment_dir) EXPERIMENT_DIR="$2";  shift 2 ;;
        --output_root)   OUTPUT_ROOT="$2";      shift 2 ;;
        --chunk_size)    CHUNK_SIZE="$2";       shift 2 ;;
        --dry-run)       DRY_RUN=true;          shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$AGENT" ]]; then
    echo "Usage: $0 --agent <dqn|ppo|sac> [--subject sub-01] [--run_prefix PREFIX] [--chunk_size 12] [--dry-run]" >&2
    exit 1
fi

REPO="${MARIHA_REPO:-$HOME/GitHub/MariHA}"
CHECKPOINT_ROOT="$EXPERIMENT_DIR/checkpoints/$SUBJECT"
DATA_ROOT="${MARIHA_DATA_ROOT:-${SCRATCH:-$HOME}/MariHA/data}"
SCENES_ROOT="$DATA_ROOT/mario.scenes"

# --------------------------------------------------------------------------- #
# Validate directories
# --------------------------------------------------------------------------- #
if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "ERROR: checkpoint root not found: $CHECKPOINT_ROOT" >&2
    exit 1
fi
if [[ ! -d "$SCENES_ROOT/$SUBJECT" ]]; then
    echo "ERROR: subject data not found: $SCENES_ROOT/$SUBJECT" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Collect run_ids from data directory (canonical chronological order)
# --------------------------------------------------------------------------- #
mapfile -t RUN_IDS < <(
    find "$SCENES_ROOT/$SUBJECT" -name "*_desc-scenes_events.tsv" \
        | grep -oP 'ses-\d+_.*_run-\d+' \
        | sed 's/_task-mario//' \
        | sort -u
)

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no run_ids found under $SCENES_ROOT/$SUBJECT" >&2
    exit 1
fi
echo "Found ${#RUN_IDS[@]} run_ids for $SUBJECT."

# --------------------------------------------------------------------------- #
# Submit one array per (run_label, run_prefix); each task handles CHUNK_SIZE
# run_ids so total tasks = ceil(N_RUN_IDS / CHUNK_SIZE) per array.
# --------------------------------------------------------------------------- #
mkdir -p "$REPO/logs"

N_RUN_IDS=${#RUN_IDS[@]}
N_CHUNKS=$(( (N_RUN_IDS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

# Wall time: 90 min per run_id in the chunk, rounded up to the nearest hour.
WALL_HOURS=$(( (CHUNK_SIZE * 90 + 59) / 60 ))
[[ $WALL_HOURS -gt 24 ]] && WALL_HOURS=24
WALL_TIME="${WALL_HOURS}:00:00"

echo "Chunk size: ${CHUNK_SIZE} run_ids/task → ${N_CHUNKS} tasks per array (wall time: ${WALL_TIME})"

TOTAL_ARRAYS=0

for rl_dir in "$CHECKPOINT_ROOT/${AGENT}" "$CHECKPOINT_ROOT/${AGENT}_"*/; do
    [[ -d "$rl_dir" ]] || continue
    run_label="$(basename "$rl_dir")"

    # Extract cl_method from run_label
    if [[ "$run_label" == "${AGENT}_"* ]]; then
        cl_method="${run_label#${AGENT}_}"
    else
        cl_method=""
    fi

    # Honour --cl_method filter if given
    if [[ -n "$FILTER_CL_METHOD" && "$cl_method" != "$FILTER_CL_METHOD" ]]; then
        continue
    fi

    # Collect unique run_prefixes (strip _taskN suffix)
    mapfile -t PREFIXES < <(
        ls "$rl_dir" \
            | grep -oP '^.*(?=_task\d+$)' \
            | sort -u
    )

    if [[ ${#PREFIXES[@]} -eq 0 ]]; then
        echo "WARNING: no task checkpoints found under $rl_dir — skipping." >&2
        continue
    fi

    for prefix in "${PREFIXES[@]}"; do
        # Honour --run_prefix filter if given
        if [[ -n "$FILTER_PREFIX" && "$prefix" != "$FILTER_PREFIX" ]]; then
            continue
        fi

        # Write a job list of 84 run_ids for this (run_label, prefix)
        # Use a deterministic name so the file is stable across job start delays.
        JOB_LIST="$REPO/logs/bk2_${run_label}_${prefix}.txt"
        printf "%s\n" "${RUN_IDS[@]}" > "$JOB_LIST"

        if $DRY_RUN; then
            echo "[dry-run] $run_label | $prefix → array 1-${N_CHUNKS} (${CHUNK_SIZE} run_ids/task, job list: $JOB_LIST)"
            continue
        fi

        JOB_NAME="bk2-${run_label}"
        sbatch \
            --job-name="$JOB_NAME" \
            --array="1-${N_CHUNKS}" \
            --time="$WALL_TIME" \
            --export="ALL,JOB_LIST=$JOB_LIST,CHUNK_SIZE=$CHUNK_SIZE,REPO=$REPO,EXPERIMENT_DIR=$EXPERIMENT_DIR,OUTPUT_ROOT=$OUTPUT_ROOT,SUBJECT=$SUBJECT,RUN_LABEL=$run_label,AGENT=$AGENT,CL_METHOD=$cl_method,RUN_PREFIX=$prefix" \
            "$REPO/scripts/bk2_worker.sbatch"

        echo "Submitted array 1-${N_CHUNKS} for $run_label | $prefix"
        (( TOTAL_ARRAYS++ )) || true
    done
done

if $DRY_RUN; then
    echo "(not submitting — pass without --dry-run to submit)"
else
    echo "Done. Submitted $TOTAL_ARRAYS array job(s) of ${N_CHUNKS} tasks each (${CHUNK_SIZE} run_ids/task)."
fi
