#!/bin/bash
# Submit one SLURM array job (84 tasks) per CL method for a given agent.
# Each array task handles one run_id; all CL methods run in parallel arrays.
#
# Usage:
#   ./scripts/submit_bk2_all.sh --agent dqn [options]
#
# Options:
#   --agent        <dqn|ppo|sac>        (required)
#   --subject      <sub-01>             (default: sub-01)
#   --run_prefix   <prefix>             only this prefix (default: all found)
#   --experiment_dir <path>             (default: $SCRATCH/MariHA/experiments)
#   --dry-run                           print job list, do not submit

set -euo pipefail

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
AGENT=""
SUBJECT="sub-01"
FILTER_PREFIX=""
DRY_RUN=false
EXPERIMENT_DIR="${MARIHA_EXPERIMENT_DIR:-${SCRATCH:-$HOME}/MariHA/experiments}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)         AGENT="$2";            shift 2 ;;
        --subject)       SUBJECT="$2";          shift 2 ;;
        --run_prefix)    FILTER_PREFIX="$2";    shift 2 ;;
        --experiment_dir) EXPERIMENT_DIR="$2";  shift 2 ;;
        --dry-run)       DRY_RUN=true;          shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$AGENT" ]]; then
    echo "Usage: $0 --agent <dqn|ppo|sac> [--subject sub-01] [--run_prefix PREFIX] [--dry-run]" >&2
    exit 1
fi

REPO="${MARIHA_REPO:-$HOME/GitHub/MariHA}"
CHECKPOINT_ROOT="$EXPERIMENT_DIR/checkpoints"
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
    find "$SCENES_ROOT/$SUBJECT" -name "*_events.tsv" \
        | grep -oP 'ses-\d+_.*_run-\d+' \
        | sed 's/_task-mario//' \
        | sort
)

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no run_ids found under $SCENES_ROOT/$SUBJECT" >&2
    exit 1
fi
echo "Found ${#RUN_IDS[@]} run_ids for $SUBJECT."

# --------------------------------------------------------------------------- #
# Submit one array per (run_label, run_prefix) — each array has 84 tasks
# --------------------------------------------------------------------------- #
mkdir -p "$REPO/logs"

N_RUN_IDS=${#RUN_IDS[@]}
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
        JOB_LIST="$(mktemp /tmp/bk2_${run_label}_XXXX.txt)"
        printf "%s\n" "${RUN_IDS[@]}" > "$JOB_LIST"

        if $DRY_RUN; then
            echo "[dry-run] $run_label | $prefix → array 1-${N_RUN_IDS} (job list: $JOB_LIST)"
            continue
        fi

        JOB_NAME="bk2-${run_label}"
        sbatch \
            --job-name="$JOB_NAME" \
            --array="1-${N_RUN_IDS}" \
            --export="ALL,JOB_LIST=$JOB_LIST,REPO=$REPO,EXPERIMENT_DIR=$EXPERIMENT_DIR,SUBJECT=$SUBJECT,RUN_LABEL=$run_label,AGENT=$AGENT,CL_METHOD=$cl_method,RUN_PREFIX=$prefix" \
            "$REPO/scripts/bk2_worker.sbatch"

        echo "Submitted array 1-${N_RUN_IDS} for $run_label | $prefix"
        (( TOTAL_ARRAYS++ )) || true
    done
done

if $DRY_RUN; then
    echo "(not submitting — pass without --dry-run to submit)"
else
    echo "Done. Submitted $TOTAL_ARRAYS array job(s) of ${N_RUN_IDS} tasks each."
fi
