#!/bin/bash
# Convert model BK2 files to GIF for specified agents, CL methods, and run prefixes.
# Each SLURM array task processes one model_bk2/{run_id} directory.
# GIFs are written next to the BK2 files.
#
# Edit the CONFIG block below, then run:
#   ./scripts/submit_gif_all.sh [--dry-run]

set -euo pipefail

# =========================================================================== #
# CONFIG — edit these arrays before submitting                                 #
# =========================================================================== #

AGENTS=(dqn)                        # e.g. (dqn ppo sac)
CL_METHODS=("")                     # "" = vanilla; e.g. ("" ewc der agem mas si)
SUBJECTS=(sub-01)                   # e.g. (sub-01 sub-02)
RUN_PREFIXES=()                     # leave empty to match all found prefixes

GIF_FPS=24
GIF_SCALE=256

EXPERIMENT_DIR="${MARIHA_EXPERIMENT_DIR:-${SCRATCH:-$HOME}/MariHA/experiments}"
REPO="${MARIHA_REPO:-$HOME/GitHub/MariHA}"

# =========================================================================== #

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "ERROR: experiment dir not found: $EXPERIMENT_DIR" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Build list of bk2 directories to process                                     #
# Structure: {experiment_dir}/{subject}/{run_label}/{run_prefix}/model_bk2/{run_id}/
# --------------------------------------------------------------------------- #
BK2_DIRS=()

for agent in "${AGENTS[@]}"; do
    for cl_method in "${CL_METHODS[@]}"; do
        if [[ -n "$cl_method" ]]; then
            run_label="${agent}_${cl_method}"
        else
            run_label="$agent"
        fi

        for subject in "${SUBJECTS[@]}"; do
            run_label_dir="$EXPERIMENT_DIR/$subject/$run_label"
            if [[ ! -d "$run_label_dir" ]]; then
                echo "WARNING: not found, skipping: $run_label_dir"
                continue
            fi

            for prefix_dir in "$run_label_dir"/*/; do
                [[ -d "$prefix_dir" ]] || continue
                prefix="$(basename "$prefix_dir")"

                # Skip if a prefix filter is set and this prefix doesn't match
                if [[ ${#RUN_PREFIXES[@]} -gt 0 ]]; then
                    match=false
                    for p in "${RUN_PREFIXES[@]}"; do
                        [[ "$prefix" == "$p" ]] && match=true && break
                    done
                    $match || continue
                fi

                model_bk2_dir="$prefix_dir/model_bk2"
                [[ -d "$model_bk2_dir" ]] || continue

                for run_dir in "$model_bk2_dir"/*/; do
                    [[ -d "$run_dir" ]] || continue
                    # Only include directories that actually have bk2 files
                    compgen -G "${run_dir}*.bk2" > /dev/null 2>&1 || continue
                    BK2_DIRS+=("$run_dir")
                done
            done
        done
    done
done

if [[ ${#BK2_DIRS[@]} -eq 0 ]]; then
    echo "No BK2 directories found. Check AGENTS/CL_METHODS/SUBJECTS/RUN_PREFIXES in the CONFIG block."
    exit 1
fi

echo "Found ${#BK2_DIRS[@]} BK2 director(ies) to process."

mkdir -p "$REPO/logs"
JOB_LIST="$(mktemp "$REPO/logs/gif_dirs_XXXX.txt")"
printf "%s\n" "${BK2_DIRS[@]}" > "$JOB_LIST"
echo "Job list: $JOB_LIST"

if $DRY_RUN; then
    echo ""
    cat "$JOB_LIST"
    echo ""
    echo "(not submitting — run without --dry-run to submit)"
    exit 0
fi

N=${#BK2_DIRS[@]}
sbatch \
    --array="1-${N}" \
    --export="ALL,JOB_LIST=$JOB_LIST,REPO=$REPO,GIF_FPS=$GIF_FPS,GIF_SCALE=$GIF_SCALE" \
    "$REPO/scripts/gif_worker.sbatch"

echo "Submitted array 1-${N}."
