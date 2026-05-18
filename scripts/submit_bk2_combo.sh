#!/bin/bash
# Submit a BK2-generation array job for ONE trained combo: an agent, a
# subject, and an optional CL method. Each array task handles one BIDS run.
#
# Called automatically by hpc_run_all.sh after a training task succeeds, and
# usable by hand. The run_prefix is discovered from the checkpoints written by
# training (subject-namespaced) unless given explicitly.
#
# Usage:
#   ./scripts/submit_bk2_combo.sh --agent ppo --subject sub-01 [--cl_method packnet]
#   ./scripts/submit_bk2_combo.sh --agent ppo --subject sub-01 --run_prefix PREFIX

set -euo pipefail

AGENT=""
SUBJECT=""
CL_METHOD=""
RUN_PREFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)      AGENT="$2";      shift 2 ;;
        --subject)    SUBJECT="$2";    shift 2 ;;
        --cl_method)  CL_METHOD="$2";  shift 2 ;;
        --run_prefix) RUN_PREFIX="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$AGENT" || -z "$SUBJECT" ]]; then
    echo "Usage: $0 --agent <name> --subject <sub-XX> [--cl_method M] [--run_prefix P]" >&2
    exit 1
fi

# MARIHA_* paths + SLURM account come from the setup_hpc.sh env file. They are
# already in scope when called by hpc_run_all.sh; sourced here for standalone use.
if [[ -z "${MARIHA_EXPERIMENT_DIR:-}" ]]; then
    MARIHA_ENV_FILE="${MARIHA_ENV_FILE:-$HOME/.config/mariha/hpc_env.sh}"
    [[ -f "$MARIHA_ENV_FILE" ]] && source "$MARIHA_ENV_FILE"
fi
REPO="${MARIHA_REPO:?MARIHA_REPO not set — run setup_hpc.sh}"
EXP="${MARIHA_EXPERIMENT_DIR:?MARIHA_EXPERIMENT_DIR not set — run setup_hpc.sh}"
DATA_ROOT="${MARIHA_DATA_ROOT:?MARIHA_DATA_ROOT not set — run setup_hpc.sh}"

run_label="$AGENT"
[[ -n "$CL_METHOD" ]] && run_label="${AGENT}_${CL_METHOD}"

# Discover the run_prefix from the (subject-namespaced) checkpoints, newest
# first, unless one was passed explicitly.
CKPT_DIR="$EXP/checkpoints/$run_label/$SUBJECT"
if [[ -z "$RUN_PREFIX" ]]; then
    [[ -d "$CKPT_DIR" ]] || { echo "ERROR: no checkpoints at $CKPT_DIR" >&2; exit 1; }
    for d in $(ls -t "$CKPT_DIR" 2>/dev/null); do
        p="${d%_task*}"
        if [[ "$p" != "$d" ]]; then RUN_PREFIX="$p"; break; fi
    done
fi
[[ -n "$RUN_PREFIX" ]] || { echo "ERROR: could not determine run_prefix in $CKPT_DIR" >&2; exit 1; }

# Collect this subject's run_ids (chronological), one per line, for the array.
SCENES_SUBJ="$DATA_ROOT/mario.scenes/$SUBJECT"
[[ -d "$SCENES_SUBJ" ]] || { echo "ERROR: subject data not found: $SCENES_SUBJ" >&2; exit 1; }
mapfile -t RUN_IDS < <(
    find "$SCENES_SUBJ" -name "*_desc-scenes_events.tsv" \
        | grep -oP 'ses-\d+_.*_run-\d+' \
        | sed 's/_task-mario//' \
        | sort -u
)
if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no run_ids found under $SCENES_SUBJ" >&2
    exit 1
fi

mkdir -p "$REPO/logs"
JOB_LIST="$(mktemp "$REPO/logs/bk2_${run_label}_${SUBJECT}_XXXX.txt")"
printf "%s\n" "${RUN_IDS[@]}" > "$JOB_LIST"

SBATCH_ARGS=()
[[ -n "${MARIHA_SLURM_ACCOUNT:-}" ]] && SBATCH_ARGS+=(--account="$MARIHA_SLURM_ACCOUNT")

sbatch "${SBATCH_ARGS[@]}" \
    --job-name="bk2-${run_label}-${SUBJECT}" \
    --array="1-${#RUN_IDS[@]}" \
    --export="ALL,JOB_LIST=$JOB_LIST,SUBJECT=$SUBJECT,AGENT=$AGENT,CL_METHOD=$CL_METHOD,RUN_PREFIX=$RUN_PREFIX" \
    "$REPO/scripts/bk2_worker.sbatch"

echo "Submitted BK2 array (${#RUN_IDS[@]} tasks): $run_label | $SUBJECT | $RUN_PREFIX"
