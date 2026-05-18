#!/bin/bash
# Bulk BK2 generation: submit one array job per (agent, cl_method, subject)
# combo found in the checkpoints tree. Thin wrapper over submit_bk2_combo.sh.
#
# Use this to (re)generate BK2s for an already-finished sweep. The normal
# path is automatic — hpc_run_all.sh chains BK2 generation per training task.
#
# Usage:
#   ./scripts/submit_bk2_all.sh                   # every combo found
#   ./scripts/submit_bk2_all.sh --agent ppo       # filter by agent
#   ./scripts/submit_bk2_all.sh --subject sub-01  # filter by subject
#   ./scripts/submit_bk2_all.sh --dry-run

set -euo pipefail

FILTER_AGENT=""
FILTER_SUBJECT=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)   FILTER_AGENT="$2";   shift 2 ;;
        --subject) FILTER_SUBJECT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true;        shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${MARIHA_EXPERIMENT_DIR:-}" ]]; then
    MARIHA_ENV_FILE="${MARIHA_ENV_FILE:-$HOME/.config/mariha/hpc_env.sh}"
    [[ -f "$MARIHA_ENV_FILE" ]] && source "$MARIHA_ENV_FILE"
fi
REPO="${MARIHA_REPO:?MARIHA_REPO not set — run setup_hpc.sh}"
CKPT_ROOT="${MARIHA_EXPERIMENT_DIR:?MARIHA_EXPERIMENT_DIR not set}/checkpoints"

[[ -d "$CKPT_ROOT" ]] || { echo "ERROR: no checkpoints at $CKPT_ROOT" >&2; exit 1; }

count=0
for rl_dir in "$CKPT_ROOT"/*/; do
    [[ -d "$rl_dir" ]] || continue
    run_label="$(basename "$rl_dir")"          # agent  or  agent_clmethod
    agent="${run_label%%_*}"
    cl_method=""
    [[ "$run_label" == *_* ]] && cl_method="${run_label#*_}"
    [[ -n "$FILTER_AGENT" && "$agent" != "$FILTER_AGENT" ]] && continue

    for subj_dir in "$rl_dir"*/; do
        [[ -d "$subj_dir" ]] || continue
        subject="$(basename "$subj_dir")"
        [[ -n "$FILTER_SUBJECT" && "$subject" != "$FILTER_SUBJECT" ]] && continue

        cl_arg=()
        [[ -n "$cl_method" ]] && cl_arg=(--cl_method "$cl_method")
        if $DRY_RUN; then
            echo "[dry-run] submit_bk2_combo.sh --agent $agent --subject $subject ${cl_arg[*]}"
        else
            bash "$REPO/scripts/submit_bk2_combo.sh" \
                --agent "$agent" --subject "$subject" "${cl_arg[@]}"
        fi
        (( count++ )) || true
    done
done

echo "Done. $count combo(s) processed."
