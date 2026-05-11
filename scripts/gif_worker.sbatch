#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --job-name=mariha-gif
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/gif_%A_%a.out
#SBATCH --error=logs/gif_%A_%a.err

# Each array task converts all .bk2 files in one directory to GIF.
# JOB_LIST, REPO, GIF_FPS, GIF_SCALE are exported by submit_gif_all.sh.

BK2_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST")

echo "=== Task ${SLURM_ARRAY_TASK_ID}: $BK2_DIR ==="

module load StdEnv/2023 python/3.12 cmake gcc opencv/4.13.0
export PYTHONPATH="$EBROOTOPENCV/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

python "$REPO/scripts/replay_model_bk2.py" \
    --bk2_dir "$BK2_DIR" \
    --gif \
    --gif-fps "${GIF_FPS:-24}" \
    --gif-scale "${GIF_SCALE:-256}" \
    -v
