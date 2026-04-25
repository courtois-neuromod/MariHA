#!/bin/bash
#SBATCH --account=def-YOUR_PI
#SBATCH --job-name=mariha_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-9

# Eval is CPU-only (forward passes on a small PPO net — no GPU needed).
# One array job per CL method; all 10 run in parallel.
# Wall time: ~84 ckpts x 292 scenes x 5 eps x 167 steps / 500 fps ≈ 11 h → 14 h with margin.

CL_METHODS=(none agem clonex der ewc l2 mas multitask packnet si)
CL_METHOD=${CL_METHODS[$SLURM_ARRAY_TASK_ID]}

RUN_PREFIX="${RUN_PREFIX:-2026_04_24__12_11_21_seed0}"

mkdir -p logs

module load StdEnv/2023 python/3.12 cmake gcc opencv/4.13.0

export PYTHONPATH="$EBROOTOPENCV/lib/python3.12/site-packages:${PYTHONPATH:-}"

REPO="${MARIHA_REPO:-$HOME/projects/MariHA}"
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

if [ "$CL_METHOD" = "none" ]; then
    echo "Evaluating base PPO (no CL method), run_prefix=$RUN_PREFIX"
    mariha-evaluate \
        --subject sub-01 \
        --agent ppo \
        --run_prefix "$RUN_PREFIX" \
        --experiment_dir "$REPO/experiments" \
        --n_episodes 5
else
    echo "Evaluating ppo_${CL_METHOD}, run_prefix=$RUN_PREFIX"
    mariha-evaluate \
        --subject sub-01 \
        --agent ppo \
        --cl_method "$CL_METHOD" \
        --run_prefix "$RUN_PREFIX" \
        --experiment_dir "$REPO/experiments" \
        --n_episodes 5
fi
