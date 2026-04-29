#!/bin/bash
#SBATCH --account=def-YOUR_PI
#SBATCH --job-name=mariha_ppo_cl
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9

CL_METHODS=(agem clonex der ewc l2 mas multitask packnet si none)
CL_METHOD=${CL_METHODS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

module load StdEnv/2023 python/3.12 cmake gcc cuda/12.2 opencv/4.13.0

export PYTHONPATH="$EBROOTOPENCV/lib/python3.12/site-packages:${PYTHONPATH:-}"

REPO="${MARIHA_REPO:-$HOME/projects/MariHA}"
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

# Prevent TF from pre-allocating all GPU VRAM (A100 = 80GB) on startup.
# Without this, TF grabs all available VRAM as virtual address space,
# which maps back to system RAM on some CC node configurations.
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [ "$CL_METHOD" = "none" ]; then
    echo "Running base PPO (no CL method)"
    mariha-run-cl --agent ppo --subject sub-01 --seed 0 --progress off
else
    echo "Running CL method: $CL_METHOD"
    mariha-run-cl --agent ppo --cl_method "$CL_METHOD" --subject sub-01 --seed 0 --progress off
fi
