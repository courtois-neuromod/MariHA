#!/bin/bash
#SBATCH --account=def-YOUR_PI        # replace with your allocation
#SBATCH --job-name=mariha_all
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-149

# Full benchmark sweep: every agent x every subject x every CL method.
# Each array task is one independent job (one mariha-run-cl invocation).
#
#   3 agents x 5 subjects x 10 CL options = 150 jobs (array 0-149)
#
# Assumes setup_hpc.sh has already created the venv. Data is resolved via
# MARIHA_DATA_ROOT (defaults to $SCRATCH/MariHA/data).

AGENTS=(sac ppo dqn)
SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
CL_METHODS=(agem clonex der ewc l2 mas multitask packnet si none)

N_CL=${#CL_METHODS[@]}        # 10
N_SUBJ=${#SUBJECTS[@]}        # 5

# Decompose the flat array index into (agent, subject, cl_method).
IDX=$SLURM_ARRAY_TASK_ID
CL_METHOD=${CL_METHODS[$(( IDX % N_CL ))]}
SUBJECT=${SUBJECTS[$(( (IDX / N_CL) % N_SUBJ ))]}
AGENT=${AGENTS[$(( IDX / (N_CL * N_SUBJ) ))]}

mkdir -p logs

module load StdEnv/2023 python/3.12 cmake gcc cuda/12.2 opencv/4.13.0

# Expose the opencv module's cv2 bindings inside the venv via EasyBuild root
export PYTHONPATH="$EBROOTOPENCV/lib/python3.12/site-packages:${PYTHONPATH:-}"

REPO="${MARIHA_REPO:-$HOME/projects/MariHA}"
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

# Prevent TF from pre-allocating all GPU VRAM on startup.
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Task $IDX: agent=$AGENT subject=$SUBJECT cl_method=$CL_METHOD"

if [ "$CL_METHOD" = "none" ]; then
    mariha-run-cl --agent "$AGENT" --subject "$SUBJECT" --seed 0 --progress off
else
    mariha-run-cl --agent "$AGENT" --cl_method "$CL_METHOD" \
        --subject "$SUBJECT" --seed 0 --progress off
fi
