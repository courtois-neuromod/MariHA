#!/bin/bash
#SBATCH --account=def-YOUR_PI        # replace with your allocation
#SBATCH --job-name=mariha_sac_cl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

# No-apptainer variant: uses the plain venv created by setup_cc.sh.
# Assumes:
#   - Repo cloned to $HOME/projects/MariHA (or wherever)
#   - Data on $SCRATCH/MariHA/data
#   - setup_cc.sh already run once to create the venv

module load StdEnv/2023 python/3.12 cmake gcc

REPO="$HOME/projects/MariHA"         # adjust if your repo is elsewhere
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"

source "$REPO/env/bin/activate"

mariha-run-cl --agent sac --subject sub-01 --seed 0
