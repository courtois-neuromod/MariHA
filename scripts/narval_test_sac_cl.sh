#!/bin/bash
#SBATCH --account=def-YOUR_PI        # replace with your allocation (e.g. def-pbellec)
#SBATCH --job-name=mariha_sac_cl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

module load StdEnv/2023 apptainer/1.4.5

SIF=$SCRATCH/mariha-gpu.sif
REPO=$SCRATCH/MariHA

apptainer exec --nv \
    --bind "$REPO":/app \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    "$SIF" \
    bash -c "cd /app && PYTHONPATH=/app python3 scripts/run_cl.py --agent sac --subject sub-01 --seed 0"
