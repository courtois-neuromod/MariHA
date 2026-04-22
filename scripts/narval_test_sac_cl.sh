#!/bin/bash
#SBATCH --account=def-YOUR_PI        # replace with your allocation
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

# Data root: set MARIHA_DATA_ROOT if your data lives outside the repo
# (e.g. repo on $HOME, data on $SCRATCH). When repo and data are co-located
# under $SCRATCH/MariHA/ the default works and this line can be omitted.
MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$REPO/data}"

apptainer exec --nv \
    --bind "$REPO":/app \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env MARIHA_DATA_ROOT="$MARIHA_DATA_ROOT" \
    "$SIF" \
    bash -c "cd /app && PYTHONPATH=/app python3 scripts/run_cl.py --agent sac --subject sub-01 --seed 0"
