#!/usr/bin/env bash
# setup_hpc.sh — MariHA setup for HPC clusters (no apptainer)
#
# Alternative to the apptainer workflow. Uses a plain Python venv and installs
# stable-retro from source (the PyPI binary does not work on CC).
# The existing Dockerfile and narval_test_sac_cl.sh are unchanged.
#
# Usage:
#   bash setup_hpc.sh               # full setup (prompts for data download path)
#   bash setup_hpc.sh --no-download # skip datalad download (data pre-staged)
#   bash setup_hpc.sh --no-scenes   # skip mario.scenes download (already have it)
#
# Assumptions:
#   - Repo cloned anywhere under $HOME (e.g. ~/projects/MariHA)
#   - Data lives (or will live) at $SCRATCH/MariHA/data
#   - Run on an HPC cluster node where the module system is available

set -euo pipefail

DOWNLOAD_DATA=true
NO_SCENES=false
for arg in "$@"; do
  [[ "$arg" == "--no-download" ]] && DOWNLOAD_DATA=false
  [[ "$arg" == "--no-scenes" ]]   && NO_SCENES=true
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[setup_hpc]${NC} $*"; }
success() { echo -e "${GREEN}[ok]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC} $*"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compute nodes have no internet — this script must be run on a login node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  die "This script requires internet access. Run it on a login node, not inside a job/salloc."
fi

# ---------------------------------------------------------------------------
# 1. Load CC modules
# ---------------------------------------------------------------------------
info "Loading modules..."
# opencv must be loaded as a CC module before venv activation — pip's opencv-python-headless is a dummy on CC
module load StdEnv/2023 python/3.12 cmake gcc cuda/12.2 opencv/4.13.0 git-annex/10.20231129
success "Modules loaded."

# ---------------------------------------------------------------------------
# 2. Data root
# ---------------------------------------------------------------------------
if [[ -z "${MARIHA_DATA_ROOT:-}" ]]; then
  export MARIHA_DATA_ROOT="$SCRATCH/MariHA/data"
  warn "MARIHA_DATA_ROOT not set — defaulting to $MARIHA_DATA_ROOT"
else
  info "MARIHA_DATA_ROOT = $MARIHA_DATA_ROOT"
fi

export MARIHA_REPO="$REPO_ROOT"
export MARIHA_EXPERIMENT_DIR="$REPO_ROOT/experiments"
info "MARIHA_REPO = $MARIHA_REPO"
info "MARIHA_EXPERIMENT_DIR = $MARIHA_EXPERIMENT_DIR"

if [[ "$DOWNLOAD_DATA" == true && -t 0 ]]; then
  echo ""
  echo -e "${CYAN}[setup_hpc]${NC} Data will be downloaded to: ${YELLOW}$MARIHA_DATA_ROOT${NC}"
  read -rp "  Press Enter to use this path, or type a new path: " USER_DATA_ROOT
  if [[ -n "$USER_DATA_ROOT" ]]; then
    export MARIHA_DATA_ROOT="$USER_DATA_ROOT"
    info "Data root set to: $MARIHA_DATA_ROOT"
  fi
  mkdir -p "$MARIHA_DATA_ROOT"
fi

# ---------------------------------------------------------------------------
# 3. Clone stable-retro to $SCRATCH (not $HOME — lustre causes checkout failures)
# ---------------------------------------------------------------------------
RETRO_DIR="$SCRATCH/stable-retro"
if [[ ! -d "$RETRO_DIR" ]]; then
  info "Cloning stable-retro into $RETRO_DIR ..."
  git clone https://github.com/Farama-Foundation/stable-retro "$RETRO_DIR"
  success "stable-retro cloned."
else
  info "stable-retro already present at $RETRO_DIR — skipping clone."
fi

# ---------------------------------------------------------------------------
# 4. Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/env"
if [[ ! -d "$VENV_DIR/bin" ]]; then
  info "Creating virtual environment at env/ ..."
  python3 -m venv "$VENV_DIR"
  success "Virtual environment created."
else
  info "Virtual environment already exists at env/ — reusing."
fi
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 5. Datalad data download
# ---------------------------------------------------------------------------
if [[ "$DOWNLOAD_DATA" == true ]]; then
  if ! command -v datalad &>/dev/null; then
    info "Installing datalad..."
    pip install datalad --quiet
    success "datalad installed."
  fi

  # mario.stimuli → $MARIHA_DATA_ROOT/mario  (cd into DATA_ROOT first so datalad names it 'mario')
  if [[ ! -d "$MARIHA_DATA_ROOT/mario/.datalad" ]]; then
    info "Downloading mario.stimuli → $MARIHA_DATA_ROOT/mario ..."
    pushd "$MARIHA_DATA_ROOT" > /dev/null
    datalad install -s https://github.com/courtois-neuromod/mario.stimuli.git mario
    popd > /dev/null
  else
    info "mario.stimuli already present — skipping install."
  fi
  info "Fetching mario.stimuli file content..."
  pushd "$MARIHA_DATA_ROOT/mario" > /dev/null
  datalad get .
  popd > /dev/null
  success "mario.stimuli ready."

  if [[ "$NO_SCENES" == false ]]; then
    # mario.scenes → $MARIHA_DATA_ROOT/mario.scenes  (cd into DATA_ROOT so datalad places it there)
    if [[ ! -d "$MARIHA_DATA_ROOT/mario.scenes/.datalad" ]]; then
      info "Downloading mario.scenes → $MARIHA_DATA_ROOT/mario.scenes ..."
      pushd "$MARIHA_DATA_ROOT" > /dev/null
      datalad install https://github.com/courtois-neuromod/mario.scenes
      popd > /dev/null
      success "mario.scenes installed."
    else
      info "mario.scenes already present — skipping install."
    fi

    info "Fetching mario.scenes data..."
    pushd "$MARIHA_DATA_ROOT/mario.scenes" > /dev/null
    datalad get .
    python code/archives/decompress.py
    popd > /dev/null
    success "mario.scenes data ready."
  else
    info "Skipping mario.scenes (--no-scenes)."
  fi
else
  info "Skipping datalad download (--no-download)."
fi

# ---------------------------------------------------------------------------
# 6. Install stable-retro from source first, then MariHA without stable-retro
#    so pip does not overwrite the source build with the PyPI binary.
# ---------------------------------------------------------------------------
info "Installing build tools..."
pip install setuptools wheel
success "Build tools installed."

info "Installing stable-retro from source..."
pip install -e "$RETRO_DIR" --no-build-isolation
success "stable-retro installed."

info "Installing MariHA (remaining deps, skipping stable-retro)..."
# --no-deps avoids pip re-resolving stable-retro from PyPI and overwriting the source build.
pip install -e "$REPO_ROOT" --no-deps
# Install all remaining deps explicitly, excluding stable-retro.
# ---------------------------------------------------------------------------
# 7. Install tensorflow with bundled CUDA/cuDNN (CC has no cudnn module)
# ---------------------------------------------------------------------------
info "Installing tensorflow[and-cuda]..."
# tensorflow[and-cuda] bundles its own cuDNN — no cudnn module needed on CC.
pip install tensorflow  # CC's wheelhouse build is already GPU-enabled
pip install \
  "tf_keras>=2.13" "tensorflow-probability>=0.21" \
  "gymnasium>=0.29" "numpy>=1.24" "pandas>=2.0" "tensorboard>=2.13" \
  "tqdm>=4.65" "rich>=13.0"
# opencv is provided by the CC module loaded above — do not pip-install it
success "tensorflow[and-cuda] + remaining deps installed."

# ---------------------------------------------------------------------------
# 8. Stimuli data check
# ---------------------------------------------------------------------------
STIMULI_DIR="$MARIHA_DATA_ROOT/mario/SuperMarioBros-Nes"
if [[ ! -d "$STIMULI_DIR" ]]; then
  warn "Stimuli directory not found: $STIMULI_DIR"
  warn "Set MARIHA_DATA_ROOT correctly and re-run, or run manually later:"
  warn "  export MARIHA_DATA_ROOT=/your/path && bash setup_hpc.sh"
else
  success "Stimuli directory found."
fi

# ---------------------------------------------------------------------------
# 9. Subject data check
# ---------------------------------------------------------------------------
SCENES_DIR="$MARIHA_DATA_ROOT/mario.scenes"
STATE_COUNT=$(find "$SCENES_DIR" -name "*.state" 2>/dev/null | head -1 | wc -l | tr -d ' ')
if [[ "$STATE_COUNT" -eq 0 ]]; then
  warn "No .state files found in $SCENES_DIR."
  warn "Pull with: cd $SCENES_DIR && git annex get sub-*/"
else
  success "Subject data present."
fi

# ---------------------------------------------------------------------------
# 10. Generate per-scene scenario files
# ---------------------------------------------------------------------------
# The scenario generator needs scenes_mastersheet.csv, which is NOT in
# git-annex. mario.scenes ships its own ensure_scenes_data() helper that
# pulls the CSV from Zenodo into sourcedata/scenes_info/.
SCENES_REPO="$MARIHA_DATA_ROOT/mario.scenes"
if [[ -d "$SCENES_REPO/code" ]]; then
  info "Ensuring scenes mastersheet (mario.scenes ensure_scenes_data)..."
  if ( cd "$SCENES_REPO" && python -c "import sys; sys.path.insert(0, 'code'); from utils import ensure_scenes_data; print(ensure_scenes_data())" ); then
    success "Scenes mastersheet present."
  else
    warn "ensure_scenes_data() failed — needs internet (run on a login node)."
  fi
else
  warn "mario.scenes/code not found at $SCENES_REPO — cannot fetch mastersheet."
fi

info "Generating per-scene scenario files..."
if mariha-generate-scenarios; then
  success "Scenarios generated."
else
  warn "mariha-generate-scenarios failed — see messages above."
  warn "Run manually after pulling data: mariha-generate-scenarios"
fi

# ---------------------------------------------------------------------------
# 11. Smoke test
# ---------------------------------------------------------------------------
info "Running smoke test..."
python - <<'EOF'
import sys
errors = []
try:
    import mariha
except Exception as e:
    errors.append(f"import mariha: {e}")
try:
    import stable_retro
except Exception as e:
    errors.append(f"import stable_retro: {e}")
try:
    import tensorflow as tf
    _ = tf.constant(1.0)
except Exception as e:
    errors.append(f"import tensorflow: {e}")
if errors:
    print("Smoke test FAILED:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print("Smoke test passed.")
EOF
success "Smoke test passed."

# ---------------------------------------------------------------------------
# 12. Write the HPC environment file
# ---------------------------------------------------------------------------
# Job scripts source this file instead of hardcoding modules/paths or relying
# on ~/.bashrc. Fixed location so any script can find it regardless of where
# it is submitted from.
ENV_FILE="$HOME/.config/mariha/hpc_env.sh"
info "Writing HPC environment file → $ENV_FILE"
mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
# MariHA HPC environment — generated by setup_hpc.sh on $(date -Iseconds).
# Sourced by job scripts (e.g. scripts/hpc_run_all.sh) and interactive shells.
# Re-run setup_hpc.sh to regenerate; do not edit by hand.

module load StdEnv/2023 python/3.12 cmake gcc cuda/12.2 opencv/4.13.0 git-annex/10.20231129

# cv2 bindings come from the opencv EasyBuild module, not from pip
export PYTHONPATH="\$EBROOTOPENCV/lib/python3.12/site-packages:\${PYTHONPATH:-}"

export MARIHA_REPO="$MARIHA_REPO"
export MARIHA_DATA_ROOT="$MARIHA_DATA_ROOT"
export MARIHA_EXPERIMENT_DIR="$MARIHA_EXPERIMENT_DIR"

source "$MARIHA_REPO/env/bin/activate"
EOF
success "HPC environment file written."

# ---------------------------------------------------------------------------
# 13. Summary
# ---------------------------------------------------------------------------
echo ""
success "Setup complete."
echo ""
echo "Environment file written to: $ENV_FILE"
echo "Job scripts source it automatically — no ~/.bashrc changes needed."
echo ""
echo "Submit the full benchmark sweep (150-job array):"
echo "    cd $MARIHA_REPO && sbatch scripts/hpc_run_all.sh"
echo ""
echo "Run a single job interactively (on a GPU node via salloc):"
echo "    source $ENV_FILE"
echo "    mariha-run-cl --agent sac --cl_method ewc --subject sub-01 --seed 0"
