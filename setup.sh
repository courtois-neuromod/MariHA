#!/usr/bin/env bash
# setup.sh — MariHA environment setup
#
# Creates a Python virtual environment, installs all dependencies, generates
# the per-scene scenario files, and smoke-tests the installation.
#
# Usage:
#   bash setup.sh               # full setup (prompts for data download path)
#   bash setup.sh --no-data     # skip data availability check (CI / no dataset)
#   bash setup.sh --no-download # skip datalad download (data already staged)
#   bash setup.sh --no-scenes   # skip mario.scenes download (already have it)
#
# Requirements:
#   - Python 3.9+
#   - SSH key configured for github.com (for datalad download)
#   - git-annex (optional, for pulling subject data files)

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[ok]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC} $*"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
CHECK_DATA=true
DOWNLOAD_DATA=true
NO_SCENES=false
CUSTOM_DATA_ROOT=false
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
for arg in "$@"; do
  [[ "$arg" == "--no-data" ]]     && CHECK_DATA=false && DOWNLOAD_DATA=false
  [[ "$arg" == "--no-download" ]] && DOWNLOAD_DATA=false
  [[ "$arg" == "--no-scenes" ]]   && NO_SCENES=true
done

# ---------------------------------------------------------------------------
# Locate repo root (directory containing this script)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
DATA_ROOT="${MARIHA_DATA_ROOT:-$REPO_ROOT/data}"

# ---------------------------------------------------------------------------
# Prompt for data download path (interactive only, skipped with --no-download)
# ---------------------------------------------------------------------------
if [[ "$DOWNLOAD_DATA" == true && -t 0 ]]; then
  echo ""
  echo -e "${CYAN}[setup]${NC} Data will be downloaded to: ${YELLOW}$DATA_ROOT${NC}"
  read -rp "  Press Enter to use this path, or type a new path: " USER_DATA_ROOT
  if [[ -n "$USER_DATA_ROOT" ]]; then
    DATA_ROOT="$USER_DATA_ROOT"
    CUSTOM_DATA_ROOT=true
    info "Data root set to: $DATA_ROOT"
  fi
  mkdir -p "$DATA_ROOT"
fi

# ---------------------------------------------------------------------------
# 1. Ensure uv is available
# ---------------------------------------------------------------------------
info "Checking for uv..."
if ! command -v uv &>/dev/null; then
  info "uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
success "uv $(uv --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 2. Resolve Python interpreter
# ---------------------------------------------------------------------------
# Prefer a system Python matching $PYTHON_VERSION that is already on PATH
# (e.g. after `module load python/3.12` on Compute Canada).  Using a system
# interpreter avoids uv embedding its own bundled CPython in the venv, whose
# path becomes a broken symlink on HPC cluster nodes that didn't create it.
info "Resolving Python $PYTHON_VERSION interpreter..."
SYSTEM_PYTHON="$(command -v "python${PYTHON_VERSION}" 2>/dev/null \
                 || command -v "python3" 2>/dev/null || true)"

if [[ -n "$SYSTEM_PYTHON" && "$SYSTEM_PYTHON" != *".local/share/uv"* ]]; then
  PYTHON_ARG="$SYSTEM_PYTHON"
  success "Using system Python at $PYTHON_ARG"
else
  info "System Python $PYTHON_VERSION not found — installing via uv..."
  uv python install "$PYTHON_VERSION"
  PYTHON_ARG="$PYTHON_VERSION"
  success "Python $PYTHON_VERSION ready (uv-managed)."
fi

# ---------------------------------------------------------------------------
# 3. Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/env"
if [[ -d "$VENV_DIR/bin" ]]; then
  info "Virtual environment already exists at env/ — reusing."
else
  info "Creating virtual environment at env/ ..."
  uv venv --python "$PYTHON_ARG" "$VENV_DIR"
  success "Virtual environment created."
fi

PYTHON_VENV="$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# 4. Install the package
# ---------------------------------------------------------------------------
info "Installing mariha and dependencies (this may take a few minutes)..."
uv pip install --python "$PYTHON_VENV" -e ".[dev]" --quiet
success "Package installed."

# On Linux, swap plain tensorflow for the CUDA-bundled wheel so TF can find
# GPU libraries without requiring exact system CUDA module versions.
if [[ "$(uname)" == "Linux" ]]; then
  info "Linux detected — installing tensorflow[and-cuda] for GPU support..."
  uv pip install --python "$PYTHON_VENV" "tensorflow[and-cuda]" --quiet
  success "tensorflow[and-cuda] installed."
fi

# Verify tf_keras is importable (needed for TF >= 2.16 / Keras 3 environments)
if "$PYTHON_VENV" -c "import tensorflow as tf; v=tf.__version__; parts=v.split('.'); assert int(parts[1]) >= 16 if int(parts[0])==2 else False" 2>/dev/null; then
  info "TF >= 2.16 detected — verifying tf_keras..."
  if ! "$PYTHON_VENV" -c "import tf_keras" 2>/dev/null; then
    warn "tf_keras not importable despite being installed. Try: pip install tf_keras"
  else
    success "tf_keras OK."
  fi
fi

# ---------------------------------------------------------------------------
# 5. Datalad data download
# ---------------------------------------------------------------------------
if [[ "$DOWNLOAD_DATA" == true ]]; then
  info "Setting up datalad..."
  if ! "$VENV_DIR/bin/datalad" --version &>/dev/null 2>&1; then
    info "datalad not found — installing into venv..."
    uv pip install --python "$PYTHON_VENV" datalad --quiet
    success "datalad installed."
  fi
  DATALAD="$VENV_DIR/bin/datalad"

  # mario.stimuli → $DATA_ROOT/mario  (cd into DATA_ROOT first so datalad names it 'mario')
  if [[ ! -d "$DATA_ROOT/mario/.datalad" ]]; then
    info "Downloading mario.stimuli → $DATA_ROOT/mario ..."
    info "(SSH key required for git@github.com)"
    pushd "$DATA_ROOT" > /dev/null
    "$DATALAD" install -s git@github.com:courtois-neuromod/mario.stimuli.git mario
    popd > /dev/null
  else
    info "mario.stimuli already present — skipping install."
  fi
  info "Fetching mario.stimuli file content..."
  pushd "$DATA_ROOT/mario" > /dev/null
  "$DATALAD" get .
  popd > /dev/null
  success "mario.stimuli ready."

  if [[ "$NO_SCENES" == false ]]; then
    # mario.scenes → $DATA_ROOT/mario.scenes  (cd into DATA_ROOT so datalad places it there)
    if [[ ! -d "$DATA_ROOT/mario.scenes/.datalad" ]]; then
      info "Downloading mario.scenes → $DATA_ROOT/mario.scenes ..."
      info "(SSH key required for git@github.com)"
      pushd "$DATA_ROOT" > /dev/null
      "$DATALAD" install git@github.com:courtois-neuromod/mario.scenes
      popd > /dev/null
      success "mario.scenes installed."
    else
      info "mario.scenes already present — skipping install."
    fi

    info "Checking out dev_refactor and fetching mario.scenes data..."
    pushd "$DATA_ROOT/mario.scenes" > /dev/null
    git checkout dev_refactor
    "$DATALAD" get .
    "$PYTHON_VENV" code/archives/decompress.py
    popd > /dev/null
    success "mario.scenes data ready."
  else
    info "Skipping mario.scenes (--no-scenes)."
  fi
else
  info "Skipping datalad download (--no-download)."
fi

# ---------------------------------------------------------------------------
# 6. Stimuli data check
# ---------------------------------------------------------------------------
STIMULI_DIR="$DATA_ROOT/mario/SuperMarioBros-Nes"
if [[ ! -d "$STIMULI_DIR" ]]; then
  die "Stimuli directory not found: $STIMULI_DIR
  The Mario game integration must be present in data/mario/stimuli/.
  Please ensure the mario.scenes dataset is correctly linked."
fi
success "Stimuli directory found."

# ---------------------------------------------------------------------------
# 7. Subject data (git-annex)
# ---------------------------------------------------------------------------
SCENES_DIR="$DATA_ROOT/mario.scenes"
if [[ "$CHECK_DATA" == true ]]; then
  info "Checking subject data availability..."
  if command -v git-annex &>/dev/null && [[ -d "$SCENES_DIR/.git" ]]; then
    MISSING=$(cd "$SCENES_DIR" && git annex find --not --in=here --include="sub-*/**/*.state.gz" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$MISSING" -gt 0 ]]; then
      warn "$MISSING state files not yet downloaded."
      warn "Run: cd data/mario.scenes && git annex get sub-*/"
      warn "Or: datalad get data/mario.scenes/sub-*/"
    else
      success "All subject data present."
    fi
  else
    # Check manually whether at least one state file exists
    STATE_COUNT=$(find "$SCENES_DIR" -name "*.state.gz" 2>/dev/null | head -1 | wc -l | tr -d ' ')
    if [[ "$STATE_COUNT" -eq 0 ]]; then
      warn "No .state.gz files found in data/mario.scenes/."
      warn "Pull the dataset with: cd data/mario.scenes && git annex get sub-*/"
    else
      success "Subject data present."
    fi
  fi
else
  info "Skipping data check (--no-data)."
fi

# ---------------------------------------------------------------------------
# 8. Generate scenario files
# ---------------------------------------------------------------------------
info "Generating per-scene scenario files..."
if "$VENV_DIR/bin/mariha-generate-scenarios" 2>/dev/null; then
  success "Scenarios generated."
else
  warn "mariha-generate-scenarios failed (mastersheet may not be present yet)."
  warn "Run manually after pulling data: mariha-generate-scenarios"
fi

# ---------------------------------------------------------------------------
# 9. Smoke test
# ---------------------------------------------------------------------------
info "Running smoke test..."
"$PYTHON_VENV" - <<'EOF'
import sys

errors = []

try:
    import mariha
except Exception as e:
    errors.append(f"import mariha: {e}")

try:
    from mariha.rl.sac import SAC
    from mariha.rl.models import MlpActor, MlpCritic
except Exception as e:
    errors.append(f"import SAC/models: {e}")

try:
    from mariha.methods import EWC, L2Regularizer, AGEM, DER, PackNet, SI, MAS, ClonEx, MultiTask
except Exception as e:
    errors.append(f"import methods: {e}")

try:
    from mariha.eval import eval_on_scene, build_scene_metadata
except Exception as e:
    errors.append(f"import eval: {e}")

try:
    import tensorflow as tf
    import tf_keras
    _ = tf.constant(1.0)
except Exception as e:
    errors.append(f"tensorflow/tf_keras: {e}")

if errors:
    print("SMOKE TEST FAILED:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("All imports OK.")
EOF

success "Smoke test passed."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  MariHA setup complete.${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
if [[ "$CUSTOM_DATA_ROOT" == true ]]; then
  echo "  Data path (add to ~/.bashrc for training scripts):"
  echo "    export MARIHA_DATA_ROOT=$DATA_ROOT"
  echo ""
fi
echo "  Activate the environment:"
echo "    source env/bin/activate"
echo ""
echo "  Train (full CL curriculum):"
echo "    mariha-run-cl --agent sac --cl_method ewc --subject sub-01 --seed 0"
echo ""
echo "  Train (single scene):"
echo "    mariha-run-single --agent sac --scene_id w1l1s0 --seed 0"
echo ""
echo "  Evaluate:"
echo "    mariha-evaluate --subject sub-01 --agent sac --cl_method ewc --run_prefix <timestamp_seed0>"
echo ""
