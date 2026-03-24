#!/usr/bin/env bash
# setup.sh — MariHA environment setup
#
# Creates a Python virtual environment, installs all dependencies, generates
# the per-scene scenario files, and smoke-tests the installation.
#
# Usage:
#   bash setup.sh            # full setup
#   bash setup.sh --no-data  # skip data availability check (CI / no dataset)
#
# Requirements:
#   - Python 3.9+
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
for arg in "$@"; do
  [[ "$arg" == "--no-data" ]] && CHECK_DATA=false
done

# ---------------------------------------------------------------------------
# Locate repo root (directory containing this script)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
info "Checking Python version..."
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  die "Python not found. Set PYTHON=python3.x or install Python 3.9+."
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 9 ) ]]; then
  die "Python 3.9+ required, found $PY_VERSION."
fi
success "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/env"
if [[ -d "$VENV_DIR/bin" ]]; then
  info "Virtual environment already exists at env/ — reusing."
else
  info "Creating virtual environment at env/ ..."
  "$PYTHON" -m venv "$VENV_DIR"
  success "Virtual environment created."
fi

PYTHON_VENV="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Upgrade pip silently
"$PIP" install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Install the package
# ---------------------------------------------------------------------------
info "Installing mariha and dependencies (this may take a few minutes)..."
"$PIP" install -e ".[dev]" --quiet
success "Package installed."

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
# 4. Stimuli data check
# ---------------------------------------------------------------------------
STIMULI_DIR="$REPO_ROOT/data/mario/stimuli/SuperMarioBros-Nes"
if [[ ! -d "$STIMULI_DIR" ]]; then
  die "Stimuli directory not found: $STIMULI_DIR
  The Mario game integration must be present in data/mario/stimuli/.
  Please ensure the mario.scenes dataset is correctly linked."
fi
success "Stimuli directory found."

# ---------------------------------------------------------------------------
# 5. Subject data (git-annex)
# ---------------------------------------------------------------------------
SCENES_DIR="$REPO_ROOT/data/mario.scenes"
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
# 6. Generate scenario files
# ---------------------------------------------------------------------------
info "Generating per-scene scenario files..."
if "$VENV_DIR/bin/mariha-generate-scenarios" 2>/dev/null; then
  success "Scenarios generated."
else
  warn "mariha-generate-scenarios failed (mastersheet may not be present yet)."
  warn "Run manually after pulling data: mariha-generate-scenarios"
fi

# ---------------------------------------------------------------------------
# 7. Smoke test
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
    from mariha.methods import (
        L2_SAC, EWC_SAC, MAS_SAC, SI_SAC, OWL_SAC,
        PackNet_SAC, AGEM_SAC, VCL_SAC, DER_SAC,
        ClonEx_SAC, MultiTask_SAC,
    )
except Exception as e:
    errors.append(f"import methods: {e}")

try:
    from mariha.eval import compute_cl_metrics, eval_on_scene
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
echo "  Activate the environment:"
echo "    source env/bin/activate"
echo ""
echo "  Train (full CL curriculum):"
echo "    mariha-run-cl --subject sub-01 --cl_method ewc --seed 0"
echo ""
echo "  Train (single scene):"
echo "    mariha-run-single --scene_id w1l1s0 --seed 0"
echo ""
echo "  Evaluate:"
echo "    mariha-evaluate --subject sub-01 --cl_method ewc --run_prefix <timestamp_seed0>"
echo ""
