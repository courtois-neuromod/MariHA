#!/usr/bin/env bash
# setup_cc.sh — MariHA environment setup for Compute Canada (no uv)
#
# Usage:
#   module load python/3.12.4
#   bash setup_cc.sh            # full setup
#   bash setup_cc.sh --no-data  # skip data availability check
#
# Requirements:
#   - module load python/3.12.4 (or any 3.9+) before running

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
# Locate repo root
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Check Python
# ---------------------------------------------------------------------------
PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
[[ -z "$PYTHON_BIN" ]] && die "python3 not found. Run: module load python/3.12.4"
success "Using Python at $PYTHON_BIN ($(python3 --version))"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/env"
if [[ -d "$VENV_DIR/bin" ]]; then
  info "Virtual environment already exists at env/ — reusing."
else
  info "Creating virtual environment at env/ ..."
  python3 -m venv "$VENV_DIR"
  success "Virtual environment created."
fi

PYTHON_VENV="$VENV_DIR/bin/python"
PIP_VENV="$VENV_DIR/bin/pip"

# ---------------------------------------------------------------------------
# 3. Install the package
# ---------------------------------------------------------------------------
info "Upgrading pip..."
"$PIP_VENV" install --upgrade pip --quiet

info "Installing mariha and dependencies (this may take a few minutes)..."
# opencv and stable-retro need special handling on Compute Canada:
# - opencv-python-headless: blocked, provided by the opencv module
# - stable-retro: must be built from source; cmake needs the real Python prefix
#   (not the venv) to find libpython correctly
"$PYTHON_VENV" -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    d = tomllib.load(f)
skip = ('opencv', 'stable-retro')
deps = [x for x in d['project']['dependencies'] if not any(s in x.lower() for s in skip)]
print('\n'.join(deps))
" | "$PIP_VENV" install -r /dev/stdin --quiet

info "Building stable-retro (this takes a few minutes)..."
PYTHON_REAL_PREFIX=$("$PYTHON_VENV" -c "import sys; print(getattr(sys, 'real_prefix', sys.base_prefix))")
CMAKE_ARGS="-DPython_ROOT_DIR=${PYTHON_REAL_PREFIX}" "$PIP_VENV" install stable-retro --quiet
success "stable-retro built."

"$PIP_VENV" install -e ".[dev]" --no-deps --quiet
success "Package installed."

info "Installing tensorflow[and-cuda] for GPU support..."
"$PIP_VENV" install "tensorflow[and-cuda]" --quiet
success "tensorflow[and-cuda] installed."

# Verify tf_keras
if "$PYTHON_VENV" -c "import tensorflow as tf; v=tf.__version__; parts=v.split('.'); assert int(parts[1]) >= 16 if int(parts[0])==2 else False" 2>/dev/null; then
  info "TF >= 2.16 detected — verifying tf_keras..."
  if ! "$PYTHON_VENV" -c "import tf_keras" 2>/dev/null; then
    warn "tf_keras not importable. Try: pip install tf_keras"
  else
    success "tf_keras OK."
  fi
fi

# ---------------------------------------------------------------------------
# 4. Stimuli / subject data check
# ---------------------------------------------------------------------------
STIMULI_DIR="$REPO_ROOT/data/mario/stimuli/SuperMarioBros-Nes"
if [[ "$CHECK_DATA" == true ]]; then
  [[ ! -d "$STIMULI_DIR" ]] && die "Stimuli directory not found: $STIMULI_DIR"
  success "Stimuli directory found."

  SCENES_DIR="$REPO_ROOT/data/mario.scenes"
  info "Checking subject data availability..."
  if command -v git-annex &>/dev/null && [[ -d "$SCENES_DIR/.git" ]]; then
    MISSING=$(cd "$SCENES_DIR" && git annex find --not --in=here --include="sub-*/**/*.state.gz" 2>/dev/null | wc -l | tr -d ' ')
    [[ "$MISSING" -gt 0 ]] && warn "$MISSING state files missing. Run: cd data/mario.scenes && git annex get sub-*/" || success "All subject data present."
  else
    STATE_COUNT=$(find "$SCENES_DIR" -name "*.state.gz" 2>/dev/null | head -1 | wc -l | tr -d ' ')
    [[ "$STATE_COUNT" -eq 0 ]] && warn "No .state.gz files found. Run: cd data/mario.scenes && git annex get sub-*/" || success "Subject data present."
  fi
else
  info "Skipping data check (--no-data)."
fi

# ---------------------------------------------------------------------------
# 5. Generate scenario files
# ---------------------------------------------------------------------------
info "Generating per-scene scenario files..."
if "$VENV_DIR/bin/mariha-generate-scenarios" 2>/dev/null; then
  success "Scenarios generated."
else
  warn "mariha-generate-scenarios failed (mastersheet may not be present yet)."
  warn "Run manually after pulling data: mariha-generate-scenarios"
fi

# ---------------------------------------------------------------------------
# 6. Smoke test
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
    for err in errors: print(f"  - {err}")
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
echo -e "${GREEN}  MariHA setup complete (Compute Canada).${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Activate the environment:"
echo "    source env/bin/activate"
echo ""
echo "  Train (full CL curriculum):"
echo "    mariha-run-cl --agent sac --subject sub-01 --seed 0"
echo ""
echo "  Train (single scene):"
echo "    mariha-run-single --agent sac --scene_id w1l1s0 --seed 0"
echo ""
echo "  Evaluate:"
echo "    mariha-evaluate --subject sub-01 --agent ewc --run_prefix <timestamp_seed0>"
echo ""
