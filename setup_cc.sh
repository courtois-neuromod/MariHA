#!/usr/bin/env bash
# setup_cc.sh — MariHA setup for Compute Canada (no apptainer)
#
# Alternative to the apptainer workflow. Uses a plain Python venv and installs
# stable-retro from source (the PyPI binary does not work on CC).
# The existing Dockerfile and narval_test_sac_cl.sh are unchanged.
#
# Usage:
#   bash setup_cc.sh
#
# Assumptions:
#   - Repo cloned anywhere under $HOME (e.g. ~/projects/MariHA)
#   - Data lives (or will live) at $SCRATCH/MariHA/data
#   - Run on a CC cluster node where the module system is available

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[setup_cc]${NC} $*"; }
success() { echo -e "${GREEN}[ok]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC} $*"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Load CC modules
# ---------------------------------------------------------------------------
info "Loading modules..."
module load StdEnv/2023 python/3.12
success "Modules loaded."

# ---------------------------------------------------------------------------
# 2. Data root
# ---------------------------------------------------------------------------
if [[ -z "${MARIHA_DATA_ROOT:-}" ]]; then
  export MARIHA_DATA_ROOT="$SCRATCH/MariHA/data"
  warn "MARIHA_DATA_ROOT not set — defaulting to $MARIHA_DATA_ROOT"
  warn "If your data is elsewhere, re-run with: export MARIHA_DATA_ROOT=/your/path && bash setup_cc.sh"
else
  info "MARIHA_DATA_ROOT = $MARIHA_DATA_ROOT"
fi

# ---------------------------------------------------------------------------
# 3. Clone stable-retro (sibling directory) if not already present
# ---------------------------------------------------------------------------
RETRO_DIR="$(dirname "$REPO_ROOT")/stable-retro"
if [[ ! -d "$RETRO_DIR" ]]; then
  info "Cloning stable-retro into $RETRO_DIR ..."
  git clone git@github.com:Farama-Foundation/stable-retro "$RETRO_DIR"
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
# 5. Install stable-retro from source first, then MariHA without stable-retro
#    so pip does not overwrite the source build with the PyPI binary.
# ---------------------------------------------------------------------------
info "Installing stable-retro from source..."
pip install -e "$RETRO_DIR"
success "stable-retro installed."

info "Installing MariHA (remaining deps, skipping stable-retro)..."
# --no-deps would skip everything; instead we install normally but stable-retro
# is already present at the right version so pip will leave it alone.
pip install -e "$REPO_ROOT"
success "MariHA installed."

# Verify stable-retro is still the source build (not PyPI binary).
RETRO_LOCATION=$(python -c "import stable_retro; print(stable_retro.__file__)")
if [[ "$RETRO_LOCATION" != "$RETRO_DIR"* ]]; then
  warn "stable-retro may have been replaced by a PyPI binary: $RETRO_LOCATION"
  warn "Reinstalling from source..."
  pip install -e "$RETRO_DIR"
  success "stable-retro reinstalled from source."
fi

# ---------------------------------------------------------------------------
# 6. Install tensorflow[and-cuda] for GPU support
# ---------------------------------------------------------------------------
info "Installing tensorflow[and-cuda] for GPU support..."
pip install "tensorflow[and-cuda]"
success "tensorflow[and-cuda] installed."

# ---------------------------------------------------------------------------
# 7. Stimuli data check
# ---------------------------------------------------------------------------
STIMULI_DIR="$MARIHA_DATA_ROOT/mario/stimuli/SuperMarioBros-Nes"
if [[ ! -d "$STIMULI_DIR" ]]; then
  warn "Stimuli directory not found: $STIMULI_DIR"
  warn "Set MARIHA_DATA_ROOT correctly and re-run, or run manually later:"
  warn "  export MARIHA_DATA_ROOT=/your/path && bash setup_cc.sh"
else
  success "Stimuli directory found."
fi

# ---------------------------------------------------------------------------
# 8. Subject data check
# ---------------------------------------------------------------------------
SCENES_DIR="$MARIHA_DATA_ROOT/mario.scenes"
STATE_COUNT=$(find "$SCENES_DIR" -name "*.state.gz" 2>/dev/null | head -1 | wc -l | tr -d ' ')
if [[ "$STATE_COUNT" -eq 0 ]]; then
  warn "No .state.gz files found in $SCENES_DIR."
  warn "Pull with: cd $SCENES_DIR && git annex get sub-*/"
else
  success "Subject data present."
fi

# ---------------------------------------------------------------------------
# 9. Generate per-scene scenario files
# ---------------------------------------------------------------------------
info "Generating per-scene scenario files..."
if mariha-generate-scenarios; then
  success "Scenarios generated."
else
  warn "mariha-generate-scenarios failed — mastersheet may not be present yet."
  warn "Run manually after pulling data: mariha-generate-scenarios"
fi

# ---------------------------------------------------------------------------
# 10. Smoke test
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
# 11. Summary
# ---------------------------------------------------------------------------
echo ""
success "Setup complete."
echo ""
echo "Add the following to your job script (or ~/.bashrc):"
echo ""
echo "    module load StdEnv/2023 python/3.12"
echo "    source $VENV_DIR/bin/activate"
echo "    export MARIHA_DATA_ROOT=$MARIHA_DATA_ROOT"
echo ""
echo "Then run training with:"
echo "    mariha-run-cl --subject sub-01 --seed 0"
