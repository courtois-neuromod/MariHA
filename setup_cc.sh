#!/usr/bin/env bash
# setup_cc.sh — MariHA setup for Compute Canada (no apptainer)
#
# Alternative to the apptainer workflow. Uses a plain Python venv and installs
# stable-retro from source. The existing Dockerfile and narval_test_sac_cl.sh
# are unchanged — the apptainer path remains available.
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
export MARIHA_DATA_ROOT="${MARIHA_DATA_ROOT:-$SCRATCH/MariHA/data}"
info "MARIHA_DATA_ROOT = $MARIHA_DATA_ROOT"

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
# 5. Install stable-retro then MariHA
# ---------------------------------------------------------------------------
info "Installing stable-retro from source..."
pip install -e "$RETRO_DIR" --quiet
success "stable-retro installed."

info "Installing MariHA..."
pip install -e "$REPO_ROOT" --quiet
success "MariHA installed."

# ---------------------------------------------------------------------------
# 6. Generate per-scene scenario files
# ---------------------------------------------------------------------------
info "Generating per-scene scenario files..."
if mariha-generate-scenarios 2>/dev/null; then
  success "Scenarios generated."
else
  warn "mariha-generate-scenarios failed — mastersheet may not be at $MARIHA_DATA_ROOT yet."
  warn "Run manually after data is available: MARIHA_DATA_ROOT=... mariha-generate-scenarios"
fi

# ---------------------------------------------------------------------------
# 7. Summary
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
