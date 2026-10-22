#!/usr/bin/env bash
# Create the rlwx conda env and build pym3t (Python bindings for DLR-RM M3T) into it.
# Works without sudo. Usage: scripts/setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MM="${MM:-$HOME/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# Pinned upstream commits (pym3t fetches the agimus fork of 3DObjectTracking/M3T at build time).
PYM3T_COMMIT=23ac5ddc2782d480d6737329dc07d23be24e99ac

if [ ! -x "$MM" ]; then
  mkdir -p "$(dirname "$MM")"
  curl -fsSL -o "$MM" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
  chmod +x "$MM"
fi

if "$MM" env list | awk '{print $1}' | grep -qx rlwx; then  # the exact name: "rlwx-train" must not count
  "$MM" install -y -n rlwx -f "$ROOT/environment.yml"
else
  "$MM" create -y -n rlwx -f "$ROOT/environment.yml"
fi

if [ ! -d "$ROOT/third_party/pym3t" ]; then
  git clone https://github.com/agimus-project/pym3t "$ROOT/third_party/pym3t"
fi
git -C "$ROOT/third_party/pym3t" checkout -q "$PYM3T_COMMIT"

"$MM" run -n rlwx bash -c '
  export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
  pip install -v --no-build-isolation "'"$ROOT"'/third_party/pym3t" \
    --config-settings=cmake.define.USE_REALSENSE=ON \
    --config-settings=cmake.build-type=Release
'
"$MM" run -n rlwx python -c "import pym3t; print('pym3t OK')"
