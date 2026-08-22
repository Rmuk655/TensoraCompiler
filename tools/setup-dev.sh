#!/usr/bin/env bash
# One-time developer bootstrap. Checks the toolchain this repo actually
# needs (cmake, a C++20 compiler, Java, LLVM/MLIR 21 -- see
# docs/development/DEPENDENCIES.md) and prints the next commands to
# run. There is no git-hook or CI setup here: formatting and linting
# are a developer's own judgment call, not an enforced gate.
#
# Usage, once after cloning:
#
#   ./tools/setup-dev.sh            # check the build toolchain only
#   ./tools/setup-dev.sh --demos    # also set up the Python venv for the
#                                    # real-world inference demos (torch etc.)
#
# The compiler itself needs no Python. `--demos` is only for the
# tools/models/prepare_*.py scripts, so it is opt-in: it pulls in torch
# and torchvision, which are large and most compiler work does not need.
set -uo pipefail
cd "$(dirname "$0")/.."

WITH_DEMOS=0
for arg in "$@"; do
  case "$arg" in
    --demos) WITH_DEMOS=1 ;;
    *) echo "unknown option: $arg (usage: ./tools/setup-dev.sh [--demos])" >&2; exit 2 ;;
  esac
done

./tools/check-deps.sh
status=$?

if [ "$status" -ne 0 ]; then
  echo
  echo "Install the missing dependencies above, then re-run ./tools/setup-dev.sh."
  exit "$status"
fi

echo
echo "All required dependencies are present. Next steps:"
echo "  cmake -S . -B build"
echo "  cmake --build build -j4"
echo "  ctest --test-dir build --output-on-failure"

if [ "$WITH_DEMOS" -eq 0 ]; then
  echo
  echo "To also run the real-world inference demos (needs torch), set up the"
  echo "Python venv with:  ./tools/setup-dev.sh --demos"
  exit 0
fi

# --demos: create a project-local virtual environment and install the
# demo dependencies INTO it. Modern Debian/Ubuntu block `pip install`
# against the system Python (PEP 668, "externally-managed-environment");
# a venv is the supported fix -- never use `pip --break-system-packages`
# for this project.
echo
echo "Setting up the Python venv for the demos (.venv/) ..."
if ! python3 -m venv .venv 2>/tmp/tensora-venv.log; then
  echo "error: could not create .venv. You may need the venv package:" >&2
  echo "  sudo apt install -y python3-venv" >&2
  echo "(details in /tmp/tensora-venv.log)" >&2
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r tools/models/requirements.txt

echo
echo "Demo dependencies installed in .venv/. To use the demos:"
echo "  source .venv/bin/activate      # then 'which python' -> .venv/bin/python"
echo "  python tools/models/prepare_mlp_demo.py        # (and prepare_cifar_demo.py, prepare_text_attn_demo.py)"
echo "  make demo-mlp-placement        # demo-cnn-image-classifier, demo-attn-text-classifier"
echo "  deactivate                     # when finished"
