#!/usr/bin/env bash
# One-time developer bootstrap. Checks the toolchain this repo actually
# needs (cmake, a C++20 compiler, Java, LLVM/MLIR 21 -- see
# docs/development/DEPENDENCIES.md) and prints the next commands to
# run. There is no git-hook or CI setup here: formatting and linting
# are a developer's own judgment call, not an enforced gate.
#
# Usage, once after cloning:
#
#   ./tools/setup-dev.sh
set -uo pipefail
cd "$(dirname "$0")/.."

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
