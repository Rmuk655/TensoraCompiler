#!/usr/bin/env bash
# The closest C++-project equivalent to a Python requirements.txt:
# there is no single manifest file, because the dependencies come from
# several different ecosystems (OS packages, LLVM/MLIR, Java; ANTLR is
# CMake-managed and does not belong here at all -- see below). This
# script is the check, not the installer: it reports what is present,
# what is missing, and the exact command to fix each gap, rather than
# blindly running `sudo apt install` on a machine it does not control.
# It also reports the optional demo/visualization dependencies (Graphviz
# for `make viz`, the upstream LLVM/MLIR CLI tools for `make
# demo-independent`, Python/venv for the real-world demos); those never
# gate the exit code, since they are not needed to build or test.
# See docs/development/DEPENDENCIES.md for the full pinned-version
# reference this script checks against.
#
# Usage: ./tools/check-deps.sh
# Exit status: 0 if every dependency is present, 1 otherwise.

set -uo pipefail
cd "$(dirname "$0")/.."

ok=0
missing=0
opt_missing=0

pass() { printf '  [ok]      %s\n' "$1"; ok=$((ok + 1)); }
fail() { printf '  [MISSING] %s\n' "$1"; missing=$((missing + 1)); }
# Optional (demo/visualization) deps: report but do NOT gate the exit
# code -- they are not needed to build or test the compiler.
note() { printf '  [opt]     %s\n' "$1"; opt_missing=$((opt_missing + 1)); }

# The LLVM/MLIR CLI tools ship in /usr/lib/llvm-21/bin, which the demos
# add to PATH themselves; accept them there even if not on PATH.
LLVM_BIN=/usr/lib/llvm-21/bin
have() { command -v "$1" >/dev/null 2>&1 || [ -x "$LLVM_BIN/$1" ]; }

echo "=== Required: build toolchain ==="

if command -v cmake >/dev/null 2>&1; then
  pass "cmake ($(cmake --version | head -1))"
else
  fail "cmake -- install with: sudo apt install -y cmake"
fi

if command -v g++ >/dev/null 2>&1; then
  cxx_version=$(g++ -dumpversion)
  if [ "${cxx_version%%.*}" -ge 10 ] 2>/dev/null; then
    pass "g++ ($cxx_version, supports C++20 per CMakeLists.txt CMAKE_CXX_STANDARD)"
  else
    fail "g++ $cxx_version is too old for C++20 -- install a newer one, e.g.: sudo apt install -y g++-13"
  fi
else
  fail "g++ -- install with: sudo apt install -y build-essential"
fi

if command -v java >/dev/null 2>&1; then
  pass "java ($(java -version 2>&1 | head -1)) -- runs the ANTLR generator jar CMake fetches"
else
  fail "java -- install with: sudo apt install -y default-jre"
fi

echo
echo "=== Required: LLVM/MLIR 21.1.8 (pinned, INTEG-003) ==="

if command -v llvm-config-21 >/dev/null 2>&1; then
  pass "llvm-config-21 ($(llvm-config-21 --version))"
else
  fail "llvm-21-dev -- install with: sudo apt install -y llvm-21-dev"
fi

if [ -d /usr/lib/llvm-21/lib/cmake/mlir ]; then
  pass "MLIR 21 CMake package (/usr/lib/llvm-21/lib/cmake/mlir)"
else
  fail "libmlir-21-dev -- install with: sudo apt install -y libmlir-21-dev mlir-21-tools"
fi

if [ -f /usr/lib/llvm-21/lib/libmlir_c_runner_utils.so ]; then
  pass "libmlir_c_runner_utils.so (needed by every JIT-executed test)"
else
  fail "libmlir_c_runner_utils.so not found -- reinstall libmlir-21-dev"
fi

echo
echo "=== Optional: demos, visualization, benchmarks ==="
echo "(not needed to build or test the compiler; each enables a demo)"

# make viz (demos/08) renders AST/TCG/MLIR graphs with Graphviz `dot`.
if command -v dot >/dev/null 2>&1; then
  pass "graphviz dot ($(dot -V 2>&1)) -- for 'make viz'"
else
  note "graphviz 'dot' -- for 'make viz': sudo apt install -y graphviz"
fi

# make demo-independent (demos/07), and viz's MLIR op-graph, feed emitted
# IR to the upstream LLVM/MLIR command-line tools.
for t in mlir-opt mlir-translate llc lli; do
  if have "$t"; then
    pass "$t (upstream LLVM/MLIR CLI) -- for 'make demo-independent'"
  else
    note "$t not on PATH or in $LLVM_BIN -- for 'make demo-independent' (part of mlir-21-tools/llvm-21)"
  fi
done

# Real-world demos (make demo-*) use Python: torch to PREPARE artifacts,
# numpy to RUN. Deps go in a project-local .venv (never system Python).
if command -v python3 >/dev/null 2>&1; then
  pass "python3 ($(python3 --version 2>&1)) -- for real-world demos"
  if [ -d .venv ]; then
    pass ".venv present (demo virtualenv, has numpy/torch once set up)"
  else
    note ".venv not created -- for 'make demo-*': ./tools/setup-dev.sh --demos"
  fi
else
  note "python3 -- for real-world demos: sudo apt install -y python3 python3-venv"
fi

echo
echo "=== Not checked here on purpose ==="
echo "  ANTLR (jar + C++ runtime): fetched reproducibly by CMake's"
echo "  FetchContent on first configure, not an OS package -- do not"
echo "  install it system-wide."

echo
echo "$ok ok, $missing missing (required), $opt_missing optional not present"
if [ "$opt_missing" -gt 0 ]; then
  echo "[opt] items are demo-only; install a line's tool only if you want that demo."
fi
if [ "$missing" -gt 0 ]; then
  echo "Fix the [MISSING] items above before running cmake -S . -B build."
  exit 1
fi
exit 0
