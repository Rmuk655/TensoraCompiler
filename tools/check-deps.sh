#!/usr/bin/env bash
# The closest C++-project equivalent to a Python requirements.txt:
# there is no single manifest file, because the dependencies come from
# several different ecosystems (OS packages, LLVM/MLIR, Java; ANTLR is
# CMake-managed and does not belong here at all -- see below). This
# script is the check, not the installer: it reports what is present,
# what is missing, and the exact command to fix each gap, rather than
# blindly running `sudo apt install` on a machine it does not control.
# See docs/development/DEPENDENCIES.md for the full pinned-version
# reference this script checks against.
#
# Usage: ./tools/check-deps.sh
# Exit status: 0 if every dependency is present, 1 otherwise.

set -uo pipefail
cd "$(dirname "$0")/.."

ok=0
missing=0

pass() { printf '  [ok]      %s\n' "$1"; ok=$((ok + 1)); }
fail() { printf '  [MISSING] %s\n' "$1"; missing=$((missing + 1)); }

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
echo "=== Not checked here on purpose ==="
echo "  ANTLR (jar + C++ runtime): fetched reproducibly by CMake's"
echo "  FetchContent on first configure, not an OS package -- do not"
echo "  install it system-wide."

echo
echo "$ok ok, $missing missing"
if [ "$missing" -gt 0 ]; then
  echo "Fix the [MISSING] items above before running cmake -S . -B build."
  exit 1
fi
exit 0
