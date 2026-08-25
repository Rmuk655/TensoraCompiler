#!/usr/bin/env bash
# Parity test for the --polyhedral spike flag (spike/polyhedral branch).
#
# The polyhedral lowering path (linalg -> affine dialect -> dependence-driven
# affine loop fusion -> LLVM) must produce results that are numerically
# IDENTICAL to the default path. For every runnable example, run --run twice
# (default vs --polyhedral), write the full result tensor with --output, and
# assert the two tensors are equal. Exit nonzero on any mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."
T="${TENSORA:-./build/tensora}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0

for ex in examples/*.tns; do
  "$T" --run "$ex" >/dev/null 2>&1 || continue   # skip non-runnable (symbolic) examples
  "$T" --run              --output "$TMP/def.npy"  "$ex" >/dev/null 2>&1
  "$T" --polyhedral --run --output "$TMP/poly.npy" "$ex" >/dev/null 2>&1
  if python3 - "$TMP/def.npy" "$TMP/poly.npy" <<'PY'
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
sys.exit(0 if np.array_equal(a, b) else 1)
PY
  then echo "  PASS  $ex (default == --polyhedral, exact)"
  else echo "  FAIL  $ex (results differ)"; fail=1
  fi
done

[ "$fail" -eq 0 ] && echo "ALL PASS: polyhedral path is numerically identical to default" || { echo "PARITY FAILURE"; exit 1; }
