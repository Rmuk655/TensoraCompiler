#!/usr/bin/env bash
# Discovery self-check: CMake registers one CTest per case by grepping
# CASE("<id>",...) out of each harness test's source (tensora_discover_cases
# in cmake/Tests.cmake). This asserts that grep result exactly matches the
# binary's OWN authoritative case list (`<bin> --list`, straight from the
# registry). So the tests themselves are the single source of truth, and a
# silent grep miss (e.g. a CASE that ever spans two lines) fails loudly here
# instead of quietly dropping a test. Registered as CTest INT-DISCOVERY-SYNC.
set -uo pipefail
cd "$(dirname "$0")/.."
BINDIR="${1:-build}"

# binary -> the source file whose CASE(...) lines CMake greps for it.
pairs="
test_lexer|tests/lexer/test_lexer.cpp
test_parser|tests/parser/test_parser.cpp
test_semantic|tests/semantic/test_semantic.cpp
test_graph|tests/semantic/test_graph.cpp
test_passes|tests/optimization/test_passes.cpp
test_tiling_decision|tests/codegen/test_tiling_decision.cpp
test_codegen_e2e|tests/codegen/test_codegen_e2e.cpp
"

fail=0
checked=0
while IFS='|' read -r bin src; do
  [ -n "$bin" ] || continue
  b="$BINDIR/$bin"
  [ -x "$b" ] || { echo "skip $bin (not built)"; continue; }
  runtime=$("$b" --list | sort -u)
  grepped=$(grep -oE '(CASE|TEST_CASE)\([A-Za-z0-9_]+, *"[^"]+"' "$src" | sed -E 's/.*"([^"]+)"/\1/' | sort -u)
  if [ "$runtime" != "$grepped" ]; then
    echo "DISCOVERY MISMATCH for $bin (grep of $src vs '$bin --list'):"
    diff <(printf '%s\n' "$grepped") <(printf '%s\n' "$runtime") | sed 's/^/  /'
    fail=1
  fi
  checked=$((checked + 1))
done <<< "$pairs"

[ "$fail" -eq 0 ] && echo "OK: CMake's grep discovery matches the --list registry for all $checked harness binaries"
exit $fail
