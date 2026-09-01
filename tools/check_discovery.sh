#!/usr/bin/env bash
# Discovery self-check: CMake registers one CTest per case by grepping
# case IDs out of each harness test's own sources -- CASE("<id>",...) or
# TEST_CASE("<id>",...) out of a hand-written .cpp (tensora_discover_cases
# in cmake/Tests.cmake), or "### <id> | ..." out of a fixture .tns file
# (tensora_discover_fixture_cases; tests/support/FixtureLoader.h reads the
# same file at runtime). This asserts the combined grep result exactly
# matches the binary's OWN authoritative case list (`<bin> --list`,
# straight from the registry). So the tests themselves are the single
# source of truth, and a silent grep miss (e.g. a CASE that ever spans two
# lines) fails loudly here instead of quietly dropping a test. Registered
# as CTest INT-DISCOVERY-SYNC.
set -uo pipefail
cd "$(dirname "$0")/.."
BINDIR="${1:-build}"

# binary -> comma-separated "cpp:<path>" and/or "fixture:<path>" entries,
# matching exactly what cmake/Tests.cmake feeds to tensora_discover_cases /
# tensora_discover_fixture_cases for that binary.
pairs="
test_lexer|cpp:tests/lexer/test_lexer.cpp
test_parser|cpp:tests/parser/test_parser.cpp,fixture:tests/fixtures/parser.tns
test_semantic|cpp:tests/semantic/test_semantic.cpp,fixture:tests/fixtures/semantic.tns
test_graph|cpp:tests/semantic/test_graph.cpp
test_passes|cpp:tests/optimization/test_passes.cpp
test_tiling_decision|cpp:tests/codegen/test_tiling_decision.cpp
test_codegen_e2e|cpp:tests/codegen/test_codegen_e2e.cpp
"

fail=0
checked=0
while IFS='|' read -r bin entrylist; do
  [ -n "$bin" ] || continue
  b="$BINDIR/$bin"
  [ -x "$b" ] || { echo "skip $bin (not built)"; continue; }
  runtime=$("$b" --list | sort -u)

  grepped=""
  IFS=',' read -ra entries <<< "$entrylist"
  for entry in "${entries[@]}"; do
    kind="${entry%%:*}"
    path="${entry#*:}"
    if [ "$kind" = cpp ]; then
      ids=$(grep -oE '(CASE|TEST_CASE)\([A-Za-z0-9_]+, *"[^"]+"' "$path" | sed -E 's/.*"([^"]+)"/\1/')
    else
      ids=$(grep -oE '^### [A-Za-z0-9_-]+ \|' "$path" | sed -E 's/^### ([A-Za-z0-9_-]+) .*/\1/')
    fi
    grepped="$grepped
$ids"
  done
  grepped=$(printf '%s\n' "$grepped" | sed '/^$/d' | sort -u)

  if [ "$runtime" != "$grepped" ]; then
    echo "DISCOVERY MISMATCH for $bin (grep of $entrylist vs '$bin --list'):"
    diff <(printf '%s\n' "$grepped") <(printf '%s\n' "$runtime") | sed 's/^/  /'
    fail=1
  fi
  checked=$((checked + 1))
done <<< "$pairs"

[ "$fail" -eq 0 ] && echo "OK: CMake's grep discovery matches the --list registry for all $checked harness binaries"
exit $fail
