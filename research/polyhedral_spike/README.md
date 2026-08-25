# Polyhedral research: index

The polyhedral spike is done. Most of it is now settled, not open
research, so its files moved out of this directory:

- **The flag, its tests, and its docs are in `main` (v0.1)**, as
  **ARCH-018** (`docs/architecture/DECISIONS.md`). The `--polyhedral`
  flag lowers `linalg` through the MLIR affine dialect and runs
  `affine-loop-tile`. It uses no Tensora-specific semantics, so it
  carried no research risk. Correctness: `tests/cli/cli_tests.txt`, the
  `INT-POLY-*` CTest cases, on `main`.
- **The open research question, and the findings that shaped it
  (tiling works, tiling alone is not enough, the fusion bug, why it is
  not evidence for a semantic-legality claim) are in
  [SEMANTIC_POLYHEDRAL.md](SEMANTIC_POLYHEDRAL.md)** in this directory.
  Read that file for the V0.3 direction.
- **The MLIR bug found along the way is filed upstream**, in
  [mlir_bug_report/](mlir_bug_report/) (`BUG_REPORT.md`, `ISSUE_BODY.md`,
  and the reproducers). Already submitted; do not change.

Everything else that was here (the standalone `mlir-opt` pipeline demo,
the CNN-level fusion diagnosis script, and the raw matmul perf-sweep
scripts) is removed. Their numbers and conclusions are folded into
`SEMANTIC_POLYHEDRAL.md`; their code was either superseded by the real
`--polyhedral` flag in `main`, or by the smaller, cleaner reproducers in
`mlir_bug_report/`.
