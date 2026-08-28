# Tensora Test Plan

> Tester document. Test suites come directly from the Language
> Reference Manual ([../language/03_LANGUAGE_REFERENCE_MANUAL.md](../language/03_LANGUAGE_REFERENCE_MANUAL.md))
> and the architecture ([../architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md),
> [../architecture/IR.md](../architecture/IR.md),
> [../architecture/INTERFACES.md](../architecture/INTERFACES.md)). The
> test suite is a second, executable reading of the specification:
> every RM rule maps to at least one test. [TEST_MATRIX.md](TEST_MATRIX.md)
> tracks rule-to-test coverage.

## 1. Principles

- **Rule-driven.** Each numbered RM rule (3.1–3.23) has tests. A rule
  with no test is not treated as a requirement.
- **Valid / near-miss / invalid triples.** For each semantic rule: one
  case that passes, one case that *just barely* passes, and one case
  that fails for exactly that rule.
- **Diagnostics are contracts.** A negative test checks the `DiagCode`
  (INTERFACES.md §0), the source line, and the key expected/actual
  values (for example `784 != 512`) — not the exact wording.
- **Stable IDs.** Every case has an ID (`LEX-001`, `SEM-MM-003`, and so
  on), used in the matrix, the corpus, and the regression log.
- **Run continuously.** Each owner runs their phase's suites during
  development. CI runs every suite on every integration change.

## 2. Suites and Layout

``` text
tests/
├── lexer/         token-dump golden tests + illegal-char diagnostics
├── parser/        AST-dump golden tests + syntax-error recovery
├── semantic/      the 100-case corpus (valid/near-miss/invalid)
├── graph/         TCG invariants via GraphBuilder/GraphVerifier
├── optimization/  per-pass before/after + negative + equivalence
├── lowering/      MLIR-dump golden tests (normalized)
├── end_to_end/    compile + run + compare to reference
└── fixtures/      hand-built ASTs and TypedPrograms (mocks, ARCH-008)
```

Harness: each `--emit-*` stage (INTERFACES.md §6) produces a
deterministic artifact, checked against a golden file. The harness
checks a diagnostic structurally: code, line, and captured values.

## 2a. Test Taxonomy — the Five Test Types

Every test has two classifications: by **phase** (the suites in
Sections 3–9) and by **type** (this section). These are independent —
one phase suite holds tests of several types. Each type below must be
**exhaustive** against the specification. Each type has an explicit
"exhaustive when" rule, so you can check completeness directly.
Golden-file comparison, differential checking against a reference, and
mutation are *techniques* used within these types. They are not
separate types.

| Type | Question it answers | Scope | Dependencies | When |
|------|--------------------|-------|--------------|------|
| Unit | Is this component correct in isolation? | one function/class | mocked / hand-built fixtures | every build, fastest |
| Smoke | Is the system alive and the toolchain wired? | shallow path through a subsystem | real | first in CI; MLIR ones in **week 1** |
| Integration | Do real phases compose across their interface? | phase-pair or sub-pipeline | real components | every build |
| Boundary | What happens at the edges of each rule? | targeted limit inputs | as needed | every build |
| User (acceptance) | Does the tool serve the user and pass the viva? | whole CLI, UX-judged | real, via CLI | pre-milestone + CI subset |

**Unit** (realized by the phase suites, Sections 3–8). *Exhaustive
when:* every public function in INTERFACES.md has one or more unit
tests; every operator contract (`checkMatMul` … `checkMaxPool`) has a
valid, near-miss, and invalid case; and each of these has a direct
test: `DimSolver` (unify over Const/Symbol pairs, `sameElementCount`),
`SymbolTable` (declare/lookup/scope/redeclare), each `Lexer` token
kind, each AST production, `GraphBuilder` plus every `GraphVerifier`
invariant, `LivenessAnalysis` (interval, bytes, peak), each of the 6
graph passes, and `TilingDecision.decide`. Some unit test produces
every `DiagCode`.

**Smoke** — Section 2b. **Integration** — Section 9a. **Boundary** —
Section 9b. **User/acceptance** — Section 9c.

Complementary suites cut across all five types: regression (Section
13), mutation (Section 11), and differential/performance testing (run
inside integration and user tests).

## 2b. Smoke Tests

Smoke tests are shallow, broad, fail-fast checks that a subsystem or
external dependency is reachable. They make no deep assertions. Two
run in **week 1, before feature work**, because they de-risk the
highest-risk dependency (MLIR/LLVM) and the whole Level-2 tiling story,
before the team builds anything on top of it:

| ID | Checks | Week 1? |
|----|--------|---------|
| SMOKE-MLIR | a C++ program builds a trivial `func.func` + `linalg.matmul`/`arith` module, lowers to LLVM, and runs — proves the pinned MLIR/LLVM link and execute | **yes** |
| SMOKE-TILING | MLIR's tiling / transform interface is callable from the pinned MLIR version on a trivial matmul loop nest — de-risks the required Level-2 transformation (ARCH-009) | **yes** |
| SMOKE-BUILD | `cmake` configure + build succeeds; ANTLR generates `TensoraLexer`/`TensoraParser` and compiles | **yes** |
| SMOKE-CLI | `tensora --check hello.tns` exits 0 on a one-line valid program | |
| SMOKE-E2E | `tensora matmul.tns -o m && ./m` runs and emits output (no numeric assert) | |

*Exhaustive when:* one smoke exists per external dependency (ANTLR,
MLIR construct+lower+run, MLIR tiling transform, LLVM object emit) and per
"is it alive" subsystem (CLI, full run). If SMOKE-MLIR or SMOKE-TILING fail
in week 1, the tiling scope (ARCH-009) is reassessed before the team builds
on it.

## 3. Lexer Tests (RM 3.1)

Cover keywords, identifiers, int literals, float literals, each operator
and delimiter, `#` comments, whitespace/newline insensitivity, illegal
characters, and `SourceSpan` correctness (line/col of each token).

- Golden token dumps for each language keyword and a representative
  program.
- Negative: `@@`, a stray `$`, an unterminated float `3.` — each expects a
  `LexIllegalChar` (or lexer-level) diagnostic at the right column and
  continued scanning (multiple errors in one file).

## 4. Parser Tests (RM 3.2)

Cover each production: declarations, `Tensor[...]` types with named axes,
graph bodies, assignments, matmul/binary/call expressions, keyword args
(`stride=`, `axis=`, `perm=[...]`), and the reshape axis-list argument.

- Golden AST dumps for the MLP, CNN, FFN, and attention programs
  (whitepaper §10).
- Statement delimiting: verify `h = relu(h)` followed by `y = ...` parses
  as two statements without a terminator.
- Negative + recovery: missing `=`, missing `]`, `return` absent, keyword
  arg without value — each expects a `ParseExpected`/`ParseUnexpectedToken`
  diagnostic and resynchronization so a second error in the same file is
  also reported.

## 5. Semantic Tests (RM 3.5–3.19) — largest suite, 30% of grade

Required academic-compiler checks, each with valid/near-miss/invalid:

- **Redeclaration** (RM 3.6): duplicate global name; duplicate graph name.
- **Undefined identifier** (RM 3.6): use before declaration/binding;
  `return` of an unbound name.
- **Scope** (RM 3.6): rebinding within a graph is legal (shadowing);
  a name from one graph is not visible in another.
- **Type mismatch** (RM 3.8): scalar where a tensor operand is required.
- **Operator arity / argument kind** (RM 3.7–3.17): `linear` with two
  args; `conv2d` missing `stride`; `softmax` with a numeric axis.

Tensora-specific checks (the differentiators):

- **matmul** (RM 3.7): valid `[.., m=M,k=K]@[..,k=K,n=N]`; invalid
  `784 != 512`; batched leading-axis mismatch.
- **linear** (RM 3.9): bias name/size compatible with `W.out`; bias size
  mismatch; missing bias.
- **elementwise + broadcasting + axis names** (RM 3.8): equal shapes;
  size-1 broadcast; **axis-name mismatch with equal sizes**
  (`[batch,feat] + [feat,batch]` -> `SemaAxisNameMismatch`); incompatible
  sizes.
- **reshape** (RM 3.12): `768 -> [12,64]` valid; `768 -> [10,64]` invalid;
  symbolic-factor mismatch.
- **flatten** (RM 3.13): concrete tail flattens; symbolic tail rejected.
- **transpose** (RM 3.14): valid perm; repeated index; out-of-range index.
- **reductions / softmax** (RM 3.11, 3.15): existing axis by name;
  nonexistent axis name.
- **conv2d / maxpool** (RM 3.16, 3.17): channel unify vs mismatch; exact
  vs non-dividing spatial arithmetic; the worked CNN shapes to `flat=200704`.
- **unresolvable shape** (RM 3.4): a program needing reasoning outside the
  bounded subset -> `SemaUnresolvableShape`, not a crash.

## 6. Graph Tests (RM 3.18, 3.20; IR.md)

Using `GraphBuilder`/`GraphVerifier` on hand-built fixtures:

- SSA single-producer; closed operands; topological order; single reachable
  output.
- Producer/consumer queries correct.
- Negative: a fixture that violates an invariant (cycle, dangling operand,
  missing type) is rejected by `GraphVerifier`.
- `--emit-graph` golden dumps for the four reference programs.

## 6a. Analysis Tests — tensor liveness (ARCH-009)

`LivenessAnalysis` is a data-flow analysis, so it is tested directly:

- **Interval correctness:** on a hand-built chain `%1 -> %2 -> %3`, assert
  each value's `[firstDef, lastUse]` interval.
- **Byte sizing:** a value of `Tensor[batch=4, feat=256]` reports
  `4*256*4` bytes; a symbolic dim is reported symbolically or via a
  supplied concrete binding.
- **Peak-memory:** on a graph with a known live pattern, assert
  `peakLiveBytes` equals the hand-computed maximum of simultaneously-live
  sizes.
- **Decision use:** assert that a transformation/report actually consumes
  the peak estimate (e.g. a scheduling choice changes when the estimate
  changes).

## 7. Optimization Tests (RM 3.21) — Level-1, per pass

For each of the six graph passes (a seventh, constant folding, was
implemented and then retired -- DECISIONS.md ARCH-014 -- since it
cannot be proven to preserve RM 3.21's exact observable-output
requirement under f32 non-associativity), six kinds (tester playbook):

1. a pattern that **must** transform;
2. a near-miss that **must not** transform;
3. with symbolic dimensions present;
4. multiple occurrences in one graph;
5. interaction with another pass (ordering);
6. semantic equivalence (numerical, via end-to-end where executable).

Examples: `TransposeCancellation` cancels `transpose(transpose(x))` but not
two non-inverse perms; `LinearBiasActivationFusion` fuses `linear`+`relu`
but not `linear`+`softmax`; `DeadOpElimination` drops a value not reaching
`return` but keeps one that does. Each records before/after `--emit-graph`.

## 7a. Loop-Optimization Tests — matmul tiling (Level-2, ARCH-009)

- **Decision correctness:** `TilingDecision` chooses to tile a large
  matmul and to skip a tiny one; tile sizes are functions of the verified
  `M/N/K` (assert the plan for known shapes).
- **Structural:** the tiled loop nest has the expected tile/point loop
  structure (`--emit-mlir` after the transform, normalized).
- **Numerical equivalence:** tiled and untiled executables produce
  outputs equal within f32 tolerance on the matmul-heavy workload.
- **Performance:** record wall-clock time tiled vs untiled (reported, not
  asserted, since it is machine-dependent).

## 8. Lowering Tests (RM 3.22)

- Normalized MLIR-dump golden tests (`--emit-mlir`) for representative
  nodes: `linalg.matmul`, `linalg.conv_2d_nchw_fchw`, elementwise/relu via
  `linalg.generic`, softmax expansion, `tensor` reshape/transpose,
  `linalg.reduce`.
- Structured-loop lowering: `linalg.matmul` -> `scf` loops produces the
  expected `for i, j, k` nest (the input to Level-2 tiling).
- Verify concrete dims lower to static tensor dims and symbolic dims to
  dynamic (`?`).
- MLIR text is normalized (SSA value renumbering, whitespace) before
  comparison to avoid brittle goldens.

## 9. End-to-End Tests (RM 3.22)

Compile and run, comparing to a trusted reference (a small NumPy script):

- **MLP**, **matmul-heavy model** (the tiling workload, §7a),
  **CNN classifier**, **Transformer FFN**, **multi-head attention**
  (whitepaper §10).
- Fixed random seeds for inputs/weights; compare outputs within an f32
  tolerance.
- Differential: generate small random valid graphs, run compiled vs
  reference.

## 9a. Integration Tests

Real phases composed across the INTERFACES.md boundaries (**no mocks**),
asserting the boundary contract. One per adjacent interface, plus a
full-pipeline test per reference workload.

- **INT-LEXPARSE:** source → AST via the real ANTLR-generated lexer and parser (golden AST).
- **INT-PARSESEMA:** source → typed SSA graph (golden `--emit-graph`).
- **INT-SEMAOPT:** typed graph → optimized graph via real passes
  (before/after).
- **INT-ANALYSISDEC:** real `LivenessAnalysis` feeds a real decision (a
  peak-memory-driven choice actually changes with the estimate).
- **INT-GRAPHMLIR:** typed graph → MLIR via real lowerer (golden
  `--emit-mlir`).
- **INT-MLIRLLVM:** MLIR → LLVM → object; links and loads.
- **INT-FULL:** source → executable for each reference workload
  (MLP, matmul-heavy, CNN).

*Exhaustive when:* every adjacent interface in INTERFACES.md has an
integration test exercised with real components, and every reference
workload has one full-pipeline integration test.

## 9b. Boundary Tests

The edges and limits of each rule, where off-by-one and degenerate cases
live. For each numeric/structural rule: **minimum, maximum, and
just-inside vs just-outside**.

- **Type/tensor:** rank-1; rank-4 (conv); a size-1 axis (broadcast edge);
  a symbolic-only shape.
- **matmul:** `K=1`; `M=1` or `N=1`; symbolic-vs-concrete contraction.
- **broadcasting:** size-1 on each side; ranks differing by one; both
  operands symbolic and equal.
- **reshape/flatten:** product exactly equal vs off-by-one (near-miss);
  flatten a single trailing dim; flatten with a symbolic trailing dim
  (the rejected edge).
- **conv/pool:** output formula exact-divide vs remainder (rejected);
  `padding=0, stride=1` minimum; kernel = input size (output = 1);
  `stride = kernel` (pooling exact).
- **reductions/softmax/transpose:** axis 0; last axis; the only axis;
  identity perm; full-reverse perm.
- **bounded solver:** a program just **inside** the affine-in-one-variable
  subset vs just **outside** it (→ `SemaUnresolvableShape`) — the
  decidability boundary (RM 3.4).
- **liveness:** a value never used; used only at `return`; used
  immediately; all values live simultaneously.
- **tiling:** a dim exactly divisible by the tile; dim = tile size; dim <
  tile size (remainder / no-tile); dim = 1.
- **lexer/parser:** empty file; comments-only file; a token at EOF with no
  trailing newline; a single-statement graph; a graph body that is only
  `return`.

*Exhaustive when:* every RM rule with a numeric or structural limit has
min, max, and just-inside/just-outside cases, and every Level-2 tile-vs-dim
relation is covered.

## 9c. User (Acceptance) Tests

The tool from the user's perspective — judged on workflow and diagnostic
quality, not internals; driven through the CLI.

- **USER-CLI:** each stage flag (`--emit-tokens/ast/graph/mlir/llvm`,
  `--check`, `explain`, `-o`) produces the documented artifact and exit
  code.
- **USER-LEARN:** first-time-engineer flow — write the MLP, `--check`
  passes, `explain` shows the shapes from the tutorial, `-o` builds, the
  run matches the reference.
- **USER-FIXIT:** a shape bug is rejected with a located diagnostic naming
  the values (`784 != 512`); after the fix it compiles.
- **USER-DIAG:** for every error class, the rendered diagnostic carries
  category + location + expected/actual and is human-readable (the
  `DiagCode` UX contract).
- **USER-VIVA:** the two required non-trivial **working** programs run
  correctly and the two required **failing** programs are rejected with the
  right diagnostics (academic compiler requirement).
- **USER-EXPLAIN:** `explain` output for the reference programs matches the
  tutorial.

*Exhaustive when:* every CLI flag is covered; every error class has a
diagnostic-quality check; the four viva programs are scripted and pass.

## 10. The 100-Case Semantic Corpus

Split (all frozen features are in v1, so attention cases count):

``` text
10  basic language semantics (decl/scope/redecl/undefined/type-mismatch)
15  matmul / linear
10  broadcasting (incl. axis-name mismatch cases)
10  reshape / flatten
10  transpose / layout
5   reductions / softmax
15  conv / pooling
10  graph semantics (SSA, dead nodes, output reachability)
10  attention / multi-head (whitepaper §10.4 shape + axis-name cases)
5   diagnostics / recovery (multi-error files, poisoning)
--- 
100
```

Each case: ID, RM rule reference, program, expected result
(accept / reject + `DiagCode` + line), and a one-line rationale. Tracked
in [TEST_MATRIX.md](TEST_MATRIX.md).

## 11. Mutation Testing

Take the valid reference programs and mutate one thing — a dimension, a
rank, an axis name, an operator argument, a dependency — and assert the
mutant is rejected with the expected `DiagCode`. This systematically
generates near-miss negatives from known-good programs.

## 12. Real-World Bug Corpus (open-source / research value, separate)

A distinct corpus derived from *public* framework issues or real model
code. Each entry records: case ID, source URL/repo, framework, operator,
original failure, minimal reproducer, the Tensora program, whether
Tensora detects it statically, and the diagnostic. Never fabricate a
"real-world" case — only cases with a real, cited source.

## 13. Regression Policy

Every confirmed bug becomes a regression test that fails on the buggy
build and passes after the fix, logged in [REGRESSION_LOG.md](REGRESSION_LOG.md).
Open defects live in [KNOWN_BUGS.md](KNOWN_BUGS.md).

## 14. Coverage Goals

Track semantic coverage, not just lines: grammar-production coverage,
operator coverage (all 15 frozen ops), semantic-rule coverage (RM
3.5–3.19), Level-1 pass coverage (all 6), **analysis coverage**
(liveness intervals plus peak-memory), **Level-2 coverage** (tiling
decision plus equivalence), and error-category coverage (every
`DiagCode`). Target: map 100% of RM rules; give every operator
positive and negative tests; test the liveness analysis and the tiling
decision directly; exercise every `DiagCode`.

Additionally track **test-type coverage**: each of the five types
(Section 2a) must meet its stated "exhaustive when" criterion — unit
(every INTERFACES function + every contract + every DiagCode), smoke (every
external dependency + subsystem), integration (every adjacent interface +
every workload full-pipeline), boundary (min/max/just-inside/just-outside
for every limited rule), and user (every CLI flag + every error class +
the four viva programs).

## 15. CI

Ordered so the cheapest, most diagnostic failures surface first: (1)
**smoke** (Section 2b) — configure/build, ANTLR generate, MLIR
construct+lower+run, MLIR tiling reachable, CLI alive; (2) **unit**
(Sections 3–8); (3) **integration** (9a); (4) **boundary** (9b); (5) the
phase suites and the semantic corpus; (6) a **user/acceptance** subset (9c,
including the viva programs). All via CTest; the main branch stays green.

The two week-1 smoke tests (SMOKE-MLIR, SMOKE-TILING) are a **gate**: until
both pass on the pinned MLIR/LLVM, the Level-2 tiling deliverable
(ARCH-009) is treated as unproven and the integrator escalates before the
team builds the loop-optimization phase on it.
