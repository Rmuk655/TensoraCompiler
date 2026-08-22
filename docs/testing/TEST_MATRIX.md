# Tensora Test Matrix

> Maps every Language Reference Manual rule to test cases. Kept in sync
> with [../language/03_LANGUAGE_REFERENCE_MANUAL.md](../language/03_LANGUAGE_REFERENCE_MANUAL.md)
> and [08_TEST_PLAN.md](08_TEST_PLAN.md). Status: `planned` until the case
> exists and runs in CI. A rule with no row is a coverage gap.

## Legend

- **Result**: `accept` (compiles clean) or `reject` (+ expected `DiagCode`).
- **Kind**: `V` valid, `N` near-miss valid, `X` invalid.
- IDs are stable and referenced by the corpus and regression log.

## Frontend — Lexer (RM 3.1)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| LEX-001 | 3.1 | V | keywords + identifiers, incl. keyword-prefix maximal munch | tokens | done |
| LEX-002 | 3.1 | V | int and float literal values, incl. leading-zero int | tokens | done |
| LEX-003 | 3.1 | V | every operator/delimiter, spaced | tokens | done |
| LEX-004 | 3.1 | V | every operator/delimiter with no whitespace (adjacency) | tokens | done |
| LEX-005 | 3.1 | V | comment variants (empty, no-newline, back-to-back, contents ignored) | tokens | done |
| LEX-006 | 3.1 | V | whitespace/newline insensitivity across separator styles | kinds equal | done |
| LEX-007 | 3.1 | X | malformed floats `3.` `.5` `3.x` `1..2` | reject: LexIllegalChar | done |
| LEX-008 | 3.1 | X | illegal-char variety incl. 7 in a row; scanning continues | reject: LexIllegalChar | done |
| LEX-009 | 3.1 | V | `SourceSpan` line/col/offset incl. tabs + CRLF; lexeme == source slice | spans correct | done |
| LEX-010 | 3.1 | V | full representative program, golden token-kind dump | tokens | done |
| LEX-011 | 3.1 | X | int literal at/over int64 max (LANG-001; fixed REG-001) | reject: LexIntegerOverflow | done |

Boundary cases for RM 3.1 are tracked as `BND-LEX-*` in the Boundary
section below (empty file, comments-only file, token at EOF, first-token
position).

## Frontend — Parser (RM 3.2, 3.3, 3.24)

Audited against the RM 3.24 grammar summary (the authoritative syntax).
Every production and alternative below maps to at least one case; the
previous 8-row table only exercised a handful of representative shapes
and no error-recovery or grammar-fidelity (decl\* before graphDecl+)
cases, so it did not meet 08_TEST_PLAN.md's "exhaustive when: every AST
production has a direct test" criterion (§2a Unit).

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| PAR-001 | 3.2, 3.24 | V | MLP program (whitepaper 10.1) -> AST golden | ast | done |
| PAR-002 | 3.2, 3.24 | V | CNN program (whitepaper 10.2) -> AST golden | ast | done |
| PAR-003 | 3.2, 3.24 | V | Transformer FFN program (whitepaper 10.3) -> AST golden | ast | done |
| PAR-004 | 3.2, 3.24 | V | multi-head attention program (whitepaper 10.4) -> AST golden | ast | done |
| PAR-005 | 3.3 | V | `tensorType`/`axis`: 1 axis, 2 axes, 4 axes; concrete and symbolic `dimSize` | ast | done |
| PAR-006 | 3.2 | V | `decl`: `inputDecl` alone, `weightDecl` alone, both, multiple of each | ast | done |
| PAR-007 | 3.2, 3.24 | V | `program`: multiple `graphDecl`s (`graphDecl+` repetition) | ast | done |
| PAR-008 | 3.2 | V | `statement`: two statements with no terminator; a bare-identifier statement (`y = h`) | ast | done |
| PAR-009 | 3.24 | V | `expression`/`binop`: each of `@ + - * /` with identifier and float-literal `operand` | ast | done |
| PAR-010 | 3.24 | V | `callExpr`: one positional arg, multiple positional args | ast | done |
| PAR-011 | 3.24 | V | `arg` kinds: keyword arg (int value), keyword arg (identifier value), reshape axis-list arg, all three mixed in one call | ast | done |
| PAR-012 | 3.12, 3.14, 3.24 | V | `argValue` kinds: `intLit` (`stride=1`), identifier (`axis=kseq`), `intList` (`perm=[0,1,3,2]`) | ast | done |
| PAR-013 | 3.16, 3.24 | V | `conv2d`-shaped call: multiple positional + multiple keyword args together | ast | done |
| PAR-014 | 3.2 | X | missing `=` in a statement | reject: ParseExpected | done |
| PAR-015 | 3.3 | X | missing `]` closing a `tensorType` | reject: ParseExpected | done |
| PAR-016 | 3.3 | X | missing `[` opening a `tensorType` | reject: ParseExpected | done |
| PAR-017 | 3.3 | X | empty `tensorType` (`Tensor[]`; `axis` list requires >= 1) | reject: ParseExpected | done |
| PAR-018 | 3.3 | X | missing `=` in an `axis` | reject: ParseExpected | done |
| PAR-019 | 3.2, 3.24 | X | `graphDecl` missing `returnStmt` before `}` | reject: ParseExpected | done |
| PAR-020 | 3.2 | X | `graphDecl` missing closing `}` | reject: ParseExpected | done |
| PAR-021 | 3.24 | X | `returnStmt` missing its identifier | reject: ParseExpected | done |
| PAR-022 | 3.24 | X | zero-arg call (`relu()`; `arg` list requires >= 1) | reject: ParseExpected | done |
| PAR-023 | 3.24 | X | missing closing `)` on a call | reject: ParseExpected | done |
| PAR-024 | 3.24 | X | trailing comma in an arg list (`relu(h,)`) | reject: ParseExpected | done |
| PAR-025 | 3.24 | X | keyword arg with no value (`stride=)`) | reject: ParseExpected | done |
| PAR-026 | 3.2, 3.24 | X | a `decl` after a `graphDecl` (violates `decl* graphDecl+` ordering) | reject: ParseExpected | done |
| PAR-027 | 3.2, 3.24 | X | a program with zero `graphDecl`s (`graphDecl+` requires >= 1) | reject: ParseExpected | done |
| PAR-028 | recovery | X | two independent syntax errors in different sections (one in a `decl`, one in a graph body) both reported | 2 diagnostics | done |
| PAR-029 | recovery | X | two independent malformed statements in the same graph body both reported | 2 diagnostics | done |
| PAR-030 | span | X | a missing-token diagnostic's `SourceSpan` (line/col) points at the actual offending token | span correct | done |
| PAR-031 | regression | X | bare int literal as a binop operand (BUG-002/REG-002: used to segfault) | reject: diagnostic, not a crash | done |
| PAR-032 | regression | X | illegal axis `dimSize` token (BUG-003/REG-003: used to segfault) | reject: diagnostic, not a crash | done |

Boundary cases for RM 3.2/3.3/3.24 are tracked as `BND-PAR-*` below (empty
statement list, minimal program, axis-count and arg-count extremes).

**Known coverage gap:** `DiagCode::ParseUnexpectedToken` is declared but
currently unreachable. `ParserErrorListener` (`src/parser/`) genuinely
distinguishes the two syntax-error shapes ANTLR can report: a
`NoViableAltException` (no expected token can be named) becomes
`ParseUnexpectedToken`; every other `RecognitionException` (a mismatched-
or missing-token error, which does name an expected set) becomes
`ParseExpected`. Every malformed input tried against this grammar --
including cases designed to look ambiguous, like a bare `f(@)` or an
empty `Tensor[]` -- still resolves to a mismatched-token error, because
ANTLR's adaptive LL(*) prediction picks the one viable alternative before
failing, in every state this small (~24-token) grammar can reach. Not
fixed further here since forcing `NoViableAltException` would require
deliberately engineering an unrealistic grammar state rather than testing
real language behavior.

## Semantic — required course checks (RM 3.6, 3.8, 3.7–3.17)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-DECL-001 | 3.6 | X | duplicate global name | reject: SemaRedeclaration | done |
| SEM-DECL-002 | 3.6 | X | duplicate graph name | reject: SemaRedeclaration | done |
| SEM-DECL-003 | 3.6 | X | graph-local assignment shadows a global input/weight name (different kind) | reject: SemaRedeclaration | done |
| SEM-DECL-004 | 3.3 | X | duplicate axis name within one tensor type (ARCH-013) | reject: SemaDuplicateAxisName | done |
| SEM-DECL-005 | 3.3 | X | duplicate axis name in a `reshape` target axis list (ARCH-013) | reject: SemaDuplicateAxisName | done |
| SEM-NAME-001 | 3.6 | X | use of undefined name | reject: SemaUndefinedName | done |
| SEM-NAME-002 | 3.6 | X | `return` of unbound name | reject: SemaUndefinedName | done |
| SEM-NAME-003 | 3.24 | X | call to an operator name outside the frozen set | reject: SemaUndefinedName | done |
| SEM-SCOPE-001 | 3.6 | V | rebinding `h = relu(h)` (shadowing) | accept | done |
| SEM-SCOPE-002 | 3.6 | X | name from graph A used in graph B | reject: SemaUndefinedName | done |
| SYM-001 | 3.6 | V | `SemanticAnalyzer::symbolTable()`: globals recorded in declaration order with correct kind/type | 2 globals, `x`(Input) then `W`(Weight), verified decl type | done |
| SYM-002 | 3.6 | V | graph-local symbol recorded with its inferred (not declared) type | `h`, kind Value, relu's inferred type | done |
| SYM-003 | 3.6 | V | shadowing (`h = relu(h)`) collapses to one recorded local entry | 1 entry, not 2 | done |
| SYM-004 | 3.6 | V | resolution log: `global::`/`local::`, deduped per distinct referenced name | 3 entries for x/h/y referenced across 3 statements | done |
| SYM-005 | 3.6 | X | resolution log records an unresolved reference as `undefined`, not silently absent | logged, no crash | done |
| SYM-006 | 3.6 | X | a rejected global redeclaration never enters the recorded global scope | 1 entry (the original), not 2 | done |
| SYM-007 | 3.6 | X | a rejected local rebind of a global name never enters the recorded local scope | 0 local entries | done |
| SYM-008 | 3.6 | V | multiple graphs each get their own recorded scope, in program order | 2 graph scopes, no cross-graph leakage | done |
| SEM-TYPE-001 | 3.8 | X | scalar where tensor operand required (`x @ 0.5`) | reject: SemaTypeMismatch | done |
| SEM-ARITY-001 | 3.9 | X | `linear` with 2 args | reject: SemaArgCount | done |
| SEM-ARITY-002 | 3.16 | X | `conv2d` missing `stride` | reject: SemaArgCount | done |
| SEM-ARITY-003 | 3.11 | X | `softmax` axis given as integer | reject: SemaArgKind | done |
| SEM-ARITY-004 | 3.24 | X | `relu` with an unknown keyword argument | reject: SemaArgKind | done |
| SEM-ARITY-005 | 3.14 | X | `transpose` missing `perm` | reject: SemaArgCount | done |
| SEM-POISON-001 | ARCH-007 | X | one bad statement does not cascade into spurious follow-on errors later in the same graph | 1 diagnostic only | done |
| SEM-POISON-002 | recovery | X | two independent semantic errors (different graphs) are both reported in one pass | 2 diagnostics | done |

## Semantic — matmul / linear (RM 3.7, 3.9)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-MM-001 | 3.7 | V | `[batch=B,feat=784]@[in=784,out=256]` | accept -> `[batch=B,out=256]` | done |
| SEM-MM-002 | 3.7 | X | `784 != 512` | reject: SemaShapeMismatch (784!=512) | done |
| SEM-MM-003 | 3.7 | N | symbolic contraction `K` unifies | accept | done |
| SEM-MM-004 | 3.7 | X | batched leading-axis mismatch | reject: SemaShapeMismatch | done |
| SEM-LIN-001 | 3.9 | V | linear with matching bias | accept | done |
| SEM-LIN-002 | 3.9 | X | bias size mismatch | reject: SemaShapeMismatch | done |
| SEM-LIN-003 | 3.9 | X | bias axis-name mismatch | reject: SemaAxisNameMismatch | done |

## Semantic — elementwise / broadcasting / axis names (RM 3.8)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-EW-001 | 3.8 | V | equal shapes and names | accept | done |
| SEM-EW-002 | 3.8 | V | size-1 broadcast | accept | done |
| SEM-EW-003 | 3.8 | X | **equal sizes, swapped axis names** | reject: SemaAxisNameMismatch | done |
| SEM-EW-004 | 3.8 | X | incompatible sizes | reject: SemaBroadcast | done |
| SEM-EW-005 | 3.8 | V | tensor-scalar `x * 0.125` | accept | done |

## Semantic — reshape / flatten (RM 3.12, 3.13)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-RS-001 | 3.12 | V | `768 -> [head=12,dim=64]` | accept | done |
| SEM-RS-002 | 3.12 | X | `768 -> [head=10,dim=64]` (640) | reject: SemaReshapeCount (640!=768) | done |
| SEM-RS-003 | 3.4, 3.12 | X | symbolic-factor multiset mismatch (`[batch=B,feat=768]` vs `[batch=C,head=12,dim=64]`, unrelated symbols) -- outside the bounded subset (RM 3.4), not a decided count mismatch | reject: SemaUnresolvableShape | done |
| SEM-RS-004 | 3.12 | N | symbolic factor cancels correctly (`[batch=B,feat=768]` -> `[batch=B,head=12,dim=64]`), near-miss of RS-003 | accept | done |
| SEM-FL-001 | 3.13 | V | concrete tail flattens to `flat=200704` | accept | done |
| SEM-FL-002 | 3.13 | X | symbolic tail dim | reject: SemaUnresolvableShape | done |

## Semantic — transpose / reductions / softmax (RM 3.14, 3.11, 3.15)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-TR-001 | 3.14 | V | valid perm on rank-4 | accept | done |
| SEM-TR-002 | 3.14 | X | repeated perm index | reject: SemaTransposePerm | done |
| SEM-TR-003 | 3.14 | X | out-of-range perm index | reject: SemaTransposePerm | done |
| SEM-RD-001 | 3.15 | V | `sum(x, axis=feat)` existing axis | accept (axis removed) | done |
| SEM-RD-002 | 3.15 | X | nonexistent axis name | reject: SemaAxisNotFound | done |
| SEM-SM-001 | 3.11 | V | `softmax(scores, axis=kseq)` | accept | done |
| SEM-SM-002 | 3.11 | X | softmax nonexistent axis | reject: SemaAxisNotFound | done |

## Semantic — conv / pooling (RM 3.16, 3.17)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-CV-001 | 3.16 | V | conv `stride=1,padding=1` preserves HxW | accept | done |
| SEM-CV-002 | 3.16 | X | channel mismatch `3 != 16` | reject: SemaConvChannel | done |
| SEM-CV-003 | 3.16 | X | non-dividing spatial arithmetic | reject: SemaConvArithmetic | done |
| SEM-CV-004 | 3.16 | X | stride=2 at 224x224 that would only floor-divide (a common-framework convention) is rejected, not floored | reject: SemaConvArithmetic | done |
| SEM-CV-005 | 3.16 | V | stride=2 at 224x224 exactly halves H/W when the division is exact | accept, output [64,112,112] | done |
| SEM-PL-001 | 3.17 | V | `maxpool size=2` halves HxW | accept | done |
| SEM-PL-002 | 3.17 | N | `stride` defaults to `size` | accept | done |

## Semantic — bounded solver (RM 3.4)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| SEM-DIM-001 | 3.4 | V | equality/unification of symbolic `B` | accept | done |
| SEM-DIM-002 | 3.4 | X | requires nonlinear reasoning (same shape as SEM-RS-003, exercised generically) | reject: SemaUnresolvableShape | done |
| SEM-DIM-003 | 3.3 | X | a zero-sized dimension (ARCH-013) | reject: SemaNonPositiveDimension | done |

**SEM-RS-003 / SEM-DIM-002 note:** `DimSolver::sameElementCount` (refining
`INTERFACES.md`'s illustrative `bool` signature to an
`ElementCountResult`, see `shared/DimSolver.h`) distinguishes two failure
modes: symbolic factors that match in identity/multiplicity but whose
constant products differ (`SemaReshapeCount`, a *decided* mismatch --
SEM-RS-002) vs. symbolic factors that don't match at all, which is
outside the bounded affine-in-one-variable subset (RM 3.4) and always
`SemaUnresolvableShape` regardless of which operator triggered it
(SEM-RS-003, SEM-DIM-002). This corrects the matrix's earlier draft,
which had listed SEM-RS-003 as `SemaReshapeCount`.

## Semantic — attention / multi-head (whitepaper §10.4)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| ATT-001 | 3.7, 3.8, 3.11, 3.14 | V | the full attention program (whitepaper 10.4) type-checks end to end: batched `@`, rank-4 `transpose`, scalar `*`, `softmax(axis=kseq)` | accept | done |
| ATT-002 | 3.8 | X | mutated ATT-001: swap `Q`'s and `K`'s leading axis names (`batch`/`head` transposed on one operand) so the batched matmul's leading-axis broadcast fails | reject: SemaAxisNameMismatch | done |

## Graph (RM 3.18, 3.20; IR.md)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| GR-001 | 3.20 | V | MLP program analyzed end to end: golden structural TCG (node count/kinds/order, param count) | graph | done |
| GR-002 | IR §1 | X | hand-built `GraphBuilder` fixture with a cycle | GraphVerifier reject: IrInvariantViolation | done |
| GR-003 | IR §1 | X | hand-built fixture with a dangling operand (out-of-range ValueId) | GraphVerifier reject: IrInvariantViolation | done |
| GR-004 | 3.18 | V | producer/consumer queries correct on a small hand-built chain | accept | done |
| GR-005 | 3.20 | V | single reachable output; every node in topological order | accept | done |
| GR-006 | ARCH-008 | V | a `GraphVerifier`-clean fixture built entirely by hand (no sema) passes, demonstrating the mock-enabler contract | accept | done |
| GR-007 | 3.18 | V | a graph referencing the same global input twice adds it as exactly one param (lazy dedup) | 1 param, 2 consumers | done |

`--emit-graph`'s textual dump format (IR.md §7) is not implemented by this
phase (driver/CLI work); GR-001's "golden" comparison is a structural
C++ assertion over the built `TypedGraph`, matching how parser golden
tests already work in this project (no invented textual format).

## Optimization — per pass (RM 3.21)

| ID | Pass | Kind | Case | Result | Status |
|----|------|------|------|--------|--------|
| OPT-SF-001/002 | ShapeFolding | V/N | constant shape -> literal / symbolic shape untouched | changed / unchanged | done |
| OPT-DCE-001 | DeadOpElimination | V | drops value not reaching return | changed | done |
| OPT-DCE-002 | DeadOpElimination | N | keeps value reaching return | unchanged | done |
| OPT-DCE-003 | DeadOpElimination | V | removes a whole dead chain (2 nodes) next to a same-shaped live chain, not just one dead node | changed (2 nodes removed) | done |
| OPT-RC-001/002/003 | ReshapeCancellation | V/V/N | reshape immediately undone / chained / non-inverse shapes | changed / changed / unchanged | done |
| OPT-TC-001/002/003 | TransposeCancellation | V/N/V | inverse perms cancel / non-inverse perms / chained inverse | changed / unchanged / changed | done |
| OPT-EF-001/002 | ElementwiseFusion | V/N | chain fuses / non-fusable interruption | changed / unchanged | done |
| OPT-LBA-001 | LinearBiasActivationFusion | V | linear+relu fuses | changed | done |
| OPT-LBA-002 | LinearBiasActivationFusion | N | linear+softmax not fused | unchanged | done |
| OPT-FIX-001 | PassManager (fixed point) | V | DeadOpElimination drops a linear's dead consumer; a later sweep then fuses the surviving relu | changed, 2+ sweeps | done |
| OPT-FIX-002 | PassManager (fixed point) | N | the same pair, run through one sweep only (no re-run) | not tagged | done |
| OPT-EQ-001 | all | V | numerical equivalence before/after | equal (e2e) | partial — LOW-E2E-MLP/NATIVE-001 verify LinearBiasActivationFusion+DeadOpElimination numerically end to end; not yet repeated per-pass for the other 6 |

All 6 passes and their `GraphPass`/`PassManager` infrastructure live in
`include/tensora/optimization/GraphPass.h` + `src/optimization/GraphPasses.cpp`,
exercised by `tests/optimization/test_passes.cpp` (20 CTest cases, 54 checks,
all passing).

## Analysis — tensor liveness (ARCH-009)

| ID | Ref | Kind | Case | Result | Status |
|----|-----|------|------|--------|--------|
| LIVE-001 | ARCH-009 | V | live intervals on a producer/consumer chain | intervals correct | done |
| LIVE-002 | ARCH-009 | V | byte size of a concrete-dim tensor | matches hand calc | done |
| LIVE-003 | ARCH-009 | V | peak-memory = max simultaneously-live | matches hand calc | done |
| LIVE-004 | ARCH-009 | V | a decision consumes peakLiveBytes | decision changes with estimate | done — feeds `TilingDecision` (LOOP-001-F), a real consumer, not just a reported number |

`include/tensora/optimization/Liveness.h` + `src/optimization/Liveness.cpp`;
exercised by `tests/optimization/test_passes.cpp`'s LIVE-* cases (4 CTest
cases).

## Loop optimization — matmul tiling (Level-2, ARCH-009, USP)

| ID | Ref | Kind | Case | Result | Status |
|----|-----|------|------|--------|--------|
| LOOP-001-A..F | 8.4 | V/N | large/tiny/exact-divide/remainder/symbolic-skip/memory-pressure `TilingDecision` cases | plan matches shapes (pure C++, no MLIR) | done |
| LOOP-002 | 8.4 | V | tiled loop nest structure after `MatmulTilingPass` | tiled copy has `scf.for`, untiled copy does not | done |
| LOOP-003 | 8.4 | V | tiled vs untiled numerical equivalence | both compute the hand-verified reference value, JIT-executed | done |
| LOOP-004 | 8.4 | V | tiled vs untiled wall-clock, 1024x1024x1024, tile=128 | reported (real speedup, single-run methodology, not a rigorous claim on its own) | done |
| LOOP-005-A/B | ARCH-009 | V/V | `MlirLowerer`'s own `LivenessAnalysis` call (not a hand-built `LivenessResult`) drives the tile-size choice through the real lowering path | tile_sizes={32,32,32} with no memory pressure; {16,16,16} once an unused 1.37 MiB weight alone pushes peak live bytes over the 1 MiB budget | done |
| LOOP-005-C | ARCH-009 | V | a plain 512x512x512 matmul (no extra weight fixture) already exceeds the default 1 MiB budget from its own operand sizes (3 MiB total) | tile_sizes={16,16,16} | done |
| LOOP-005-D | ARCH-009 | V | the identical 512x512x512 matmul, lowered under a 4 MiB `TilingPolicy::memoryBudgetBytes` instead of the 1 MiB default | tile_sizes={32,32,32} -- confirms the "budget" is a policy field compared against a static shape-based estimate, not a real memory measurement, so modeling a machine with more/less available memory for this decision is exactly changing this one field, not an OS-level memory limit on the process | done |

**Measured tile-size evidence** (`tools/benchmark_matmul_tiling`, not a
CTest case — a reporting tool, best-of-5-runs per config, this
machine, 1024x1024x1024 f32 matmul):

| config | seconds (best of 5) | vs. untiled |
|---|---|---|
| untiled | ~2.96–3.44 | 1.00x |
| 16x16x16 | ~1.30–1.34 | ~2.3–2.6x |
| 32x32x32 | ~1.32–1.37 | ~2.2–2.5x |
| 64x64x64 | ~1.32–1.35 | ~2.2–2.6x |

Run `make benchmark` (or `build/benchmark_matmul_tiling` directly) to
reproduce on your own machine —
these are real measurements, not assumed numbers, and this project does
not claim they generalize beyond this workload/machine. All three tile
sizes beat untiled by roughly the same margin here; the differences
*between* 16, 32, and 64 are within run-to-run noise on this machine, so
this workload does not, by itself, identify a single "optimal" tile
size among the three -- it does show tiling itself is worth doing, and
that the existing default (32) is a reasonable, evidence-checked choice
among options that are empirically close, not a value picked by feel.

`TilingDecision` (`include/tensora/codegen/LoopTiling.h`, pure C++,
tested in `tests/codegen/test_tiling_decision.cpp`) is the
Tensora-authored decision; `MatmulTilingPass`
(`include/tensora/codegen/MatmulTilingPass.h`, a real
`PassWrapper<..., OperationPass<func::FuncOp>>` using MLIR's own
`TilingInterface`/`scf::tileUsingSCF`) is the project-authored MLIR pass
that applies it — this is the Level-2 / USP deliverable. Both are
distinguished from standard MLIR/LLVM infrastructure (bufferization,
loop/control-flow/arith/memref-to-LLVM conversion) in
`tests/codegen/test_tiling_e2e.cpp`'s pipeline, which runs the
tiling pass first, then the standard lowering passes, then JIT-executes
and compares results.

## Lowering (RM 3.22)

| ID | RM | Kind | Case | Result | Status |
|----|----|------|------|--------|--------|
| LOW-001 | 3.22 | V | matmul (rank 2) -> `linalg.matmul` (zero-filled accumulator); matmul (rank >= 2 with leading batch axes, incl. broadcasting) -> `linalg.generic` contraction; linear (rank(x) 1 or >= 3) -> `linalg.generic` batched vector-matrix contraction, no separate M axis | numerically verified via JIT (LOW-E2E-MLP, LOOP-002/003, NATIVE-001, RANK-E2E) | done |
| LOW-002 | 3.22 | V | conv -> `linalg.conv_2d_nchw_fchw`, zero-filled accumulator, `tensor.pad` when padding>0, channel-broadcast bias | numerically verified via JIT, padding=0 (LOW-009, E2E-002) and padding=1 (LOW-009b, exercises the `tensor.pad` path specifically) | done |
| LOW-003 | 3.22 | V | relu/bias -> `linalg.generic` (incl. linear+relu fusion into one generic body) | numerically verified via JIT (LOW-E2E-MLP, NATIVE-001) | done |
| LOW-004 | 3.22 | V | softmax expansion (max-shift, `math.exp`, sum, normalize — 4 chained `linalg.generic` ops) | numerically verified via JIT against a non-trivial (non-saturated) input (LOW-008, E2E-002, E2E-004) | done |
| LOW-005 | 3.22 | V | concrete dim -> static; symbolic -> dynamic `?` | `lowerTensorType` implements this per ARCH-005 | broken (confirmed, not just untested) — `lowerTensorType` correctly lowers a symbolic dim to a dynamic `?` *type*, but every lowering function that materializes a fresh tensor (`tensor.empty()` in elementwise, relu, matmul/conv2d's zero-filled accumulator, etc.) calls it with zero dynamic-size operands, so a module with any symbolic dimension fails MLIR verification (`incorrect number of dynamic sizes`). Found by running the real `tensora` CLI driver against `examples/mlp.tns` (a symbolic-batch source file that already passes semantic analysis); the driver now rejects symbolic shapes before lowering with a clear message (INT-FULL-SYMBOLIC-REJECT) rather than hitting this. Fixing it properly means threading a same-shape existing SSA value (to query dynamic dims from via `tensor.dim`) through every lowering function -- a real, separate piece of work, not a one-line patch. |
| LOW-006 | 3.14 | V | transpose -> `linalg.transpose` | numerically verified via JIT (LOW-006, E2E-004) | done |
| LOW-007 | 3.13 | V | flatten -> `tensor.collapse_shape` (reassociation `[[0],[1..rank-1]]`) | numerically verified via JIT (LOW-007, E2E-002) | done |
| LOW-010 | 3.15 | V | sum -> rank-reducing `linalg.generic` reduction (drops the named axis entirely, unlike softmax's keep-dims) | numerically verified via JIT (LOW-010) | done |
| LOW-011 | 3.15 | V | mean -> sum + divide-by-axis-size elementwise pass | numerically verified via JIT (LOW-011) | done |
| LOW-012 | 3.17 | V | maxpool -> `linalg.pooling_nchw_max` (-inf-filled accumulator, shape-only window operand) | numerically verified via JIT (LOW-012) | done |
| LOW-013 | 3.8 | V | elementwise add with real tensor-tensor broadcasting (a shorter-rank operand, not a tensor-scalar op) -> `linalg.generic` with `broadcastOperandMap` | numerically verified via JIT (LOW-013): `[[1,2,3],[4,5,6]] + [10,20,30]` | done |

General reshape (`tensor.reshape` with a runtime shape operand —
flatten's fixed collapse-to-2D reassociation was sufficient for every
workload attempted so far) is still not in `MlirLowerer`'s switch — out
of scope for now, not silently miscompiled (the switch's default case
is `llvm_unreachable`, not a fallthrough). Batched matmul (RANK-003/004)
and rank-1/rank>=3 linear (RANK-001/002) are now lowered -- this closed
a real language/backend gap: sema accepted these ranks (RM 3.7/3.9) but
MlirLowerer previously only handled rank 2, which would have produced
an invalid `linalg.matmul` rather than a clean rejection. Multi-head
attention (which needs batched matmul) is not otherwise attempted;
LOW-006..009/E2E-004 still only demonstrate single-head attention.
Every RM 3.x frozen operator except general reshape is now lowered.

## End-to-End (RM 3.22, whitepaper §10)

| ID | Kind | Case | Result | Status |
|----|------|------|--------|--------|
| E2E-001 | V | MLP compile+run vs hand-computed reference | within tol (exact match) | done — `LOW-E2E-MLP` (JIT) and `NATIVE-001` (ahead-of-time object file + system linker + standalone executable) both compute linear->relu->linear = 3.0 |
| E2E-002 | V | CNN classifier vs reference | within tol (1e-4) | done — `test_cnn_e2e.cpp`: conv2d(stride=2)->relu->flatten->linear->softmax, checked against a Python-double-computed reference (0.539902); actual JIT result 0.539915, well within f32 tolerance |
| E2E-003 | V | Transformer FFN vs reference | within tol (exact match) | done — `test_ffn_e2e.cpp`: position-wise linear->relu->linear at FFN shapes (seq_len=2, d_model=4, hidden=8), reusing the same lowering LOW-E2E-MLP already proved (no new ops needed); residual connection and layer-norm are not modeled (neither is a lowered op) |
| E2E-004 | V | (single-head) scaled dot-product attention vs reference | within tol (1e-4) | done for single head — `test_attention_e2e.cpp`: matmul->transpose->scale->softmax->matmul, checked against a Python-double-computed reference (1.660477); multi-head specifically is not started (needs batched matmul) |
| E2E-005 | V | matmul-heavy model, tiled vs untiled vs reference | within tol + timing | done — `LOOP-002/003/004` (single 8x8x8 matmul; timing reported, not asserted, since the workload is too small for tiling to pay off) |
| E2E-006 | V | differential random small graphs | — | not started |

E2E-001/002/004/005 use a hand-computed (or, for E2E-002/004, a
Python-double-precision-computed) reference value rather than a literal
NumPy invocation from the C++ test suite (no Python/NumPy runtime
dependency introduced) — the numerical rigor (an exact expected value,
checked via real JIT and, for E2E-001, native execution) is the same
either way.

## Smoke (TEST_PLAN §2b) — week-1 gate marked

| ID | Checks | Week 1 | Status |
|----|--------|--------|--------|
| SMOKE-MLIR | trivial `func.func`+`linalg.matmul` builds, lowers to LLVM, JIT-runs, checks 2x2 matmul result | yes (gate) | done |
| SMOKE-TILING | MLIR tiling/transform interface callable on a matmul loop nest | yes (gate) | done — folded into `LOOP-002` rather than a separate smoke case, since `MatmulTilingPass` (the real deliverable) already exercises exactly this |
| SMOKE-BUILD | cmake configure+build; ANTLR generates `TensoraLexer`/`TensoraParser` from `grammar/Tensora.g4` & compiles | yes | done — verified via a full `cmake -S . -B build && cmake --build build` from a clean tree; INTEG-003 additionally pins the MLIR/LLVM 21 apt packages required |
| SMOKE-CLI | `tensora --check hello.tns` exits 0 | | planned |
| SMOKE-E2E | `tensora matmul.tns -o m && ./m` runs | | planned |

## Integration (TEST_PLAN §9a) — real phases, no mocks

| ID | Boundary (INTERFACES) | Case | Result | Status |
|----|-----------------------|------|--------|--------|
| INT-LEXPARSE | Lexer→Parser | source → AST (golden) | ast | planned |
| INT-PARSESEMA | Parser→Sema | source → typed SSA graph (golden) | graph | planned |
| INT-SEMAOPT | Sema→Passes | typed graph → optimized (before/after) | changed | planned |
| INT-ANALYSISDEC | Analysis→Decision | liveness estimate drives a choice | decision varies | done — `LIVE-004`/`LOOP-001-F`: a `LivenessResult` over budget selects `TilingDecision`'s conservative tile size instead of the default |
| INT-GRAPHMLIR | Lowering | typed graph → MLIR (golden) | mlir | partial — verified via numerical JIT/native-execution correctness (LOW-E2E-MLP, LOOP-002/003, NATIVE-001), not a literal golden-MLIR-text diff |
| INT-MLIRLLVM | Backend | MLIR → LLVM → object | links | done — `NATIVE-001`: `codegen_mlp_native` emits a real x86-64 ELF object via `llvm::TargetMachine` (verified with `file`), linked by the system linker into a standalone executable (verified with `ldd`: only libc, no MLIR/LLVM runtime) |
| INT-FULL-MLP | full pipeline | MLP source → executable | runs | done — the real `tensora` CLI (`src/driver/main.cpp`) now runs `examples/mlp_concrete.tns` through Lexer→Parser→SemanticAnalyzer→Level-1 passes→`MatmulTilingPass`→`MlirLowerer`→standard MLIR/LLVM lowering, both JIT (`--run`, `INT-FULL-MLP-RUN`, hand-verified result 145 for all-ones synthetic input) and ahead-of-time object-file emission (`-o`, verified via `file`) |
| INT-FULL-MM | full pipeline | matmul-heavy source → executable | runs | done — `examples/matmul_large.tns` (256x256x256, above `MatmulTilingPass`'s default tiling threshold) compiles and JIT-runs through the same CLI (`INT-FULL-MM-RUN`); `--emit-mlir` shows 3 nested `scf.for` loops confirming the tiling pass actually fired, and the result (256, the contracted dimension, for all-ones inputs) is hand-verified |
| INT-FULL-CNN | full pipeline | CNN source → executable | runs | done — `examples/cnn.tns` compiles and JIT-runs through the same CLI (`INT-FULL-CNN-RUN`), hand-verified result 0.5/0.5 (symmetric logits from an all-ones conv+linear stack) |
| INT-FULL-SYMBOLS | Sema (RM 3.6) | `--emit-symbols` on a non-trivial program | golden symbol-table text | done — `INT-FULL-SYMBOLS`: `examples/mlp.tns` through the real CLI, checked against a `PASS_REGULAR_EXPRESSION` golden fragment; full determinism/scope/shadowing/error-path coverage is in `SYM-001`..`008` above |

The CLI driver (`tensora --emit-tokens/--emit-ast/--check/--emit-symbols/--emit-graph/--explain/--emit-mlir/--emit-llvm/--run/-o`)
did not exist before this pass; building and testing it against real `.tns` source is what surfaced the LOW-005
symbolic-dimension bug above -- exactly the value of testing the full pipeline rather than only its hand-built-fixture
unit tests. `--run`/`-o`/`--emit-mlir`/`--emit-llvm` all require concrete (non-symbolic) shapes for this reason
(enforced with a clear error, `INT-FULL-SYMBOLIC-REJECT`); `--emit-tokens`/`--emit-ast`/`--check`/`--emit-graph`/
`--explain` work on any well-formed program, symbolic or not, since they stop before lowering.
`examples/attention.tns` (single-head) is also verified this way but has no dedicated INT-FULL-ATTN row in
RM/whitepaper terms -- see `INT-FULL-ATTN-RUN`. Batched matmul itself is no longer a gap (RANK-003/004 above); a
rank-4 multi-head attention example is simply not yet written, not blocked on missing lowering support.

## Boundary (TEST_PLAN §9b) — min / max / just-inside / just-outside

| ID | Rule | Case | Result | Status |
|----|------|------|--------|--------|
| BND-MM-001 | 3.7 | matmul `K=1` | accept | done |
| BND-MM-002 | 3.7 | matmul `M=1` or `N=1` | accept | done |
| BND-BC-001 | 3.8 | broadcast size-1 each side | accept | done |
| BND-RS-001 | 3.12 | reshape product off by one | reject: SemaReshapeCount | done |
| BND-FL-001 | 3.13 | flatten symbolic trailing dim | reject: SemaUnresolvableShape | done |
| BND-CV-001 | 3.16 | conv output exact-divide vs remainder | accept / reject | done |
| BND-CV-002 | 3.16 | kernel = input size (output 1) | accept | done |
| BND-RD-001 | 3.15 | reduce axis 0 / last / only axis | accept | done |
| BND-TR-001 | 3.14 | identity perm; full-reverse perm | accept | done |
| BND-DIM-001 | 3.4 | just inside vs just outside affine subset | accept / SemaUnresolvableShape | done |
| BND-LIVE-001 | ARCH-009 | value never used / only at return / all live | intervals correct | planned |
| BND-TILE-001 | 8.4 | dim divisible / = tile / < tile / = 1 | plan correct | planned |
| BND-LEX-001 | 3.1 | empty file | only eof, at line 1 col 1 | done |
| BND-LEX-002 | 3.1 | comments-only file (single + multiple lines) | only eof | done |
| BND-LEX-003 | 3.1 | token at EOF, no trailing newline, per token category | token then eof | done |
| BND-LEX-004 | 3.1 | first token of any file starts at line 1, col 1, offset 0 | span correct | done |
| BND-PAR-001 | 3.2, 3.24 | graph body that is only `return` (zero statements) vs a single statement | ast | done |
| BND-PAR-002 | 3.2, 3.24 | minimal legal program: zero decls, exactly one graph, return-only body | ast | done |
| BND-PAR-003 | 3.3 | `tensorType` axis-count extremes: 1 axis (min) vs 4 axes (conv-style) | ast | done |
| BND-PAR-004 | 3.16, 3.24 | call arg-count extreme: `conv2d`'s 5-arg call (3 positional + 2 keyword) | ast | done |

## User / Acceptance (TEST_PLAN §9c) — CLI + diagnostics + viva

| ID | Case | Result | Status |
|----|------|--------|--------|
| USER-CLI-001 | each stage flag emits documented artifact + exit code | correct | planned |
| USER-LEARN-001 | MLP: check → explain → build → run vs reference | matches tutorial | planned |
| USER-FIXIT-001 | shape bug rejected with located values; fix compiles | clear diagnostic | planned |
| USER-DIAG-001 | every error class: category+location+expected/actual | UX contract met | planned |
| USER-VIVA-001 | 2 working programs run correctly | pass | planned |
| USER-VIVA-002 | 2 failing programs rejected with right diagnostics | pass | planned |
| USER-EXPLAIN-001 | `explain` output matches tutorial for reference programs | matches | planned |

## Coverage roll-up

| Dimension | Target | Notes |
|---|---|---|
| RM rules mapped | 100% | 3.1–3.23 all have >=1 row |
| Operators (15) | pos + neg each | matmul,linear,+,-,*,/,relu,softmax,reshape,flatten,transpose,sum,mean,conv2d,maxpool — **lowered: matmul (rank 2 and rank >= 2 batched/broadcasting), linear (rank 1, rank 2, rank >= 3), relu, +,-,*,/ (elementwise, incl. broadcast), softmax, flatten, transpose, conv2d (incl. padding), sum, mean, maxpool; not yet lowered: general reshape (arbitrary regrouping beyond flatten's fixed collapse, documented scope boundary, `MlirLowerer`'s default case is `llvm_unreachable`, not silent)** |
| Symbol table (RM 3.6) | direct | `SymbolTable`/`SymbolTableReport` (SYM-001..008, done) — declaration order, scope isolation, shadowing, both error paths; `--emit-symbols` golden output (`INT-FULL-SYMBOLS`, done) |
| Analysis | direct | tensor liveness intervals + peak-memory (LIVE-*, done) — feeds a real decision in `TilingDecision`, not just a reported number |
| Loop opt (Level-2 / USP) | decision + equivalence | matmul tiling: `TilingDecision` (LOOP-001-A..F, done) + project-authored `MatmulTilingPass` (LOOP-002/003/004, done — structural, numerical, and reported timing, all via real JIT execution) |
| DiagCodes | all exercised | every code in INTERFACES.md §0 |
| Passes (6) | positive + near-miss each | see TEST_PLAN §7; constant folding was implemented then retired (DECISIONS.md ARCH-014) |
| 100-case corpus | tracked | split in TEST_PLAN §10 |
| Test type: unit | exhaustive | phase sections; every INTERFACES fn + contract + DiagCode |
| Test type: smoke | exhaustive | SMOKE-* ; per dependency + subsystem; week-1 gate |
| Test type: integration | exhaustive | INT-* ; every adjacent interface + every workload |
| Test type: boundary | exhaustive | BND-* ; min/max/just-in/just-out per limited rule |
| Test type: user | exhaustive | USER-* ; every CLI flag + error class + 4 viva programs |
