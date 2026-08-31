# Tensora Changelog

## Phase 1 — Lexer

Added:

- An ANTLR4 lexer (part of `grammar/Tensora.g4`) for the full v0.1
  token set: the five keywords, identifiers, integer and float
  literals, all operators and delimiters, and `#` line comments.
- `Token` / `TokenKind` and a source-position model (`SourceSpan`,
  `SourceBuffer`).
- Shared `Diagnostic` / `DiagnosticEngine` with stable `DiagCode`s.
- `Lexer` driver that builds tokens with spans and reports illegal
  characters (`LexIllegalChar`) while continuing to scan.
- `tensora --emit-tokens` command-line stage.
- Lexer tests: unit (keywords, identifiers, literals, punctuation,
  comments), boundary (empty input, comment-only, token at end of input),
  and diagnostics (illegal characters), run through CTest.
- CMake build and `docs/development/BUILD.md`.

Design notes:

- Matches `docs/architecture/INTERFACES.md` §1 (Token/Lexer) and
  `docs/architecture/06_COMPILER_ARCHITECTURE.md` §5.1.
- `TokenAdapter` converts each ANTLR-generated token into the project
  `Token`, with line/column tracking in C++ (`src/lexer/Lexer.cpp`).
  The parser (Phase 2) consumes the resulting token stream, per
  INTERFACES §2.

Known limitations / bugs:

- Not yet built or run in this workspace. The first build and test run
  happen on a toolchain machine, following `BUILD.md`. The
  developer/tester loop will fix any build errors found there.
- Integer literals are parsed with `std::stoll`; a value that overflows
  `int64_t` will throw rather than produce a diagnostic. To be handled when
  literal range checking is specified.

## Phase 2 — Parser

Added:

- An ANTLR4 parser (part of `grammar/Tensora.g4`); `AstBuilder` walks
  the generated parse tree and builds the AST.
- AST node set (`include/tensora/shared/Ast.h`): `Decl`, `TensorTypeSyntax`/
  `AxisSyntax`/`DimSyntax`, `Expr` and its `NameExpr`/`ScalarExpr`/
  `MatMulExpr`/`BinaryExpr`/`CallExpr` subtypes, `Arg` (positional/
  keyword, with `Name`/`Int`/`Float`/`IntList`/`AxisList` kinds),
  `Stmt`, `GraphDecl`, `AstProgram`. Syntax only -- no shapes or types;
  per-operator argument arity/kind checking is deferred to semantic
  analysis (RM 3.7-3.17), so the parser accepts any `arg` shape.
- `Parser::parseProgram() -> Result<AstProgram>` (`include/tensora/
  parser/Parser.h`, `src/parser/Parser.cpp`): best-effort AST
  construction with syntax-error recovery to the next statement or
  top-level declaration/graph, so one file can report several syntax
  errors in one pass.
- `tensora --emit-ast` command-line stage (pretty-prints the parsed
  AST; added properly in Phase 4 when the driver was rewritten into a
  multi-phase tool -- see below).
- Parser tests: unit and boundary cases (`tests/parser/test_parser.cpp`),
  including `BND-PAR-001..004` (zero-statement graph body, minimal
  legal program, tensorType axis-count extremes, conv2d's 5-argument
  call).

Design notes:

- Matches `docs/architecture/INTERFACES.md` §2 (Parser/AST) and
  `docs/architecture/06_COMPILER_ARCHITECTURE.md` §5.2.
- The grammar is LL(1)-recognizable in practice (a `statement` by
  `identifier "="`, a `returnStmt` by `"return"`); documented formally
  in `docs/language/03_LANGUAGE_REFERENCE_MANUAL.md` §3.24, "in the
  spirit of Appendix A.13 of Kernighan and Ritchie."

## Phase 3 — Semantic Analysis

Added:

- `SemanticAnalyzer::analyze(const AstProgram&) -> Result<TypedProgram>`
  (`include/tensora/semantic/SemanticAnalyzer.h`,
  `src/semantic/SemanticAnalyzer.cpp`): resolves names (input/weight
  declarations, graph-local bindings, lazy global-param promotion),
  infers/verifies types and shapes via the operator contracts, and
  builds a Typed Computation Graph per `graph` block.
- Operator contracts for every RM 3.7-3.17 rule (`include/tensora/
  semantic/Contracts.h`, `src/semantic/Contracts.cpp`): `checkMatMul`,
  `checkLinear`, `checkRelu`, `checkSoftmax`, `checkReshape`,
  `checkFlatten`, `checkTranspose`, `checkReduce` (sum/mean),
  `checkConv2D`, `checkMaxPool`, plus the shared `broadcastAxes`
  (elementwise, RM 3.8) and axis-name/size-compatibility machinery.
- The Typed Computation Graph itself (`include/tensora/shared/Ir.h`,
  `src/shared/Ir.cpp`): `Value`/`GraphNode`/`OpKind`/`NodeAttrs`/`FactSet`,
  `TypedGraph`/`TypedProgram`, and `GraphBuilder` -- the only sanctioned
  way to construct nodes/values (ARCH-008), used identically by sema
  and by hand-built test/mock fixtures.
- `GraphVerifier` (IR.md §6): defends an already-built graph's SSA,
  topological-order, and dangling-operand invariants; used both as a
  sema post-check and as the "mock-enabler contract" downstream phases
  rely on to synthesize valid fixtures without waiting on sema.
- `DimSolver` (`include/tensora/shared/DimSolver.h`) and `TypeContext`/
  `TensorType`/`Dimension` (`include/tensora/shared/Type.h`): union-find
  symbolic-dimension unification plus constant arithmetic, scoped per
  graph (RM 3.4); `describeType`/`describeDimension` for consistent
  diagnostic and driver-dump rendering.
- `SymbolTable` (`include/tensora/semantic/Symbol.h`): name/scope rules
  (RM 3.6) -- redeclaration and illegal-rebind detection.
- Semantic tests (`tests/semantic/test_semantic.cpp`): `SEM-*` (one
  positive/negative case per RM rule and DiagCode), plus the RM
  boundary corpus (`BND-MM-*`, `BND-BC-*`, `BND-RS-*`, `BND-FL-*`,
  `BND-CV-*`, `BND-RD-*`, `BND-TR-*`, `BND-DIM-*`) and TCG structural
  tests (`GR-*`, `tests/semantic/test_graph.cpp`).

Design notes:

- Matches `docs/architecture/INTERFACES.md` §3 (Sema) and
  `docs/architecture/06_COMPILER_ARCHITECTURE.md` §6; never depends on MLIR/LLVM
  (enforced by which CMake targets link which libraries -- `tensora_semantic`
  has no MLIR include paths at all).
- Axis names are a property of the surface language and the TCG; they
  are erased on lowering to MLIR (RM 3.5, 3.20), where axes become
  positional.

Known limitations / bugs:

- `Value::debugName` was originally always empty for every
  sema-produced (non-parameter) value -- `SemanticAnalyzer`'s per-
  operator `emit()` calls all passed `""`, since the surface statement's
  bound name (e.g. `h` in `h = linear(x, W1, b1)`) wasn't threaded down
  into the operator-specific analyze helpers. Fixed in Phase 4 while
  building `--emit-graph`/`--explain` (which need it to be useful) via
  an additive `GraphBuilder::setDebugName(ValueId, string)` called once
  per statement in `analyzeGraph`'s loop, rather than threading a name
  parameter through every one of the ~11 operator call sites.

## Phase 4 — Optimization & Code Generation

Added:

- **Level-1 graph optimization** (`include/tensora/optimization/GraphPass.h`,
  `src/optimization/GraphPasses.cpp`): the `GraphPass`/`PassManager`/
  `PassContext`/`PassResult` infrastructure and the 7 frozen RM 3.21
  passes -- constant folding, shape folding, dead-op elimination,
  reshape cancellation, transpose cancellation, elementwise fusion,
  linear-bias-activation fusion. 20 CTest cases / 54 checks
  (`tests/optimization/test_passes.cpp`).
- **Tensor liveness analysis** (`include/tensora/optimization/Liveness.h`,
  `src/optimization/Liveness.cpp`): per-value live intervals and peak-memory
  estimation over a `TypedGraph`; feeds a real decision (not just a
  reported number) in the Level-2 tiling pass below.
- **MLIR lowering** (`include/tensora/codegen/MlirLowerer.h`,
  `src/codegen/MlirLowerer.cpp`, plus the shared dialect registry in
  `codegen/DialectSetup.cpp`): 11 of 13 `OpKind`s lowered to `func`/
  `arith`/`linalg`/`tensor`/`math` -- matmul (rank-2 only), linear
  (fused with an immediately-following relu when tagged), relu,
  elementwise `+ - * /` (including broadcast), softmax, transpose,
  flatten, conv2d (including `padding>0` via `tensor.pad`), sum, mean,
  maxpool. General reshape and batched matmul are not implemented
  (documented via `llvm_unreachable` in the lowering switch's default
  case, not a silent gap).
- **Level-2 loop optimization (project-authored MLIR pass):**
  `TilingDecision` (`include/tensora/codegen/LoopTiling.h`, pure C++,
  no MLIR dependency) is the Tensora-authored shape-guided tiling
  decision; `MatmulTilingPass` (`include/tensora/lowering/
  MatmulTilingPass.h`, a real `PassWrapper<...,
  OperationPass<func::FuncOp>>` using MLIR's own `TilingInterface`/
  `scf::tileUsingSCF`) is the project-authored MLIR pass that applies
  it to `linalg.matmul`. Verified structurally (tiled copy has
  `scf.for`, untiled doesn't), numerically (tiled/untiled JIT results
  match a hand-verified reference), and via measured (not asserted)
  wall-clock timing.
- **Native execution:** JIT via `mlir::ExecutionEngine` (proven first
  via `tests/smoke/smoke_mlir.cpp`, then every `*_e2e.cpp` test), and a
  genuinely ahead-of-time-compiled path (`tools/codegen_mlp_native.cpp`,
  via `llvm::TargetMachine`, not the JIT) producing a real x86-64 ELF
  object file, linked by the system linker into a standalone executable
  (`test_mlp_native`) with zero MLIR/LLVM runtime dependency (verified
  with `file`/`ldd`).
- **End-to-end workloads**, each checked against a hand- or
  Python-double-computed reference: MLP (`LOW-E2E-MLP`, JIT;
  `NATIVE-001`, ahead-of-time), a 256x256x256 tiled matmul
  (`LOOP-002/003/004`), a CNN classifier (conv2d->relu->flatten->
  linear->softmax, `E2E-002`), a Transformer FFN sublayer (`E2E-003`,
  reusing already-proven lowering, no new ops), and single-head
  scaled-dot-product attention (`E2E-004`).
- **CLI driver rewrite** (`src/driver/main.cpp`, grown from a 71-line
  `--emit-tokens`-only stub into a real multi-phase driver): every flag
  `docs/language/02_LANGUAGE_TUTORIAL.md` §2.12 documents is now real --
  `--emit-tokens`, `--emit-ast`, `--check`, `--emit-graph`, `--explain`,
  `--emit-mlir`, `--emit-llvm`, `--run`, `-o`. New concrete-shaped
  example programs (`examples/mlp_concrete.tns`, `examples/
  matmul_large.tns`, `examples/cnn.tns`, `examples/attention.tns`) and
  6 CTest integration cases (`INT-FULL-MLP-CHECK/RUN`, `INT-FULL-MM-RUN`,
  `INT-FULL-CNN-RUN`, `INT-FULL-ATTN-RUN`, `INT-FULL-SYMBOLIC-REJECT`)
  exercise the full source-to-execution pipeline, distinct from every
  e2e test above (which builds its `TypedGraph` by hand in C++).
- 156 CTest cases passing as of this phase, stable across repeated runs.

Design notes:

- Matches `docs/architecture/06_COMPILER_ARCHITECTURE.md` §7 (rewritten this phase
  to describe what's actually implemented, not the earlier aspirational
  `Backend`/`CodegenOptions` abstraction) and INTERFACES §4.
- Three layers are kept explicit and distinguishable throughout:
  Tensora's Level-1 TCG passes, the project-authored Level-2
  `MatmulTilingPass`, and standard MLIR/LLVM infrastructure
  (bufferization, loop/control-flow/arith/memref-to-LLVM conversion,
  `ConvertMathToLLVM`, `ExpandStridedMetadata`, `LowerAffine`).

Known limitations / bugs:

- **Not implemented:** general reshape (arbitrary regrouping beyond
  flatten's fixed batch-preserving collapse), batched matmul (so
  multi-head attention type-checks but does not lower or run), average
  pooling.
- **Symbolic-dimension lowering is broken** (LOW-005): `lowerTensorType`
  correctly lowers a symbolic dim to a dynamic `?` *type*, but every
  lowering function that materializes a fresh tensor calls
  `tensor.empty()` with zero dynamic-size operands, so any graph with a
  symbolic dimension fails MLIR verification. The CLI driver rejects
  this case before lowering with a clear message rather than crashing.
- `linalg.matmul`/`linalg.conv_2d_nchw_fchw`/`linalg.pooling_nchw_max`
  all *accumulate* into their `outs` operand rather than initializing
  it; an uninitialized `tensor.empty()` accumulator produced a
  genuinely intermittent (sometimes correct, sometimes garbage) test
  failure that only repeated runs caught. Fixed with an explicit
  zero-fill (or `-inf`-fill for the max-pooling/max-reduction case) via
  `linalg.fill` before the op.
- `scf`'s bufferizable-op-interface was never registered in the shared
  dialect registry, so tiling-introduced nested `scf.for` loops with
  tensor `iter_args` could not bufferize (a confusing "op was not
  bufferized" error). Fixed via `scf::
  registerBufferizableOpInterfaceExternalModels`.
- `memref.subview` (introduced by tiling) does not lower to LLVM via
  `FinalizeMemRefToLLVMConversion` alone -- it needs
  `ExpandStridedMetadataPass` first, and that pass's own emitted
  `affine.apply`/index arithmetic needs `LowerAffinePass` run *before*
  `ArithToLLVM`, not after (an ordering bug, not a missing-pass bug).
- `mlir::ExecutionEngine::invokePacked` does not add the
  `"_mlir_ciface_"` prefix itself (unlike the templated `invoke()`
  helper, which does) -- passing the plain function name caused a
  silent segfault rather than a clean lookup error while building the
  CLI driver's `--run`. Worked around by looking up
  `_mlir_ciface_<name>` directly and calling it as a raw function
  pointer, dispatched over argument count via a `switch`, matching the
  calling convention already proven reliable in every JIT test this
  phase.

