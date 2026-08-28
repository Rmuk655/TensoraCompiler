# 6. Compiler Architecture

> Written by the System Architect. This is the report-facing overview;
> the full authoritative detail (module boundaries, interfaces, decision
> records) lives in [ARCHITECTURE.md](ARCHITECTURE.md),
> [IR.md](IR.md), [INTERFACES.md](INTERFACES.md), and
> [DECISIONS.md](DECISIONS.md) (ARCH-001..ARCH-011). This document must
> not contradict those; it summarizes them for the final report.

## 6.1 Overview

``` text
Source (.tns)
  |
Lexer (ANTLR) / Parser (ANTLR)
  |
AST
  |
Semantic Analyzer (attribute grammar)
  |
Typed Computation Graph (TCG)
  |
Level-1 optimization: 6 graph passes + tensor-liveness/peak-memory analysis
  |
MLIR Lowering (func / linalg / tensor / arith / scf / math)
  |
Level-2 optimization: project-authored shape-guided matmul tiling (MLIR pass)
  |
Standard MLIR/LLVM lowering (bufferization, loop/control-flow/arith/memref
  -to-LLVM conversion, ExpandStridedMetadata, LowerAffine, ConvertMathToLLVM)
  |
LLVM IR -> native object (via llvm::TargetMachine) -> linked executable
```

The team has built and run this pipeline end to end. 183 passing tests
cover every phase. Two tests build real ahead-of-time native
executables: the object file comes from `llvm::TargetMachine`, the
system linker links it, and `ldd` confirms it has zero MLIR/LLVM
runtime dependency. Other tests JIT-run real workloads: an MLP, a CNN
classifier, a Transformer feed-forward block, single-head scaled-dot-
product attention, and a tiled-vs-untiled matmul benchmark.

## 6.2 Architectural Principle

Compiler phases talk to each other through stable data, not through
each other's internal code. The two main IRs are the **AST** (pure
syntax) and the **Typed Computation Graph** (verified semantics,
ARCH-001). At the TCG/lowering boundary and the AST/sema boundary, mock
fixtures let the team build and test each phase on its own
(`GraphBuilder`/`GraphVerifier`, ARCH-008). Every MLIR-lowering test in
this project builds its `TypedGraph` by hand this way. These tests do
not need the lexer, parser, or semantic analyzer at all.

## 6.3 Frontend

One ANTLR4 grammar (`grammar/Tensora.g4`, ARCH-011) creates the lexer
and parser. The grammar has no embedded
actions. So team-owned code (`src/parser/AstBuilder.cpp`) builds the
AST by walking the parse tree. Shared diagnostics convert ANTLR errors
at one clear point (`ParserErrorListener`, `TokenAdapter`). The lexer,
parser, and semantic analyzer never include an MLIR/LLVM header. Only
`codegen/` depends on MLIR/LLVM (Section 6.12).

## 6.4 AST

Representative nodes: `Program`, `InputDecl`, `WeightDecl`, `GraphDecl`,
`TensorType`, `Assignment`, `CallExpr`, `BinaryExpr`, `ReturnStmt`.

## 6.5 Source and Diagnostic Layer

Every AST node carries a `SourceSpan`. Diagnostics contain severity,
message, span, notes, and optional expected/actual semantic information.

## 6.6 Symbol Table and Type Environment

`SymbolTable` (`include/tensora/semantic/Symbol.h`) is the semantic
environment. It maps identifiers to their declarations or bindings and
their types, and it enforces scope rules (RM 3.6). It has only two
scopes, matching the grammar: a flat global scope (`declareGlobal`/
`lookupGlobal`, filled once from every top-level `input`/`weight`,
shared by every graph) and one flat local scope per graph (`bindLocal`,
reset by `enterGraph()` at the start of each graph). Every operand
reference in `SemanticAnalyzer` uses one lookup path, `resolve`: it
checks the local scope first, then the global scope. `bindLocal`
returns `BindOutcome::KindConflict` instead of inserting when a local
rebind would shadow a global input/weight (RM 3.6 rejects a rebind to a
different kind). A rebind of an existing *local* value is always
accepted; it updates that entry in place (shadowing).

`SemanticAnalyzer::symbolTable()` returns a `SymbolTableReport`. This
report is a declaration-order snapshot of exactly what `SymbolTable`
accepted during that `analyze()` call: the global declarations, one
scope per graph with its final local bindings, and a log of what each
referenced name resolved to. The report comes straight from the same
`SymbolTable` instance that did the resolution — the compiler does not
build a second, separate copy. `tensora --emit-symbols` (main.cpp)
prints this report; see INTERFACES.md §3 and whitepaper §4.1 for the
full output format and reasons. The symbol table itself does not
survive past semantic analysis. It resolves source names into the
`ValueId`s the Typed Computation Graph stores, and no later phase reads
it.

## 6.7 Tensor Type Representation

A tensor type is an ordered list of axes. Each axis stores a **name**
(an identifier) and a **size** (a concrete integer or a symbolic
variable). Every tensor's element type is f32 in v1 (whitepaper Section
2.3). A dtype field is a future extension, not part of the v1 design.

## 6.8 Constraint Engine

Operator checks build a small, bounded set of constraints — dimension
equality, element-count equality — through a `DimSolver` (union-find
plus constant arithmetic, ARCH-006). The solver has no SMT and no
general nonlinear solving. If a program needs more than this, the
compiler rejects it with a clear "cannot statically resolve shape"
diagnostic. It never guesses.

## 6.9 Typed Computation Graph

This is the central compiler IR (ARCH-001). A node holds:

``` text
NodeId
OpKind
Input ValueIds
Output ValueId
Verified tensor type
SourceSpan
Semantic facts (e.g. a "fused_linear_relu_*" tag the fusion pass leaves
                for MlirLowerer to consume, rather than re-deriving the
                pattern at lowering time)
```

The graph tracks each value's producer and consumers (`consumersOf`).

## 6.10 Graph Pass Interface (Level-1 optimization)

`include/tensora/optimization/GraphPass.h` and
`src/optimization/GraphPasses.cpp` define `GraphPass`, `PassManager`,
`PassContext`, and `PassResult`. A pass takes a verified `TypedGraph`,
changes it in place, and reports whether it changed anything. The
project implements and tests all six frozen v1 passes (RM 3.21): shape
folding, dead-op elimination, reshape cancellation, transpose
cancellation, elementwise fusion, and linear-bias-activation fusion.
(The team built a constant-folding pass, then retired it — see
DECISIONS.md. Its scalar-chain reassociation cannot guarantee that
`return`'s output stays the same under IEEE-754 f32 non-associativity,
and RM 3.21 requires that guarantee.)

`PassManager::runAll` runs the whole pass list to a fixed point, not
just once. It repeats the full ordered sweep as long as the last sweep
changed the graph, up to a defensive cap (see DECISIONS.md ARCH-015).
This is what lets one pass's result open up a new opportunity for an
earlier pass in the list. For example: dead-op elimination can drop a
`linear` node's second consumer, leaving only one consumer, which is
exactly what linear-bias-activation fusion needs but did not see in the
sweep before. Every pass here only removes, merges, or tags nodes, and
never re-tags a node it already tagged, so a sweep with no change means
the whole pass list has actually reached a real fixed point.

`include/tensora/optimization/Liveness.h` and
`src/optimization/Liveness.cpp` implement the one required data-flow
analysis: **tensor liveness**. It builds a live interval for each value
and a peak-live-byte estimate from verified tensor byte sizes. This
estimate is not just a report number. The Level-2 tiling decision
(Section 6.12) uses it to pick a smaller, safer tile size under memory
pressure (ARCH-009).

## 6.11 Lowering Interface

The backend takes a valid `TypedProgram` and produces an
`mlir::ModuleOp`. The semantic analyzer never depends on MLIR classes.
Only `codegen/` includes MLIR/LLVM headers.

## 6.12 MLIR Lowering (`MlirLowerer`) and the Level-2 Tiling Pass

`include/tensora/codegen/DialectSetup.h` and
`src/codegen/DialectSetup.cpp` build one shared `mlir::DialectRegistry`.
Every lowering test and tool uses it. It holds `func`, `arith`,
`linalg`, `tensor`, `scf`, `math`, and `LLVM`, plus every
bufferizable-op-interface registration those dialects need (including
`scf`'s — easy to forget, and a real bug this phase; see Section
6.12.3) and `linalg`'s `TilingInterface` external model.

`include/tensora/codegen/MlirLowerer.h` and
`src/codegen/MlirLowerer.cpp` lower each `TypedGraph` to a `func.func`.
Lowering **erases axis names**. Each axis size maps to a tensor
dimension: a concrete size becomes a static dimension, and a symbolic
size becomes a dynamic `?` dimension (ARCH-005). The table below lists
each node's lowering. Real JIT execution or native execution checks
every one of these numerically — the team does not rely only on
structural golden-IR comparison:

| OpKind | MLIR | Notes |
|---|---|---|
| `Add/Sub/Mul/Div` | `linalg.generic` over `arith` | scalar and broadcasting tensor-tensor; replays sema's already-verified broadcast decision rather than re-deriving it |
| `Relu` | `linalg.generic` (`arith.maximumf` with 0) | |
| `MatMul` | rank 2: `linalg.matmul`; rank >= 2 with batch axes (including broadcasting): `linalg.generic` contraction | rank-2 path uses a zero-filled accumulator (see 6.12.1); this is the only path `MatmulTilingPass` tiles |
| `Linear` | rank 2: `linalg.matmul` + bias-add `linalg.generic`; rank 1 or rank >= 3: `linalg.generic` batched vector-matrix contraction | fuses an immediately-following `Relu` into the same generic body when the Level-1 fusion pass tagged the pair; only the rank-2 path is tiled |
| `Softmax` | 4 chained `linalg.generic` ops | numerically-stable max-shift, `math.exp`, sum, normalize; keep-dims reduction (RM 3.11: result has the same rank as the input) |
| `Transpose` | `linalg.transpose` | |
| `Flatten` | `tensor.collapse_shape` | reassociation `[[0],[1..rank-1]]` (RM 3.13: keeps axis 0, collapses the rest) |
| `Conv2D` | `linalg.conv_2d_nchw_fchw` | zero-filled accumulator; `tensor.pad` inserted only when padding>0; bias broadcasts over the channel axis (index 1), not the last axis like `Linear`'s bias |
| `Sum`/`Mean` | rank-reducing `linalg.generic` reduction | drops the named axis entirely (RM 3.15), unlike softmax's keep-dims; `Mean` adds a divide-by-axis-size pass |
| `MaxPool` | `linalg.pooling_nchw_max` | -inf-filled accumulator (max-identity), shape-only pooling-window operand |

General `reshape` (arbitrary regrouping beyond flatten's fixed
collapse) is not yet lowered. Batched matmul is now lowered (see the
table above), so multi-head attention is possible; the project has not
yet written a multi-head attention example, but nothing blocks it. The
lowering switch's default case is `llvm_unreachable` — this is a
documented scope boundary, not a silent fallthrough.

### 6.12.1 Accumulate into `outs`; do not initialize

`linalg.matmul`, `linalg.conv_2d_nchw_fchw`, and `linalg.pooling_nchw_max`
all **accumulate** into their `outs` operand instead of initializing
it. An uninitialized `tensor.empty()` accumulator causes a real
intermittent bug (correct on some runs, garbage on others, since it
reads whatever memory happened to be there) — every accumulating op in
this codebase explicitly zero-fills (or -inf-fills, for max) its
accumulator via `linalg.fill` first.

### 6.12.2 Level-2: project-authored shape-guided matmul tiling (the academic compiler's USP requirement)

This has two distinct pieces. Code, tests, and docs keep them separate
from each other and from standard MLIR/LLVM infrastructure:

1. **`TilingDecision`** (`include/tensora/codegen/LoopTiling.h`, pure
   C++, no MLIR dependency) is the Tensora-authored decision: whether
   and how to tile a matmul, based on its verified `M`/`K`/`N` shapes
   and the Level-1 liveness estimate. `MlirLowerer` computes a real
   `LivenessResult` for every graph it lowers (`LivenessAnalysis().run`,
   `lowerGraph`) and passes it into every `decide()` call -- this is not
   a hand-built fixture, the way the isolated `TilingDecision` unit
   tests use one; it is the same analysis Level-1 already runs, read a
   second time at the point the tiling decision actually needs it. A
   large matmul gets tiled; a tiny one does not. Under memory pressure,
   the decision picks a smaller, safer tile size. Unit tests check this
   decision on its own, with no MLIR dependency
   (`tests/codegen/test_tiling_decision.cpp`, 6 cases); `LOOP-005-A/B`
   (`tests/codegen/test_liveness_tiling_e2e.cpp`) additionally proves
   the real `MlirLowerer` path picks the same conservative tile size
   once its own liveness computation -- not a fixture -- finds the peak
   live-byte estimate over budget.
2. **`MatmulTilingPass`** (`include/tensora/codegen/MatmulTilingPass.h`
   and `src/codegen/MatmulTilingPass.cpp`) is the project-authored MLIR
   pass. It is a real `PassWrapper<MatmulTilingPass,
   OperationPass<func::FuncOp>>` that walks `linalg.matmul` ops, reads
   the tile sizes `MlirLowerer` stamped on each op from
   `TilingDecision`, and rewrites the op with MLIR's own
   `TilingInterface`/`scf::tileUsingSCF`. This is *our* pass. The
   pipeline registers and runs it like any other pass; it does not
   change MLIR itself.

Everything after this step — bufferization, `convert-linalg-to-loops`,
control-flow/arith/memref-to-LLVM conversion, `ExpandStridedMetadata`,
`LowerAffine`, `ConvertMathToLLVM` — is standard, unmodified MLIR/LLVM
infrastructure. The report and the tests
(`tests/codegen/test_tiling_e2e.cpp`) keep this three-layer split
clear: (1) Tensora's Level-1 TCG passes, (2) this one project-authored
MLIR pass, (3) everything else, which belongs to MLIR/LLVM. The tests
check for `scf.for` after the pass runs, compare tiled and untiled JIT
results numerically, and report — but do not assert — wall-clock
timing.

### 6.12.3 Two ordering bugs this phase found, both in the standard pipeline (not in team-authored code)

To make the tiled path bufferize and lower to LLVM, the team had to
find two real MLIR pipeline bugs. Neither is obvious from the
documentation:

- MLIR does not register `scf`'s bufferizable-op-interface by default.
  Without `scf::registerBufferizableOpInterfaceExternalModels`, a
  tiling-introduced nested `scf.for` with tensor `iter_args` cannot
  bufferize. It fails late with an unclear "op was not bufferized"
  error.
- `memref.subview` (introduced by tiling) does not lower to LLVM
  through `FinalizeMemRefToLLVMConversion` alone. It needs
  `ExpandStridedMetadataPass` first. That pass emits its own
  `affine.apply`/index arithmetic, and `LowerAffinePass` must lower
  this *before* `ArithToLLVMConversionPass` runs, not after.

## 6.13 LLVM and Native Execution

MLIR lowering reaches the `LLVM` dialect. The project proves two
separate execution paths:

- **JIT**, through `mlir::ExecutionEngine`. Every numeric-correctness
  test uses this path: it calls `engine->lookup(name)` and casts the
  raw result to a function pointer, instead of using the templated
  `invoke()`, which is unreliable for plain scalar returns.
- **Ahead-of-time native compilation** (`tools/codegen_mlp_native.cpp`).
  This path turns the lowered module into LLVM IR
  (`mlir::translateModuleToLLVMIR`), then calls
  `llvm::TargetMachine::addPassesToEmitFile` to write a real object
  file. CMake links this object file, along with a small dependency-free
  C driver, into a standalone executable using the system linker. `file`
  confirms this is a real ELF object, and `ldd` confirms it depends only
  on `libc`, with zero MLIR/LLVM runtime dependency. This is the "native
  executable, not an interpreter" deliverable.

## 6.14 Driver

`src/driver/main.cpp` is a general-purpose CLI driver. It takes any
`.tns` source file and runs every phase, one flag per phase's own
intermediate form:
`--emit-tokens`/`--emit-parse-tree` (lexer/parser), `--emit-ast`/
`--emit-ast-dot` (parser), `--check`/`--emit-symbols`/`--emit-graph`/
`--emit-graph-dot`/`--explain` (semantic analysis),
`--emit-optimized-graph` (optimization), `--emit-mlir` (lowering),
`--emit-llvm` (backend), plus `--run`/`-o` to run every phase together
(JIT-execute, or write a native object file). Each flag stops right
after its own phase, so the CLI itself works as a map of the pipeline
for debugging and demos. Every MLIR-lowering and codegen *test*,
though, still builds its `TypedGraph` directly with `GraphBuilder` (the
ARCH-008 mock-enabler pattern) instead of going through the driver. The
driver and the test fixtures stay independent on purpose (Section 8).

## 6.15 Testing Boundaries

Each interface has its own tests and can take synthetic inputs. This
matters for parallel development. 183 CTest cases currently pass, and
they stay stable across repeated full-suite runs.

## 6.16 Future Extensions

The architecture should allow these additions without a frontend
redesign: general `reshape` lowering; GPU lowering; layout metadata;
memory-cost metadata; PyTorch interchange; more operators; and
profiling annotations. Batched matmul lowering is done (Section 6.12);
the general-purpose `.tns`-source-to-executable CLI driver (Section
6.14) is also done.
