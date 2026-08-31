# Tensora Compiler Architecture

> System Architect document. Authoritative for module boundaries, phase
> ownership, and the frontend/semantic/codegen designs. Companion files:
> [IR.md](IR.md) (Typed Computation Graph), [INTERFACES.md](INTERFACES.md)
> (concrete cross-phase C++ interfaces), [DECISIONS.md](DECISIONS.md)
> (architecture decision records). The language it implements is frozen in
> [../language/00_WHITEPAPER.md](../language/00_WHITEPAPER.md) and
> [../language/03_LANGUAGE_REFERENCE_MANUAL.md](../language/03_LANGUAGE_REFERENCE_MANUAL.md).

## 1. Pipeline Overview

``` text
Tensora source (.tns)
   -> Lexer            (ANTLR)              -> Token stream
   -> Parser           (ANTLR)              -> AST
   -> Semantic Analyzer (attribute grammar) -> Typed SSA Computation Graph (TCG)
   -> Level-1 opt: graph passes + tensor-liveness/peak-memory analysis -> Optimized TCG
   -> MLIR Lowering                         -> mlir::ModuleOp (func/linalg/tensor/arith/scf)
   -> Level-2 opt: structured-loop tiling on matmul (project decision, MLIR rewrite)
   -> MLIR + LLVM standard pipeline         -> LLVM IR -> native CPU object
   -> Runtime link                          -> executable
```

The project has two optimization levels (Level-1 over the SSA graph,
Level-2 over structured loops) plus one data-flow analysis. MLIR and
LLVM do the mechanical lowering and the standard scalar optimization.

Every arrow above is a **stable interface** ([Section 4](#4-cross-phase-interfaces-summary)).
A downstream module depends only on the upstream module's published
interface, never on its internals.

## 2. Module Map

``` text
grammar/            Tensora.g4: one ANTLR4 grammar (lexer + parser rules)
include/tensora/
├── shared/         SourceSpan, SourceBuffer, Result<T>, Diagnostic/
│                   DiagnosticEngine, AST (Decl/Expr nodes/AstProgram),
│                   IR (Value, GraphNode, TypedGraph, GraphBuilder,
│                   GraphVerifier -- see IR.md), Type/TensorType/DimSolver:
│                   every genuinely cross-phase representation, in one place
├── lexer/          Token, TokenKind, Lexer (wraps the ANTLR-generated
│                   lexer), TokenAdapter (ANTLR token -> project Token)
├── parser/         Parser (runs the ANTLR-generated lexer+parser directly
│                   over source text; src/parser/AstBuilder.cpp walks the
│                   parse tree to build the AST -- no grammar actions)
├── semantic/       SemanticAnalyzer, operator contract checkers (attribute
│                   grammar), Symbol/SymbolTable
├── optimization/   GraphPass, PassManager, the six v1 graph passes,
│                   LivenessAnalysis (Level-1 data-flow)
├── codegen/        MlirLowerer, TilingDecision, DialectSetup, MatmulTilingPass
└── driver/         CLI, phase orchestration, --emit-* flags

src/  mirrors include/  ;  tests/ mirrors phases (see docs/testing/08_TEST_PLAN.md)
```

**No circular dependencies.** `shared/` does not know about `codegen/`.
`semantic/` never includes MLIR. `codegen/` is the only module that
touches MLIR/LLVM headers.

## 3. Phase Ownership (five people)

| Owner | Area | Deliverable boundary |
|---|---|---|
| M1 | Language + Frontend | `Source -> Token -> AST` (`lexer/`, `parser/`, `shared/` AST) |
| M2 | Type system + Architecture | `shared/` (types, IR) contracts, this document |
| M3 | Semantic analysis | `AST -> verified TCG` (`semantic/`) |
| M4 | MLIR/LLVM + integration | `TCG -> executable` (`codegen/`, `driver/`) |
| M5 | Optimization + Testing | `optimization/`, `tests/`, corpus, benchmarks |

`shared/` (support types, diagnostics, AST, IR) is a shared foundation.
It is agreed on first, and owned jointly across phases (curated under
the M2 role).

## 4. Cross-Phase Interfaces (summary)

Every arrow in the pipeline above is a real function signature, not
just a diagram label; see [INTERFACES.md](INTERFACES.md) for the full,
concrete C++ signatures behind each one:

``` text
Lexer       : Source            -> std::vector<Token>
Parser      : SourceBuffer      -> Result<AstProgram>
Semantic    : AstProgram        -> Result<TypedProgram>   (+ DiagnosticEngine)
Optimization: TypedGraph&       -> PassResult             (in place)
Codegen     : const TypedProgram& -> mlir::ModuleOp / ObjectModule
```

Two representations carry the data across every phase boundary in that
table, so every downstream phase depends on their shape: the **AST**
(pure syntax, produced by parsing) and the **Typed Computation Graph**
(verified semantics, produced by semantic analysis;
[ADR ARCH-001](DECISIONS.md#arch-001-typed-computation-graph-as-the-central-ir)).
The AST/semantic boundary and the TCG/codegen boundary are also where
mocks are injected for parallel work
([Section 8](#8-phase-parallelism-via-mocks)).

## 5. Frontend Design

### 5.1 Lexer and Parser — ANTLR4-generated ([ADR ARCH-011](DECISIONS.md#arch-011-antlr4-based-frontend))

The token set is tiny: 5 keywords (`input weight graph return Tensor`),
identifiers, integer and float literals, the operators `+ - * / @ =`, the
delimiters `: , [ ] { } ( )`, and `#` line comments. The grammar
([RM 3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure),
[3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary))
is small and LL(1):

``` text
program   ::= decl* graph+
decl      ::= ("input" | "weight") ident ":" tensorType
tensorType::= "Tensor" "[" axis ("," axis)* "]"
axis      ::= ident "=" (intLit | ident)
graph     ::= "graph" ident "{" stmt* "return" ident "}"
stmt      ::= ident "=" expr
expr      ::= call | ident (binop operand)?      // one operation per statement
call      ::= ident "(" arg ("," arg)* ")"
operand   ::= ident | floatLit
binop     ::= "@" | "+" | "-" | "*" | "/"
arg       ::= ident | ident "=" argval | "[" axis ("," axis)* "]"
argval    ::= intLit | ident | "[" intLit ("," intLit)* "]"
```

- One **ANTLR4** grammar (`grammar/Tensora.g4`) defines both the
  lexical and syntactic grammar. ANTLR generates `TensoraLexer` and
  `TensoraParser` as separate C++ classes. So the lexer stays
  independently runnable and demonstrable (`Lexer`, `--emit-tokens`,
  the `LEX-*` tests), even though one grammar file defines both.
- The grammar has **no embedded C++ actions**. `Parser`
  ([`src/parser/Parser.cpp`](../../src/parser/Parser.cpp)) runs the
  ANTLR lexer and parser directly on source text. Then `AstBuilder`
  ([`src/parser/AstBuilder.cpp`](../../src/parser/AstBuilder.cpp))
  walks the resulting parse tree and builds the AST
  ([INTERFACES §2](INTERFACES.md#2-parser)) as plain recursive descent
  over context pointers -- a hand-written function per grammar rule
  that reads the parse tree and constructs the matching AST node,
  calling itself for nested sub-rules. It does not use the
  ANTLR-generated Visitor base class instead, because that base class
  returns each visit as a `std::any` (a type-erased "any value"
  container copied by value), and the AST nodes it would need to
  return are `std::unique_ptr`-owned (move-only, not copyable) -- the
  two are incompatible.
- The parser builds AST nodes only. It does **no** shape, type, or axis
  reasoning (every rule in
  [RM Section 3](../language/03_LANGUAGE_REFERENCE_MANUAL.md) is a
  semantic rule). It parses argument lists permissively into a general
  `Arg` node; semantic analysis
  ([Section 6](#6-semantic-analyzer-design-the-compilers-heaviest-weighted-phase))
  checks each operator's argument count and kind.
- The parser skips whitespace and `#...` comments. It sends illegal
  characters to ANTLR's hidden channel, so the parser never sees them.
  This stops one bad character from also causing a spurious "unexpected
  token" error. The parser reports each illegal character once, as a
  `LexIllegalChar` diagnostic
  ([`include/tensora/lexer/TokenAdapter.h`](../../include/tensora/lexer/TokenAdapter.h)),
  shared by `Lexer` and `Parser`.
- Error recovery uses `ParserErrorStrategy`
  ([`src/parser/ParserErrorStrategy.h`](../../src/parser/ParserErrorStrategy.h)),
  which turns off ANTLR's single-token repair (insert/delete one token
  to hide an error) so every syntax error falls back to panic-mode
  recovery: it resynchronizes at the enclosing rule's follow set. So
  one file can still report several independent syntax errors in one
  pass.

## 6. Semantic Analyzer Design (the compiler's heaviest-weighted phase)

`SemanticAnalyzer::analyze(const AstProgram&, TypedProgram& out)` does
this:

1. **Global declarations.** It registers every `input`/`weight` in the
   symbol table with its `TensorType`, and interns each symbolic
   dimension name as a `SymbolId`. Redeclaring a global name is an
   error ([RM 3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules)).
2. **Per graph.** It enters a graph scope, then walks statements in
   source order:
   - It resolves the RHS expression to a `Value` with an inferred
     `TensorType`, by calling the operator's contract checker
     ([Section 6.2](#62-tensora-specific-checks-the-differentiators)).
     It binds the LHS name to that `Value` (a rebind shadows the old
     value, [RM 3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules)).
     It emits the matching TCG node through `GraphBuilder`.
   - On any error, it emits a `Diagnostic`, gives the value
     `InvalidType`, and **keeps going** ("poisoning"). An `InvalidType`
     operand stops cascading errors, but lets the analyzer still find
     other, independent errors
     ([ADR ARCH-007](DECISIONS.md#arch-007-error-recovery-by-type-poisoning)).
3. **Return.** It resolves the `return` name and sets the graph's
   output value.
4. **Finalize.** It runs `GraphVerifier` ([IR.md Section 6](IR.md#6-construction-and-verification))
   on each graph and attaches the program to `out`.

### 6.1 Required academic-compiler checks

- redeclaration (global names; [RM 3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules));
- undefined identifier ([RM 3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules));
- scope (global vs. graph-local bindings; [RM 3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules));
- type mismatch (using a scalar where a tensor is required, or the
  reverse; [RM 3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting));
- operator argument count and kind ("function argument type" — each
  contract checker checks argument count and argument kind,
  [RM 3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication)–[3.17](../language/03_LANGUAGE_REFERENCE_MANUAL.md#317-maxpool)).

### 6.2 Tensora-specific checks (the differentiators)

The project implements one contract checker per operator. Each checker
returns a `Result<TensorType>`:

- **matmul / linear:** checks the positional contraction by unifying
  sizes; broadcasts leading axes; names the result axis
  ([RM 3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication),
  [3.9](../language/03_LANGUAGE_REFERENCE_MANUAL.md#39-linear)).
- **elementwise + broadcasting with axis-name compatibility:** this
  check catches `[batch,feat] + [feat,batch]` even when the sizes
  match ([RM 3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting)).
  This is the project's main semantic feature.
- **reshape / flatten:** checks that element counts stay equal, within
  the bounded dimension subset
  ([RM 3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape),
  [3.13](../language/03_LANGUAGE_REFERENCE_MANUAL.md#313-flatten)).
- **transpose:** checks that the permutation is valid; names and sizes
  move together ([RM 3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose)).
- **sum / mean / softmax:** checks that the named axis exists
  ([RM 3.11](../language/03_LANGUAGE_REFERENCE_MANUAL.md#311-softmax),
  [3.15](../language/03_LANGUAGE_REFERENCE_MANUAL.md#315-sum-mean)).
- **conv2d / maxpool:** unifies channel counts and checks exact spatial
  arithmetic ([RM 3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d),
  [3.17](../language/03_LANGUAGE_REFERENCE_MANUAL.md#317-maxpool)).

These checks form an **attribute grammar**. Each node builds its tensor
type from its operands' types (for `A @ B`, the checker confirms
`A.shape[-1] == B.shape[-2]`, then builds the result shape). This is
the standard attribute-grammar semantic-analysis method, applied to tensors.

### 6.3 Dimension reasoning

A `DimSolver` (union-find over `SymbolId`, plus integer arithmetic)
implements the bounded subset of
[RM 3.4](../language/03_LANGUAGE_REFERENCE_MANUAL.md#34-symbolic-dimensions):
it checks equality and unification, and it compares element counts by
matching the multiset of symbolic factors and comparing constant
products. It has no SMT (Satisfiability Modulo Theories -- a general
class of automated solvers for logical formulas over integers, arrays,
etc., much more powerful, and much more expensive, than the pattern it
replaces here) and no general nonlinear solving
([ADR ARCH-006](DECISIONS.md#arch-006-bounded-dimension-solver-no-smt)).
If a program needs more than this, the compiler rejects it with a
clear "cannot statically resolve shape" diagnostic.

### 6.4 Explain mode

`explain` prints the facts already stored on TCG values: each value's
inferred type and the contract that produced it (whitepaper
[Section 5](../language/00_WHITEPAPER.md#5-tensora-is-explainable)). It
only prints the TCG; it does not run a second analysis.

## 6a. Optimization and Analysis (two levels)

- **Level-1 (SSA graph):** the six graph passes the project itself
  designed and wrote (`optimization/`,
  [RM 3.21](../language/03_LANGUAGE_REFERENCE_MANUAL.md#321-optimization-semantics)),
  plus one real data-flow analysis — **tensor liveness** (`optimization/`,
  [INTERFACES §4a](INTERFACES.md#4a-analysis-and-loop-transformation)).
  Liveness computes each value's live interval, and, using verified
  shapes for byte sizes, a **peak intermediate-memory estimate**. A
  transformation or a report uses this estimate. This is
  analysis-driven optimization, not just pattern rewriting.
  `PassManager::runAll` repeats the whole ordered pass list to a fixed
  point (up to a defensive cap), not just once, since one pass can
  create the exact condition an earlier pass in the list was looking
  for -- for example, dead-op elimination dropping a `linear` node's
  unused second consumer is exactly what lets linear-bias-activation
  fusion tag that pair, but only on the *next* sweep
  ([ADR ARCH-015](DECISIONS.md#arch-015-passmanagerrunall-runs-the-pass-list-to-a-fixed-point)).
  A seventh pass, constant folding, was built and then retired: its
  scalar-chain reassociation could not guarantee `return`'s output
  stays identical under IEEE-754 f32 non-associativity, which
  [RM 3.21](../language/03_LANGUAGE_REFERENCE_MANUAL.md#321-optimization-semantics)
  requires
  ([ADR ARCH-014](DECISIONS.md#arch-014-semabackend-rank-gap-closed-unsafe-cast-retired)).
- **Level-2 (structured loops):** after matmul is lowered to `linalg`
  and `scf` loops, `TilingDecision`
  ([INTERFACES §4a](INTERFACES.md#4a-analysis-and-loop-transformation))
  picks tile sizes from the verified `M/N/K` and asks for tiling. MLIR
  performs the rewrite. The decision is this project's own contribution;
  MLIR provides the mechanics
  ([ARCH-009](DECISIONS.md#arch-009-two-optimization-levels-one-data-flow-analysis)).

Matmul is the reference computation carried through both levels
(whitepaper [§8.3](../language/00_WHITEPAPER.md#83-matrix-multiplication-the-reference-computation)–[§8.4](../language/00_WHITEPAPER.md#84-structured-loop-tiling-level-2-optimization)).

## 7. Code Generation Design

Optimization happens **before** and **during** lowering: Level-1 over
the TCG (`optimization/`), and the Level-2 tiling decision at the loop
level. Code generation itself is lowering plus a call into standard
infrastructure. The two stay distinct, so the report can separate the
project's own optimization work from reused MLIR/LLVM infrastructure.

### 7.1 TCG -> MLIR (`MlirLowerer`)

- `DialectSetup` (`include/tensora/codegen/DialectSetup.h`,
  `src/codegen/DialectSetup.cpp`) builds one shared
  `mlir::DialectRegistry` -- `func`, `arith`, `linalg`, `tensor`,
  `scf`, `math`, `LLVM` -- that every lowering test and tool reuses,
  plus every bufferizable-op-interface registration those dialects
  need. That last part is easy to get wrong: MLIR does not register
  `scf`'s bufferizable-op-interface by default, so without
  `scf::registerBufferizableOpInterfaceExternalModels` a
  tiling-introduced nested `scf.for` with tensor `iter_args` cannot
  bufferize, and fails late with an unclear "op was not bufferized"
  error rather than a clear one at the point the interface is missing.
- Each `TypedGraph` becomes a `func.func`. Lowering **erases axis
  names**; each axis size maps to a tensor dimension: a concrete size
  becomes a static dimension, a symbolic size becomes a dynamic (`?`)
  dimension ([ADR ARCH-005](DECISIONS.md#arch-005-axis-names-live-in-the-surface-language-and-tcg-erased-at-the-mlir-boundary)).
- Per-node lowering (whitepaper [7.2](../language/00_WHITEPAPER.md#72-lowering-to-mlir)),
  as `MlirLowerer` actually implements it (11 of 13 `OpKind`s; the
  switch's default case is `llvm_unreachable` for the other 2, not a
  silent fallthrough):
  `MatMul` -> `linalg.matmul` for rank 2, or a `linalg.generic`
  contraction for rank >= 2 with batch axes (including broadcasting);
  `Linear` -> `linalg.matmul` (rank 2) or a `linalg.generic` batched
  vector-matrix contraction (rank 1 or rank >= 3), plus a
  `linalg.generic` bias-add, fused with an immediately-following `Relu`
  when `LinearBiasActivationFusion` tagged the pair; `Conv2D` ->
  `linalg.conv_2d_nchw_fchw` (zero-filled accumulator, `tensor.pad`
  when `padding>0`, channel-broadcast bias-add through a
  `linalg.generic`); `Add/Sub/Mul/Div/Relu` -> `linalg.generic` bodies
  over `arith`; `Softmax` -> a numerically-stable 4-step `linalg.generic`
  chain (reduce-max, subtract and `math.exp`, reduce-sum, divide);
  `Transpose` -> `linalg.transpose`; `Flatten` -> `tensor.collapse_shape`
  (fixed reassociation `[[0],[1..rank-1]]`, since
  [RM 3.13](../language/03_LANGUAGE_REFERENCE_MANUAL.md#313-flatten)
  always keeps axis 0 and collapses the rest); `Sum`/`Mean` -> a
  hand-built rank-reducing `linalg.generic` (not the dedicated
  `linalg.reduce` op); `Mean` adds a divide-by-axis-size pass;
  `MaxPool` -> `linalg.pooling_nchw_max` (`-inf`-filled accumulator,
  shape-only window operand).
  **Not implemented:** general `Reshape` (arbitrary regrouping beyond
  `Flatten`'s fixed collapse) -- `Flatten`'s reassociation is a single
  fixed pattern (keep axis 0, collapse the rest), decidable at lowering
  time with no extra bookkeeping; a general `Reshape` would need a
  fresh reassociation computed per call site from two arbitrary shapes,
  which is more lowering machinery than v1 needed for any example
  program, so it was left out rather than half-implemented. Similarly,
  only the rank-2 matmul and rank-2 linear paths get the Level-2 tiling
  pass below, because `TilingDecision` and `MatmulTilingPass` were
  designed and verified against the rank-2 case
  ([Section 6a](#6a-optimization-and-analysis-two-levels)); the
  other ranks still lower correctly, they simply keep whatever loop
  structure `convert-linalg-to-loops` produces by default, untiled.
- Every op that builds a fresh accumulator tensor needs the same fix:
  `linalg.matmul`/`conv_2d_nchw_fchw` *accumulate* into `outs`, so they
  need a zero-fill through `linalg.fill` first, not an uninitialized
  `tensor.empty()`. `linalg.pooling_nchw_max` needs a `-inf`-fill in the
  same way, matching its max-reduction identity.
- **Known limitation (not yet fixed):** `lowerTensorType` correctly
  maps a symbolic axis size to a dynamic `?` *type*
  ([ARCH-005](DECISIONS.md#arch-005-axis-names-live-in-the-surface-language-and-tcg-erased-at-the-mlir-boundary)),
  but the *value*-level lowering does not. Every `tensor.empty()` call
  is built with zero dynamic-size operands, so any graph with a
  symbolic dimension fails MLIR verification once it reaches
  `MlirLowerer`. The CLI driver
  ([`src/driver/main.cpp`](../../src/driver/main.cpp)) checks for this
  and rejects a symbolic-shaped program before lowering, with a clear
  error, instead of lowering it and crashing. Fixing it properly is not
  a small patch: it means threading a same-shape existing SSA value
  (to query the missing dynamic sizes from via `tensor.dim`) through
  every lowering function that materializes a tensor, a real, separate
  piece of work of its own. This is tracked as
  LOW-005 in the [Lowering table](../testing/TEST_MATRIX.md#lowering-rm-322)
  of TEST_MATRIX.md.
- **Level-2 loop optimization:** after `MlirLowerer` produces a
  rank-2 `linalg.matmul`, `MatmulTilingPass`
  ([`src/codegen/MatmulTilingPass.cpp`](../../src/codegen/MatmulTilingPass.cpp),
  a real `PassWrapper<..., OperationPass<func::FuncOp>>`) reads the
  tile sizes `MlirLowerer` stamped on the op from `TilingDecision`
  ([`include/tensora/codegen/LoopTiling.h`](../../include/tensora/codegen/LoopTiling.h),
  [INTERFACES §4a](INTERFACES.md#4a-analysis-and-loop-transformation)),
  and applies them through MLIR's own
  `TilingInterface`/`scf::tileUsingSCF`. Only then does
  `convert-linalg-to-loops` turn the (now possibly tiled)
  `linalg.matmul` into `scf` loops. This project's contribution is the
  decision (`TilingDecision`) and the pass that applies it
  (`MatmulTilingPass`); `TilingInterface`'s mechanical rewrite is
  infrastructure
  ([ARCH-009](DECISIONS.md#arch-009-two-optimization-levels-one-data-flow-analysis)).
  This was verified end to end against a 256x256x256 matmul
  ([`examples/matmul_large.tns`](../../examples/matmul_large.tns),
  above the pass's default tiling threshold); `--emit-mlir` shows the
  resulting nested `scf.for` loops.
- Scalar operands (for example `* 0.125`) are node attributes, emitted
  as `arith` constants inside the elementwise body.

### 7.2 MLIR -> native

Both [`src/driver/main.cpp`](../../src/driver/main.cpp) and every
lowering test wire up the same standard MLIR pipeline: one-shot
bufferization (`tensor` -> `memref`,
`bufferizeFunctionBoundaries=true`, `functionBoundaryTypeConversion=
FullyDynamicLayoutMap`). One-shot bufferization only allocates; it does
not free, so right after it the pipeline runs MLIR's ownership-based
buffer deallocation (`createOwnershipBasedBufferDeallocationPass` +
`createBufferDeallocationSimplificationPass` +
`createLowerDeallocationsPass`, plus
`createConvertBufferizationToMemRefPass` for the runtime alias check a
returned buffer needs), so every intermediate tensor's buffer is freed
and only the one actually returned survives -- checked, for every
example workload, by `malloc`/`free` call counts in the emitted LLVM IR
(`free` count = `malloc` count - 1). Then `convert-linalg-to-loops`
(`scf`),
`scf`-to-control-flow, `memref.subview`'s `ExpandStridedMetadataPass`
(needed once tiling adds subviews; it must run *before*
`LowerAffinePass`, since it emits `affine.apply`, which must in turn
run before `ArithToLLVM` converts `index` to `i64`), `ConvertMathToLLVM`
(for softmax's `math.exp`), `arith`/`memref`/`func`-to-LLVM,
`ReconcileUnrealizedCasts`, then `mlir::translateModuleToLLVMIR` to
LLVM IR. From there, two separate execution paths exist: JIT through
`mlir::ExecutionEngine` (`--run`; the driver calls each function's
`_mlir_ciface_<name>` wrapper directly as a raw function pointer, and
picks the right call by argument count through a switch, since
`ExecutionEngine::invokePacked` needs that exact prefix from the
caller, and its own dispatch was unreliable in practice); and
ahead-of-time native object-file output through `llvm::TargetMachine`
(`-o`; `file` confirms a real ELF relocatable object, and `ldd` on the
linked executable confirms zero MLIR/LLVM runtime dependency, only
libc). `--run` supplies synthetic all-ones data for every declared
input/weight (Tensora source declares shapes, not values — there is no
tensor-literal syntax). There is no separate C runtime `main` template
and no "print all outputs" tool. The driver does not pass symbolic
dimensions as runtime sizes; see the
[LOW-005 limitation above](../testing/TEST_MATRIX.md#lowering-rm-322),
which blocks any symbolic-shaped program from reaching this stage at all.

### 7.3 CLI driver (`src/driver/main.cpp`)

The actual implementation has no separate `Backend`/`CodegenOptions`/
`ObjectModule` abstraction layer. `main.cpp` is a direct, procedural
driver, with one flag per compiler phase, matching the CLI surface
[docs/language/02_LANGUAGE_TUTORIAL.md §2.12](../language/02_LANGUAGE_TUTORIAL.md#212-running-programs-yourself)
documents: `--emit-tokens`
(lexer only), `--emit-parse-tree`/`--emit-ast`/`--emit-ast-dot` (+
parser), `--check`/`--emit-symbols`/`--emit-graph`/`--emit-graph-dot`/
`--explain` (+ semantic analysis; these also work on symbolic shapes,
since they stop before lowering), `--emit-optimized-graph` (+ the
Level-1 passes only, stopping before MLIR exists), `--emit-mlir` (+
`MlirLowerer`, `MatmulTilingPass`), `--emit-llvm` (+ translation to
LLVM IR text), `--run`/`-o` (+ the full [§7.2](#72-mlir---native)
pipeline, JIT or ahead-of-time). `runOptimizationPasses()` and
`lowerToModule()` are two separate functions so that
`--emit-optimized-graph` and `--emit-mlir` each show only their own
phase's work, instead of one flag hiding both.
`--emit-mlir`/`--emit-llvm`/`--run`/`-o` all require concrete shapes
throughout ([LOW-005](../testing/TEST_MATRIX.md#lowering-rm-322)). There is no custom MLIR dialect in v1
([ADR ARCH-004](DECISIONS.md#arch-004-no-custom-mlir-dialect-in-v1)).

## 8. Phase Parallelism via Mocks

- **Sema (M3)** builds test `AstProgram`s by hand. It does not need the
  parser to be finished first.
- **Lowering (M4)** builds `TypedProgram` fixtures with `GraphBuilder`.
  It does not need sema to be finished first.
- **Optimization (M5)** builds synthetic `TypedGraph`s and checks the
  before/after structure.
- **Tester (M5)** runs `--check` semantic tests with no backend at all.

`GraphBuilder` and `GraphVerifier` ([IR.md](IR.md);
[ADR ARCH-008](DECISIONS.md#arch-008-graphbuilder--graphverifier-as-the-mock-enablers))
make this possible: any downstream owner can build valid IR without the
upstream phase.

## 9. Data Ownership

- `SourceManager` owns the source buffers for the whole run.
  `SourceSpan` is a lightweight handle into it.
- The `AstProgram` owns all AST nodes (an arena, or a `unique_ptr`
  tree).
- The `TypedProgram` owns all `Type`, `Value`, and `GraphNode` objects
  (arenas keyed by id). `ValueId`/`NodeId` are stable indices, never
  raw pointers, across phases.
- Only `codegen/` owns `mlir::ModuleOp` and LLVM objects.
- There is no shared mutable global state. The code passes the
  `DiagnosticEngine` explicitly everywhere.

## 10. Final Checklist (from the architect playbook)

- [x] Each module has one responsibility.
- [x] No circular dependencies.
- [x] AST independent of MLIR.
- [x] Sema independent of LLVM.
- [x] TCG is the semantic/codegen boundary.
- [x] Passes have an explicit interface.
- [x] The code-generation target is isolated to one module (`codegen/`),
      so it could be replaced without changing any earlier phase --
      even though v1 has no separate `Backend` interface class of its
      own ([Section 7.3](#73-cli-driver-srcdrivermaincpp)); the module
      boundary already gives this isolation.
- [x] Every phase/role can work in parallel via mocks.
