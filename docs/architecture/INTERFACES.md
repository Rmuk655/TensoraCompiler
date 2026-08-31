# Tensora Cross-Phase Interfaces

> Concrete C++ interface signatures at each compiler boundary. These are
> the contracts that let the frontend, type system, semantic analysis,
> optimization, and backend work develop in parallel, each against a
> stable boundary. Once frozen, they stay stable. Companion file:
> [06_COMPILER_ARCHITECTURE.md](06_COMPILER_ARCHITECTURE.md). Where a
> snippet below is short enough to stay in sync by hand, it is quoted
> directly; longer or more detailed listings link straight to the real
> header instead of duplicating it, since the header is what actually
> compiles.

## 0. Shared Foundations

Real header:
[include/tensora/shared/SourceSpan.h](../../include/tensora/shared/SourceSpan.h),
[include/tensora/shared/Result.h](../../include/tensora/shared/Result.h),
[include/tensora/shared/Diagnostic.h](../../include/tensora/shared/Diagnostic.h).

``` cpp
// shared/SourceSpan.h
struct SourceSpan {
    uint32_t fileId;
    uint32_t beginOffset, endOffset;    // byte offsets into the source buffer
    uint32_t line, col;                 // 1-based, of beginOffset
};

// shared/Result.h  — lightweight expected<T>
template <class T> class Result {
public:
    bool ok() const;
    T&        value();
    const DiagnosticList& errors() const;
};

// shared/Diagnostic.h
enum class Severity { Error, Warning, Note };
enum class DiagCode {                    // stable codes -> tested by category, not prose
    LexIllegalChar, LexIntegerOverflow, ParseUnexpectedToken, ParseExpected,
    SemaUndefinedName, SemaRedeclaration, SemaArgCount, SemaArgKind,
    SemaTypeMismatch, SemaRankMismatch, SemaShapeMismatch, SemaAxisNameMismatch,
    SemaBroadcast, SemaReshapeCount, SemaTransposePerm, SemaAxisNotFound,
    SemaConvChannel, SemaConvArithmetic, SemaUnresolvableShape,
    SemaNonPositiveDimension,  // RM 3.3: an axis size must be positive or symbolic
    SemaDuplicateAxisName,     // RM 3.3: axis names unique within one tensor type (ARCH-013)
    IrInvariantViolation       // GraphVerifier (Section 3a.6) catches a compiler-internal defect
};
struct Diagnostic {
    Severity    severity;
    DiagCode    code;
    SourceSpan  span;
    std::string message;      // human text, e.g. "784 != 512"
};
using DiagnosticList = std::vector<Diagnostic>;

class DiagnosticEngine {
public:
    void report(const Diagnostic&);
    bool hasErrors() const;
    const DiagnosticList& all() const;
    std::string render(const SourceBuffer&) const;   // one "file:line:col: error: msg" line each
};
```

Every phase takes a `DiagnosticEngine&` and reports into it. No phase
prints directly. `Diagnostic` has no notes/sub-diagnostics field — each
diagnostic is one flat `message` string; there is no `SourceManager`
type anywhere in the codebase, `render()` takes a `SourceBuffer`.

## 1. Lexer

``` cpp
// lexer/Token.h
enum class TokenKind {
    KwInput, KwWeight, KwGraph, KwReturn, KwTensor,
    Identifier, IntLiteral, FloatLiteral,
    Plus, Minus, Star, Slash, At, Equal,
    Colon, Comma, LBracket, RBracket, LBrace, RBrace, LParen, RParen,
    EndOfFile, Error
};
struct Token {
    TokenKind        kind;
    std::string_view lexeme;             // view into the source buffer
    SourceSpan       span;
    int64_t          intValue   = 0;     // valid iff kind == IntLiteral
    double           floatValue = 0.0;   // valid iff kind == FloatLiteral
};

// lexer/Lexer.h  — thin C++ wrapper over the ANTLR-generated scanner
class Lexer {
public:
    Lexer(const SourceBuffer&, DiagnosticEngine&);
    std::vector<Token> tokenize();       // always ends with EndOfFile
};
```

**ANTLR4** generates the scanner (`grammar/Tensora.g4`). This class
adapts the scanner's output to the `Token`/`SourceSpan` interface,
through the shared `lexer/TokenAdapter.h` conversion. Contract: it
skips whitespace and `#` line comments. It does not treat newlines
specially. For an illegal character, it emits an `Error` token and a
`LexIllegalChar` diagnostic. For an `intLiteral` too wide for a signed
64-bit integer, it emits an `Error` token and a `LexIntegerOverflow`
diagnostic (LANG-001). In both cases, it keeps scanning. Its output is
deterministic, for golden tests.

## 2. Parser

``` cpp
// parser/Parser.h — runs the ANTLR-generated lexer+parser (grammar/
// Tensora.g4) directly over source text
class Parser {
public:
    Parser(const SourceBuffer&, DiagnosticEngine&);
    Result<AstProgram> parseProgram();   // ANTLR parser; AstBuilder walks the parse tree to build the AST
};
```

There is no project `Token` vector between the ANTLR lexer and parser.
`Parser` lexes and parses source text in one call, with no grammar
actions. `AstBuilder` (`src/parser/AstBuilder.cpp`) builds the AST
nodes below by walking the parse tree in project-owned C++.

AST nodes (`ast/`), syntax only — no types or shapes:

``` cpp
struct TensorTypeSyntax { std::vector<AxisSyntax> axes; SourceSpan span; };
struct AxisSyntax       { std::string name; DimSyntax size; SourceSpan span; };
struct DimSyntax        { enum class K { Const, Symbol } k; int64_t constant; std::string symbol; };

struct Decl { enum class Kind { Input, Weight } kind;
              std::string name; TensorTypeSyntax type; SourceSpan span; };

struct Expr { SourceSpan span; virtual ~Expr() = default; };
struct NameExpr   : Expr { std::string name; };
struct ScalarExpr : Expr { double value; };
struct MatMulExpr : Expr { ExprPtr lhs, rhs; };
struct BinaryExpr : Expr { enum class Op { Add, Sub, Mul, Div } op; ExprPtr lhs, rhs; };
struct CallExpr   : Expr { std::string callee; std::vector<Arg> args; };

struct Arg {                             // parsed permissively; sema validates per operator
    std::optional<std::string> keyword;  // e.g. "stride", "axis", "perm", "size", or none
    enum class Kind { Name, Int, Float, IntList, AxisList } kind;
    std::string           name;          // Name / axis reference
    int64_t               intVal;
    double                floatVal;
    std::vector<int64_t>  intList;       // perm=[...]
    std::vector<AxisSyntax> axisList;    // reshape target [batch=B, head=12, dim=64]
    SourceSpan            span;
};

struct GraphDecl { std::string name; std::vector<Stmt> stmts;
                   std::string returnName; SourceSpan span; };
struct Stmt      { std::string name; ExprPtr rhs; SourceSpan span; };
struct AstProgram { std::vector<Decl> decls; std::vector<GraphDecl> graphs; };
```

Contract: this module builds a complete AST, or reports syntax errors
and recovers at the next statement or top-level keyword. It never
looks at shapes.

## 3. Semantic Analyzer

``` cpp
// semantic/Symbol.h — the environment name resolution actually uses (RM 3.6)
struct Symbol { enum class Kind { Input, Weight, Value };
                std::string name; Kind kind; const Type* type; SourceSpan span; };

class SymbolTable {
public:
    bool declareGlobal(const Symbol&);               // false = rejected duplicate, not inserted
    const Symbol* lookupGlobal(std::string_view) const;
    void enterGraph();                                // resets the local scope
    enum class BindOutcome { Ok, KindConflict };
    BindOutcome bindLocal(const std::string&, Symbol); // KindConflict = local would shadow a global
    const Symbol* resolve(std::string_view) const;     // local, then global; nullptr = unresolved
    const std::vector<const Symbol*>& orderedGlobals() const;  // declaration order
    const std::vector<const Symbol*>& orderedLocals() const;   // current graph, declaration order
};

// semantic/SemanticAnalyzer.h
class SemanticAnalyzer {
public:
    SemanticAnalyzer(TypeContext&, DiagnosticEngine&);
    Result<TypedProgram> analyze(const AstProgram&);
    const SymbolTableReport& symbolTable() const;  // valid after analyze(); --emit-symbols' source
};
```

`SymbolTableReport` (`SemanticAnalyzer.h`) is a declaration-order
snapshot of what `SymbolTable` actually accepted while `analyze()` ran —
global entries, plus one scope per graph with its final local bindings
and a resolution log (referenced name -> `global::name`/`local::name`/
`undefined`) — captured from the real `SymbolTable` instance, not a
parallel reconstruction. `tensora --emit-symbols` renders it (main.cpp);
see whitepaper §4.1.

Internal operator-contract interface (one per operator; the unit of M3's
work and of the semantic test suite):

``` cpp
// semantic/Contracts.h   — each returns the result TensorType or reports and returns invalid()
struct ContractCtx { TypeContext& types; DimSolver& solver; DiagnosticEngine& diags; SourceSpan span; };

const Type* checkMatMul   (const Type* a, const Type* b, ContractCtx&);
const Type* checkLinear   (const Type* x, const Type* w, const Type* b, ContractCtx&);
const Type* checkElementwise(BinaryExpr::Op, const Type* a, const Type* bOrScalar, ContractCtx&);
const Type* checkRelu     (const Type* x, ContractCtx&);
const Type* checkSoftmax  (const Type* x, std::string_view axis, ContractCtx&);
const Type* checkReshape  (const Type* x, const std::vector<Axis>& target, ContractCtx&);
const Type* checkFlatten  (const Type* x, ContractCtx&);
const Type* checkTranspose(const Type* x, const std::vector<uint32_t>& perm, ContractCtx&);
const Type* checkReduce   (OpKind sumOrMean, const Type* x, std::string_view axis, ContractCtx&);
const Type* checkConv2D   (const Type* x, const Type* w, const Type* b, int64_t stride, int64_t pad, ContractCtx&);
const Type* checkMaxPool  (const Type* x, int64_t size, int64_t stride, ContractCtx&);
```

``` cpp
// shared/DimSolver.h (full header: ../../include/tensora/shared/DimSolver.h)
struct ElementCountResult {
    enum class Status { Equal, Mismatch, Unresolvable } status;
    int64_t lhsConstProduct = 1, rhsConstProduct = 1;
};

class DimSolver {
public:
    bool unify(Dimension a, Dimension b);            // Const==Const; Symbol~Symbol; Symbol:=Const
    Dimension resolve(Dimension) const;              // representative
    // element-count equality within the bounded subset (RM 3.4/3.12).
    // Returns Mismatch only when the counts are decidably unequal
    // (SemaReshapeCount); Unresolvable when a symbolic factor's
    // identity/multiplicity differs, which needs reasoning outside the
    // bounded subset (SemaUnresolvableShape) — these are two different
    // diagnostics, so this is an enum, not a bool.
    ElementCountResult sameElementCount(const std::vector<Dimension>& lhs,
                                        const std::vector<Dimension>& rhs) const;
};
```

Contract: this module produces a fully typed `TypedProgram`, or a set
of diagnostics. It uses `InvalidType` internally for recovery, but
never puts it into a graph that verifies clean. It never includes an
MLIR/LLVM header.

## 3a. Typed Computation Graph (TCG)

The TCG is the central compiler IR and the boundary between semantic
analysis and the backend. It is produced by `semantic/`, transformed by
`optimization/`, and consumed by `codegen/`. It implements the verified
semantics of
[03_LANGUAGE_REFERENCE_MANUAL.md §3.20](../language/03_LANGUAGE_REFERENCE_MANUAL.md#320-the-typed-computation-graph).
The real header is
[include/tensora/shared/Ir.h](../../include/tensora/shared/Ir.h) — it
lives in the `shared/` module, alongside the type system, not a
separate `ir/` module.

### 3a.1 Goals and Invariants

The TCG exists so a *verified* program can be optimized and lowered
without re-parsing or re-checking it. `GraphVerifier` (Section 3a.6)
enforces these invariants:

1. **SSA.** Every `Value` has exactly one producer: a `GraphNode`
   output, or an external declaration for an input or weight. Source
   names are gone; only `ValueId`s remain.
2. **Fully typed.** Every `Value` carries a verified `TensorType`
   (named axes and sizes). A graph that reaches the backend never
   contains `InvalidType`.
3. **Acyclic and ordered.** The graph stores nodes in a valid
   topological order. An operand's producer always comes before its
   consumer.
4. **Closed.** Every operand `ValueId` is either a graph parameter
   (input or weight), or the output of an earlier node in the same
   graph.
5. **Single output.** Each graph has exactly one output `ValueId` (the
   `return` value), and the parameters can reach it.

Axis names are part of the TCG; passes and `explain` use them. Lowering
erases them only at the MLIR boundary (ARCH-005).

### 3a.2 Types

The compiler interns and shares types (`TypePtr = const Type*`). Every
tensor's element type is f32 in v1; there is no dtype field (RM 3.23).
Full header:
[include/tensora/shared/Type.h](../../include/tensora/shared/Type.h).

``` cpp
using SymbolId = uint32_t;                 // interned symbolic dimension name

struct Dimension {
    enum class Kind { Const, Symbol };
    Kind      kind;
    int64_t   constant = 0;                 // valid iff kind == Const  (> 0)
    SymbolId  symbol   = 0;                 // valid iff kind == Symbol
};

struct Axis { std::string name; Dimension size; };  // e.g. "batch"=B, "out"=256

class Type {
public:
    enum class Kind { Tensor, Scalar, Invalid };
    Kind kind() const;
};
class TensorType : public Type {  // kind() == Tensor
public:
    const std::vector<Axis>& axes() const;  // ordered, non-empty
    size_t rank() const;
};
class ScalarType  : public Type {};  // kind() == Scalar  (f32 scalar)
class InvalidType : public Type {};  // kind() == Invalid (sema error recovery only)

class TypeContext {  // interns and owns every Type object, plus the SymbolId table
public:
    const TensorType* tensor(std::vector<Axis> axes);
    const ScalarType* scalar();
    const InvalidType* invalid();
    SymbolId          internSymbol(std::string_view name);
    std::string_view  symbolName(SymbolId) const;
};
```

### 3a.3 Values

``` cpp
using ValueId = uint32_t;                   // stable index into TypedGraph::values
using NodeId  = uint32_t;                   // stable index into TypedGraph::nodes
constexpr NodeId kNoProducer = /* -1 */;    // for external (input/weight) values

struct Value {
    ValueId          id;
    const Type*      type;                    // verified; TensorType for all graph values
    NodeId           producer = kNoProducer;  // node whose output this is, or kNoProducer
    SourceSpan       span;                    // where it was defined
    std::string      debugName;               // original source name (best effort, non-unique)
    enum class Origin { Input, Weight, Internal };
    Origin           origin = Origin::Internal;
};
```

Inputs and weights are `Value`s with `origin == Input/Weight` and
`producer == kNoProducer`. They are the graph's parameters.

### 3a.4 Operations

``` cpp
enum class OpKind {
    Add, Sub, Mul, Div,                 // Mul/Div may carry a scalar attribute instead of a 2nd tensor
    MatMul, Linear,
    Relu, Softmax,
    Reshape, Flatten, Transpose,
    Sum, Mean,
    Conv2D, MaxPool
};

// Operator attributes; only the fields relevant to a given OpKind are used.
struct NodeAttrs {
    std::optional<double>      scalar;         // Mul/Div: e.g. `scores * 0.125`
    std::optional<std::string> axisName;       // Softmax/Sum/Mean: axis referenced by name
    std::vector<uint32_t>      perm;           // Transpose: permutation of 0..rank-1
    std::vector<Axis>          reshapeTarget;  // Reshape: target axes (names + sizes)
    int64_t stride = 0, padding = 0, poolSize = 0;  // Conv2D stride/padding; MaxPool size
};

struct GraphNode {
    NodeId               id;
    OpKind                op;
    std::vector<ValueId> inputs;  // tensor operands, in operator argument order
    ValueId               output; // single result value (v1: no multi-result ops)
    NodeAttrs             attrs;
    SourceSpan            span;
    FactSet               facts;  // free-form proven properties (e.g. a fusion candidate) — see 3a.6
};
```

Operand order matches the surface operator: `Linear` inputs are
`[x, W, b]`; `Conv2D` inputs are `[x, W, b]`; `MatMul` inputs are
`[lhs, rhs]`. Keyword operator arguments (`stride`, `axis`, `perm`, and
so on) live in `attrs`, not in `inputs`.

### 3a.5 Graph and Program

``` cpp
class TypedGraph {
public:
    std::string          name;
    std::vector<ValueId> params;             // inputs + weights, declaration order
    ValueId              output;             // the single returned value

    const std::vector<Value>&     values() const;
    const std::vector<GraphNode>& nodes()  const;  // topological order
    const Value&     value(ValueId) const;
    const GraphNode& node(NodeId)  const;

    NodeId                      producerOf(ValueId) const;   // kNoProducer for params
    const std::vector<NodeId>& consumersOf(ValueId) const;

    // Adds a proven property to an already-built node (e.g. a
    // recognized fusion group), without changing graph structure.
    void addFact(NodeId, std::string fact);
};

class TypedProgram {
public:
    TypeContext&                    types();
    const std::vector<TypedGraph>&  graphs() const;
    const TypedGraph*               graph(std::string_view name) const;
};
```

### 3a.6 Construction and Verification

`GraphBuilder` is the *only* approved way to create nodes and values.
It keeps SSA order, topological order, and the producer/consumer
index. Sema uses it, **and** so do test fixtures and downstream mocks
(ARCH-008).

``` cpp
class GraphBuilder {
public:
    explicit GraphBuilder(std::string graphName);

    ValueId addInput (std::string name, const TensorType*, SourceSpan);
    ValueId addWeight(std::string name, const TensorType*, SourceSpan);

    // Append a node; returns its (single) output value. The result type is
    // supplied by the caller (sema computed it via the operator contract).
    ValueId emit(OpKind, std::vector<ValueId> inputs, NodeAttrs,
                 const TensorType* resultType, SourceSpan, std::string debugName);

    void    setOutput(ValueId);
    void    setDebugName(ValueId, std::string name);  // cosmetic; SSA/verification never read it
    TypedGraph finish();                     // hands the built graph to the caller
};
```

`GraphVerifier` re-checks the Section 3a.1 invariants, no matter how
the graph was built. This guards against a buggy pass or a bad
hand-built fixture:

``` cpp
class GraphVerifier {
public:
    // Returns diagnostics for any invariant violation; empty == valid.
    DiagnosticList verify(const TypedGraph&) const;
};
```

Every pass must leave the graph verifier-clean. In debug builds, CI
runs the verifier after every pass.

### 3a.7 Textual Format (`--emit-graph`, golden tests)

`printTypedGraph` (`src/driver/main.cpp`) renders this format: one
`param` line per graph parameter, then one line per node — the node's
debug name, its operator, its operands (by debug name), and its
verified result type via the same `describeType` diagnostics use — then
one `return` line. There is no `%N`-prefixed SSA numbering and no
inline attribute rendering; `--explain` adds only a trailing
`# RM <OpKind>` comment per node.

``` text
graph mlp {
  param x : Tensor[batch=B, feat=784]
  param W1 : Tensor[in=784, out=256]
  param b1 : Tensor[out=256]
  param W2 : Tensor[in=256, out=10]
  param b2 : Tensor[out=10]
  h = linear(x, W1, b1) : Tensor[batch=B, out=256]
  a = relu(h) : Tensor[batch=B, out=256]
  y = linear(a, W2, b2) : Tensor[batch=B, out=10]
  return y
}
```

### 3a.8 Why this shape

- **Value/Node SSA with integer ids** keeps the IR easy to serialize,
  arena-owned, and stable across passes (ids do not move like pointers
  can). This matters for keeping phases decoupled during parallel
  development and for golden testing.
- **Named axes live on the type, not the node.** So every pass and the
  `explain` printer see axis meaning for free, and lowering can erase
  them at one clear point (ARCH-005).
- **Retained `facts`** give optimization passes a place to record a
  proven property (for example, a recognized `Linear+Relu` region)
  without proving it again later. This is the mechanism behind the
  whitepaper's "semantic information is available to optimization"
  design (whitepaper Section 6.1).
- **One output per node and per graph** matches the frozen v1 operator
  set (it has no multi-result ops) and keeps the verifier simple.
  Multi-result ops are a documented future extension.

## 4. Graph Passes

``` cpp
// optimization/GraphPass.h
struct PassResult { bool changed; DiagnosticList diagnostics; PassStats stats; };
struct PassContext { TypeContext& types; DiagnosticEngine& diags; };

class GraphPass {
public:
    virtual ~GraphPass() = default;
    virtual std::string_view name() const = 0;
    virtual PassResult run(TypedGraph&, PassContext&) = 0;
};

class PassManager {
public:
    void add(std::unique_ptr<GraphPass>);
    PassResult runAll(TypedGraph&, PassContext&);    // verifies after each pass in debug
};
```

The six v1 passes (RM 3.21) each implement `GraphPass`:
`ShapeFolding`, `DeadOpElimination`, `ReshapeCancellation`,
`TransposeCancellation`, `ElementwiseFusion`,
`LinearBiasActivationFusion`. (A `ConstantFolding` pass was implemented
and then retired -- see DECISIONS.md -- because its same-kind
scalar-chain combination cannot be proven to preserve `return`'s
observable output under IEEE-754 f32 non-associativity.)

## 4a. Analysis and Loop Transformation

``` cpp
// optimization/Liveness.h (full header: ../../include/tensora/optimization/Liveness.h)
// codegen/LoopTiling.h    (full header: ../../include/tensora/codegen/LoopTiling.h)
struct Interval  { NodeId firstDef; NodeId lastUse; int64_t bytes; };  // bytes from verified shape
struct LivenessResult {
    std::unordered_map<ValueId, Interval> intervals;
    int64_t peakLiveBytes;               // max total size of simultaneously-live tensors
};
class LivenessAnalysis {
public:
    LivenessResult run(const TypedGraph&, /* optional symbol->size bindings */) const;   // pure; no mutation
};

struct TilingPlan { bool tile; std::array<int64_t,3> tileSizes; };     // for matmul i,j,k M/N/K
class TilingDecision {
public:
    // Chooses whether/how to tile from verified shapes (+ optional liveness).
    // MlirLowerer (Section 5) stamps the plan onto the lowered
    // `linalg.matmul` op; MatmulTilingPass then applies exactly that
    // plan via MLIR's own transform infrastructure — one source of
    // truth for the tile sizes, not two independent policies.
    TilingPlan decide(const GraphNode& matmul, const TypedGraph&,
                      const LivenessResult* liveness = nullptr) const;
};
```

Contract: `LivenessAnalysis` is a read-only data-flow analysis. Its
`peakLiveBytes` feeds optimization decisions and the evaluation report.
`TilingDecision` is the project-authored Level-2 contribution: it builds a
plan from verified shapes, and MLIR performs the mechanical loop
rewrite (ARCH-009).

## 5. Lowering / Backend

`codegen/` is the only module that includes any MLIR/LLVM header. The
real class is `MlirLowerer`
([include/tensora/codegen/MlirLowerer.h](../../include/tensora/codegen/MlirLowerer.h)),
not a `Backend`/`CodegenOptions` abstract-interface hierarchy:

``` cpp
class MlirLowerer {
public:
    explicit MlirLowerer(mlir::MLIRContext&, TilingPolicy = {});

    // One mlir::ModuleOp containing one func.func per graph in
    // `program`, each marked for JIT/native invocation
    // (llvm.emit_c_interface).
    mlir::ModuleOp lower(const TypedProgram& program);
};
```

Contract: this module takes a verifier-clean `TypedProgram`, erases
axis names — a concrete dimension becomes a static tensor dimension, a
symbolic dimension becomes dynamic (`?`) — and lowers to the
`func`/`arith`/`tensor`/`linalg` dialects, without redefining any
language semantics. There is no custom Tensora dialect in v1 (ARCH-004).
`src/driver/main.cpp` (Section 6) drives the rest by hand: MLIR's own
passes (bufferization, affine lowering, LLVM dialect conversion),
`mlir::translateModuleToLLVMIR`, and then either
`mlir::ExecutionEngine` (`--run`) or `llvm::TargetMachine` object-file
emission (`-o`) — there is no separate `ObjectModule`/`Driver` class;
this orchestration is the CLI's `main()` itself.

## 6. Driver / CLI

There is no `Driver` class: `src/driver/main.cpp` is a plain `int
main(int argc, char** argv)` that reads a `--stage` flag and a `.tns`
path, then dispatches with a chain of `if (stage == "--emit-mlir")`
checks, string-testing which stage was requested and stopping right
after that stage's artifact is produced.
```

CLI surface (RM/whitepaper Section 11):

``` text
tensora --emit-tokens          file.tns
tensora --emit-parse-tree      file.tns
tensora --emit-ast             file.tns
tensora --emit-ast-dot         file.tns
tensora --check                file.tns
tensora --emit-symbols         file.tns
tensora --emit-graph           file.tns
tensora --emit-graph-dot       file.tns
tensora --explain              file.tns
tensora --emit-optimized-graph file.tns
tensora --emit-optimized-graph-dot file.tns
tensora --emit-mlir            file.tns
tensora --emit-llvm            file.tns
tensora --run                  file.tns
tensora file.tns -o program
```

Each `--emit-*` stage stops right after it produces its artifact. This
is exactly what the golden-test harness drives
(docs/testing/08_TEST_PLAN.md).
