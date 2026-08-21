# Tensora Cross-Phase Interfaces

> Concrete C++ interface signatures at each compiler boundary. These are
> the contracts that let five team members build in parallel. Once the
> team freezes them, they stay stable. Companion files:
> [ARCHITECTURE.md](ARCHITECTURE.md), [IR.md](IR.md). This C++ is
> illustrative; the exact headers live in `include/tensora/`.

## 0. Shared Foundations

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
    SemaConvChannel, SemaConvArithmetic, SemaUnresolvableShape
};
struct DiagnosticNote { SourceSpan span; std::string message; };
struct Diagnostic {
    Severity                    severity;
    DiagCode                    code;
    SourceSpan                  span;
    std::string                 message;      // human text, e.g. "784 != 512"
    std::vector<DiagnosticNote> notes;
};
using DiagnosticList = std::vector<Diagnostic>;

class DiagnosticEngine {
public:
    void report(Diagnostic);
    bool hasErrors() const;
    const DiagnosticList& all() const;
    std::string render(const SourceManager&) const;   // caret diagnostics for the CLI
};
```

Every phase takes a `DiagnosticEngine&` and reports into it. No phase
prints directly.

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
see 06_COMPILER_ARCHITECTURE.md §6.6 and whitepaper §4.1.

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
// shared/DimSolver.h
class DimSolver {
public:
    bool unify(Dimension a, Dimension b);            // Const==Const; Symbol~Symbol; Symbol:=Const
    Dimension resolve(Dimension) const;              // representative
    // element-count equality within the bounded subset (RM 3.4)
    bool sameElementCount(const std::vector<Dimension>& lhs,
                          const std::vector<Dimension>& rhs);
};
```

Contract: this module produces a fully typed `TypedProgram`, or a set
of diagnostics. It uses `InvalidType` internally for recovery, but
never puts it into a graph that verifies clean. It never includes an
MLIR/LLVM header.

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
// optimization/Liveness.h  — data-flow analysis over the SSA typed graph
struct Interval  { NodeId firstDef; NodeId lastUse; int64_t bytes; };  // bytes from verified shape
struct LivenessResult {
    std::unordered_map<ValueId, Interval> intervals;
    int64_t peakLiveBytes;               // max total size of simultaneously-live tensors
};
class LivenessAnalysis {
public:
    LivenessResult run(const TypedGraph&) const;   // pure analysis; no mutation
};

// codegen/LoopTiling.h  — Level-2 decision consumed by the MLIR backend
struct TilingPlan { bool tile; std::array<int64_t,3> tileSizes; };     // for matmul i,j,k
class TilingDecision {
public:
    // Chooses whether/how to tile from verified shapes (+ optional liveness).
    // The MLIR backend applies the plan via MLIR's transform infrastructure.
    TilingPlan decide(const GraphNode& matmul, const TypedGraph&,
                      const LivenessResult&) const;
};
```

Contract: `LivenessAnalysis` is a read-only data-flow analysis. Its
`peakLiveBytes` feeds optimization decisions and the evaluation report.
`TilingDecision` is the team-authored Level-2 contribution: it builds a
plan from verified shapes, and MLIR performs the mechanical loop
rewrite (ARCH-009).

## 5. Lowering / Backend

``` cpp
// codegen/Backend.h   — the ONLY module that includes MLIR/LLVM
struct CodegenOptions { OptLevel opt; bool emitDebugInfo; std::string targetTriple; };
struct ObjectModule   { std::vector<std::byte> object; std::string entrySymbol; };

class Backend {
public:
    virtual ~Backend() = default;
    virtual Result<ObjectModule> compile(const TypedProgram&, const CodegenOptions&) = 0;
    virtual std::string emitMlir(const TypedProgram&) = 0;   // for --emit-mlir / golden
    virtual std::string emitLlvm(const TypedProgram&) = 0;   // for --emit-llvm / golden
};

class MlirCpuBackend : public Backend { /* TCG -> func/linalg/tensor/arith/scf -> LLVM -> obj */ };
```

Contract: this module takes a verifier-clean `TypedProgram`, erases
axis names, maps a concrete dimension to a static tensor dimension and
a symbolic dimension to a dynamic one, and produces MLIR, LLVM, or an
object file without redefining any language semantics. There is no
custom dialect in v1 (ARCH-004).

## 6. Driver / CLI

``` cpp
// driver/Driver.h
enum class Stage { Tokens, Ast, Check, Explain, Graph, Mlir, Llvm, Executable };
struct DriverOptions { Stage stage; std::string inputPath, outputPath; CodegenOptions codegen; };

class Driver {
public:
    int run(const DriverOptions&);       // orchestrates phases, renders diagnostics
};
```

CLI surface (RM/whitepaper Section 11):

``` text
tensora --emit-tokens          file.tns
tensora --emit-parse-tree      file.tns
tensora --emit-ast             file.tns
tensora --emit-ast-dot         file.tns
tensora --check                file.tns
tensora --emit-graph           file.tns
tensora --emit-graph-dot       file.tns
tensora --explain              file.tns
tensora --emit-optimized-graph file.tns
tensora --emit-mlir            file.tns
tensora --emit-llvm            file.tns
tensora file.tns -o program
```

Each `--emit-*` stage stops right after it produces its artifact. This
is exactly what the golden-test harness drives
(docs/testing/08_TEST_PLAN.md).
