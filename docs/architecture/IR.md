# Tensora Typed Computation Graph (TCG) — IR Design

> The TCG is the central compiler IR and the boundary between semantic
> analysis and the backend. It is produced by `semantic/`, transformed by
> `optimization/`, and consumed by `codegen/`. It implements the verified
> semantics of [../language/03_LANGUAGE_REFERENCE_MANUAL.md](../language/03_LANGUAGE_REFERENCE_MANUAL.md)
> Section 3.20. This document is normative for the `ir/` module.

## 1. Goals and Invariants

The TCG exists so a *verified* program can be optimized and lowered
without re-parsing or re-checking it. `GraphVerifier` (Section 6)
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

## 2. Types

The compiler interns and shares types (`TypePtr = const Type*`). Every
tensor's element type is f32 in v1; there is no dtype field (RM 3.23).

``` cpp
using SymbolId = uint32_t;                 // interned symbolic dimension name

struct Dimension {
    enum class Kind { Const, Symbol };
    Kind      kind;
    int64_t   constant = 0;                 // valid iff kind == Const  (> 0)
    SymbolId  symbol   = 0;                 // valid iff kind == Symbol
    // Note: v1 output dimensions are only Const or Symbol. Products of a
    // symbol with a constant are used transiently during element-count
    // checking (RM 3.4) but are never stored on a Value.
};

struct Axis {
    std::string name;                       // e.g. "batch", "out", "kseq"
    Dimension   size;
};

class Type {
public:
    enum class Kind { Tensor, Scalar, Invalid };
    Kind kind() const;
    virtual ~Type() = default;
};

class TensorType : public Type {            // kind() == Tensor
public:
    const std::vector<Axis>& axes() const;  // ordered, non-empty
    size_t rank() const;
};

class ScalarType  : public Type {};         // kind() == Scalar  (f32 scalar)
class InvalidType : public Type {};         // kind() == Invalid (sema error recovery only)
```

`TypeContext` interns and owns every `Type` object, plus the
`SymbolId` table:

``` cpp
class TypeContext {
public:
    const TensorType* tensor(std::vector<Axis> axes);   // interned
    const ScalarType* scalar();
    const InvalidType* invalid();
    SymbolId          internSymbol(std::string_view name);
    std::string_view  symbolName(SymbolId) const;
};
```

## 3. Values

``` cpp
using ValueId = uint32_t;                   // stable index into TypedGraph::values
using NodeId  = uint32_t;                   // stable index into TypedGraph::nodes
constexpr NodeId kNoProducer = UINT32_MAX;  // for external (input/weight) values

struct Value {
    ValueId          id;
    const Type*      type;                   // verified; TensorType for all graph values
    NodeId           producer = kNoProducer; // node whose output this is, or kNoProducer
    SourceSpan       span;                    // where it was defined
    std::string      debugName;               // original source name (best effort, non-unique)

    enum class Origin { Input, Weight, Internal };
    Origin           origin = Origin::Internal;
};
```

Inputs and weights are `Value`s with `origin == Input/Weight` and
`producer == kNoProducer`. They are the graph's parameters.

## 4. Operations

``` cpp
enum class OpKind {
    // elementwise (Mul/Div may carry a scalar attribute instead of a 2nd tensor)
    Add, Sub, Mul, Div,
    // linear algebra
    MatMul, Linear,
    // activation
    Relu, Softmax,
    // shape
    Reshape, Flatten, Transpose,
    // reduction
    Sum, Mean,
    // vision
    Conv2D, MaxPool
};

// Operator attributes. Only the fields relevant to the OpKind are used.
struct NodeAttrs {
    // Mul/Div: optional scalar operand (e.g. scores * 0.125)
    std::optional<double>          scalar;
    // Softmax / Sum / Mean: axis referenced by name (resolved to index at lowering)
    std::optional<std::string>     axisName;
    // Transpose: permutation of 0..rank-1
    std::vector<uint32_t>          perm;
    // Reshape: target axes (names + sizes)
    std::vector<Axis>              reshapeTarget;
    // Conv2D: stride, padding ; MaxPool: size, stride
    int64_t stride = 0, padding = 0, poolSize = 0;
};

struct GraphNode {
    NodeId               id;
    OpKind               op;
    std::vector<ValueId> inputs;             // tensor operands, in operator argument order
    ValueId              output;             // single result value (v1)
    NodeAttrs            attrs;
    SourceSpan           span;

    // Retained semantic facts for passes/explain (e.g. "batch is symbolic",
    // "this Linear+Relu is a fusion candidate"). Free-form, additive.
    FactSet              facts;
};
```

Operand order matches the surface operator: `Linear` inputs are
`[x, W, b]`; `Conv2D` inputs are `[x, W, b]`; `MatMul` inputs are
`[lhs, rhs]`. Keyword operator arguments (`stride`, `axis`, `perm`, and
so on) live in `attrs`, not in `inputs`.

## 5. Graph and Program

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

    // Producer/consumer queries (built once, kept in sync by GraphBuilder/passes)
    NodeId                 producerOf(ValueId) const;    // kNoProducer for params
    const std::vector<NodeId>& consumersOf(ValueId) const;
};

class TypedProgram {
public:
    TypeContext&                  types();
    const std::vector<TypedGraph>& graphs() const;
    const TypedGraph*             graph(std::string_view name) const;
};
```

## 6. Construction and Verification

`GraphBuilder` is the *only* approved way to create nodes and values.
It keeps SSA order, topological order, and the producer/consumer
index. Sema uses it, **and** so do test fixtures and downstream mocks
(ARCH-8).

``` cpp
class GraphBuilder {
public:
    explicit GraphBuilder(TypedProgram& program, std::string graphName);

    ValueId addInput (std::string name, const TensorType*, SourceSpan);
    ValueId addWeight(std::string name, const TensorType*, SourceSpan);

    // Append a node; returns its (single) output value. The result type is
    // supplied by the caller (sema computed it via the operator contract).
    ValueId emit(OpKind, std::vector<ValueId> inputs, NodeAttrs,
                 const TensorType* resultType, SourceSpan, std::string debugName);

    void    setOutput(ValueId);
    TypedGraph finish();                     // validates topo order, returns the graph
};
```

`GraphVerifier` re-checks the Section 1 invariants, no matter how the
graph was built. This guards against a buggy pass or a bad hand-built
fixture:

``` cpp
class GraphVerifier {
public:
    // Returns diagnostics for any invariant violation; empty == valid.
    DiagnosticList verify(const TypedGraph&) const;
};
```

Every pass must leave the graph verifier-clean. In debug builds, CI
runs the verifier after every pass.

## 7. Textual Format (`--emit-graph`, golden tests)

This format is deterministic, SSA-style, one node per line. It is the
format the tester checks against golden output
(docs/testing/08_TEST_PLAN.md).

``` text
graph mlp(
    %x : Tensor[batch=B, feat=784]  input,
    %W1: Tensor[in=784, out=256]    weight,
    %b1: Tensor[out=256]            weight,
    %W2: Tensor[in=256, out=10]     weight,
    %b2: Tensor[out=10]             weight
) {
    %0 = linear %x, %W1, %b1        : Tensor[batch=B, out=256]
    %1 = relu   %0                  : Tensor[batch=B, out=256]
    %2 = linear %1, %W2, %b2        : Tensor[batch=B, out=10]
    return %2
}
```

Attributes render inline, e.g.:

``` text
    %3 = conv2d %img, %Wc, %bc {stride=1, padding=1} : Tensor[batch=B, channel=32, height=224, width=224]
    %4 = softmax %scores {axis=kseq}                 : Tensor[batch=B, head=12, qseq=S, kseq=S]
    %5 = transpose %K {perm=[0,1,3,2]}               : Tensor[batch=B, head=12, dim=64, kseq=S]
    %6 = mul %scores {scalar=0.125}                  : Tensor[batch=B, head=12, qseq=S, kseq=S]
```

## 8. Why this shape

- **Value/Node SSA with integer ids** keeps the IR easy to serialize,
  arena-owned, and stable across passes (ids do not move like pointers
  can). This matters for five-person parallel work and golden testing.
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
