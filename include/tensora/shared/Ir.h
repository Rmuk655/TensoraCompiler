#pragma once
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/SourceSpan.h"
#include "tensora/shared/Type.h"

// The Typed Computation Graph (TCG), RM 3.20; INTERFACES.md §3a. This is the
// boundary between semantic analysis and the backend: a verified, SSA,
// arena-owned graph. Later phases (passes and lowering) read this
// graph without re-checking its semantics. See INTERFACES.md §3a for the full
// reasoning.
namespace tensora {

using ValueId = uint32_t;
using NodeId = uint32_t;
constexpr NodeId kNoProducer = static_cast<NodeId>(-1);

struct Value {
  ValueId id = 0;
  const Type* type = nullptr;      // verified; TensorType for every graph value in v1
  NodeId producer = kNoProducer;   // producing node, or kNoProducer for a parameter
  SourceSpan span;                 // where it was declared/bound in the source
  std::string debugName;           // original source name (best effort, non-unique)

  enum class Origin { Input, Weight, Internal };
  Origin origin = Origin::Internal;
};

enum class OpKind {
  Add, Sub, Mul, Div,
  MatMul, Linear,
  Relu, Softmax,
  Reshape, Flatten, Transpose,
  Sum, Mean,
  Conv2D, MaxPool
};

const char* opKindName(OpKind op);

// Free-form facts kept for later passes and for `explain` (whitepaper
// §6.1) -- for example "symbolic:batch". A pass only ever adds facts,
// never removes them. This class is deliberately simple: a set of
// strings.
class FactSet {
 public:
  void add(std::string fact) { facts_.insert(std::move(fact)); }
  bool has(std::string_view fact) const { return facts_.count(std::string(fact)) != 0; }
  const std::set<std::string>& all() const { return facts_; }

 private:
  std::set<std::string> facts_;
};

// Operator attributes; only the fields relevant to a given OpKind are used.
struct NodeAttrs {
  std::optional<double> scalar;               // Mul/Div: e.g. `scores * 0.125`
  std::optional<std::string> axisName;        // Softmax/Sum/Mean: axis referenced by name
  std::vector<uint32_t> perm;                 // Transpose: permutation of 0..rank-1
  std::vector<Axis> reshapeTarget;             // Reshape: target axes (names + sizes)
  int64_t stride = 0, padding = 0, poolSize = 0;  // Conv2D: stride/padding; MaxPool: size (stride reuses `stride`)
};

struct GraphNode {
  NodeId id = 0;
  OpKind op = OpKind::Add;
  std::vector<ValueId> inputs;   // operand order matches the surface operator
  ValueId output = 0;            // single result value (v1: no multi-result ops)
  NodeAttrs attrs;
  SourceSpan span;
  FactSet facts;
};

class GraphBuilder;

class TypedGraph {
 public:
  std::string name;
  std::vector<ValueId> params;  // inputs + weights, declaration order
  ValueId output = 0;           // the single returned value

  const std::vector<Value>& values() const { return values_; }
  const std::vector<GraphNode>& nodes() const { return nodes_; }

  const Value& value(ValueId id) const { return values_.at(id); }
  const GraphNode& node(NodeId id) const { return nodes_.at(id); }

  NodeId producerOf(ValueId id) const { return values_.at(id).producer; }
  const std::vector<NodeId>& consumersOf(ValueId id) const { return consumers_.at(id); }

  // Adds a proven property to an already-built node (INTERFACES.md §3a.4: a place
  // to record proven properties without re-deriving them). A pass uses
  // this, for example, to mark a fusion group it found, without
  // changing the graph's structure. This never changes SSA order,
  // topological order, or any other GraphVerifier invariant.
  void addFact(NodeId id, std::string fact) { nodes_.at(id).facts.add(std::move(fact)); }

 private:
  friend class GraphBuilder;
  std::vector<Value> values_;
  std::vector<GraphNode> nodes_;
  std::vector<std::vector<NodeId>> consumers_;  // indexed by ValueId
};

class TypedProgram {
 public:
  explicit TypedProgram(TypeContext& types) : types_(types) {}

  TypeContext& types() { return types_; }
  const std::vector<TypedGraph>& graphs() const { return graphs_; }
  const TypedGraph* graph(std::string_view name) const {
    for (const TypedGraph& g : graphs_) {
      if (g.name == name) return &g;
    }
    return nullptr;
  }
  void addGraph(TypedGraph g) { graphs_.push_back(std::move(g)); }

 private:
  TypeContext& types_;
  std::vector<TypedGraph> graphs_;
};

// The one allowed way to create nodes and values (ARCH-008). It keeps
// SSA order, topological order, and the producer/consumer index
// correct. Semantic analysis uses it, and so do test fixtures and
// downstream mocks.
class GraphBuilder {
 public:
  explicit GraphBuilder(std::string graphName) { graph_.name = std::move(graphName); }

  ValueId addInput(std::string name, const TensorType* type, SourceSpan span) {
    return addParam(std::move(name), type, span, Value::Origin::Input);
  }
  ValueId addWeight(std::string name, const TensorType* type, SourceSpan span) {
    return addParam(std::move(name), type, span, Value::Origin::Weight);
  }

  // Adds a node and returns its one output value. The caller supplies
  // the result type (semantic analysis found it through the operator
  // contract).
  //
  // A correct caller (SemanticAnalyzer) only ever passes ValueIds it
  // already got from this same builder, so `inputs` is always in
  // range. This function does *not* throw or assert on an out-of-range
  // or forward-referencing ValueId. Instead, it bounds-checks the
  // consumer-index update rather than using .at(). This lets
  // GraphVerifier do its real job: catch bugs from a broken pass or a
  // bad hand-built fixture (INTERFACES.md §3a.6). Without this, a hand-built test
  // could not build a dangling or forward-referencing operand to
  // check that GraphVerifier catches it.
  ValueId emit(OpKind op, std::vector<ValueId> inputs, NodeAttrs attrs, const TensorType* resultType,
               SourceSpan span, std::string debugName) {
    ValueId out = pushValue(resultType, span, std::move(debugName), Value::Origin::Internal);
    NodeId nid = static_cast<NodeId>(graph_.nodes_.size());
    GraphNode node;
    node.id = nid;
    node.op = op;
    node.inputs = std::move(inputs);
    node.output = out;
    node.attrs = std::move(attrs);
    node.span = span;
    for (ValueId in : node.inputs) {
      if (in < graph_.consumers_.size()) graph_.consumers_[in].push_back(nid);
    }
    graph_.nodes_.push_back(std::move(node));
    graph_.values_[out].producer = nid;
    return out;
  }

  void setOutput(ValueId v) { graph_.output = v; }

  // Sets a cosmetic `debugName`, "best effort, non-unique" as Value's
  // own comment says. Diagnostics and driver dumps (--emit-graph,
  // --explain) use this name. SSA, verification, and lowering never
  // read it. This lets a caller, such as SemanticAnalyzer, attach the
  // source-level name after the fact, once it knows it (SemanticAnalyzer
  // only learns the statement's bound name after `emit` already picked
  // the OpKind-specific helper and returned). This avoids passing that
  // name through every call site.
  void setDebugName(ValueId id, std::string name) { graph_.values_.at(id).debugName = std::move(name); }

  TypedGraph finish() { return std::move(graph_); }

 private:
  ValueId addParam(std::string name, const TensorType* type, SourceSpan span, Value::Origin origin) {
    ValueId id = pushValue(type, span, std::move(name), origin);
    graph_.params.push_back(id);
    return id;
  }
  ValueId pushValue(const Type* type, SourceSpan span, std::string debugName, Value::Origin origin) {
    ValueId id = static_cast<ValueId>(graph_.values_.size());
    Value v;
    v.id = id;
    v.type = type;
    v.span = span;
    v.debugName = std::move(debugName);
    v.origin = origin;
    graph_.values_.push_back(std::move(v));
    graph_.consumers_.emplace_back();
    return id;
  }

  TypedGraph graph_;
};

// Re-checks INTERFACES.md §3a.1's invariants, no matter how the graph was built.
// This catches bugs from a broken pass or a bad hand-built fixture. An
// empty result means the graph is valid. Every diagnostic here uses
// DiagCode::IrInvariantViolation.
class GraphVerifier {
 public:
  DiagnosticList verify(const TypedGraph& g) const;
};

}  // namespace tensora
