#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "tensora/optimization/GraphPass.h"

// The six frozen v1 graph passes (RM 3.21). Every *structural* pass
// (Dead, Reshape, Transpose, ShapeFolding) builds a fresh graph with
// GraphBuilder (ARCH-008: GraphBuilder is the only allowed way to
// create nodes and values), instead of changing TypedGraph's internals
// directly. It then move-assigns the result back into the caller's
// reference. This satisfies the INTERFACES.md signature
// `run(TypedGraph&, PassContext&)`, which looks like true in-place
// mutation to the caller, without breaking that invariant. The two
// *fusion* passes only add annotations (TypedGraph::addFact) and
// change the graph in place directly, since tagging changes no
// structural invariant.
namespace tensora {

namespace {

bool isElementwise(OpKind op) {
  return op == OpKind::Add || op == OpKind::Sub || op == OpKind::Mul || op == OpKind::Div;
}

// True if `node` already carries a fact starting with `prefix`. Both
// fusion passes below tag a node once and must never tag it again: since
// PassManager::runAll re-sweeps the pass list to a fixed point (e.g. a
// later DeadOpElimination sweep can drop a node's other consumer, which
// then makes an earlier fusion pass see this same pair as eligible
// again), a fusion pass that re-tagged an already-tagged pair every
// sweep would report `changed` forever and never reach a fixed point.
bool hasFactPrefix(const GraphNode& node, std::string_view prefix) {
  for (const std::string& fact : node.facts.all()) {
    if (fact.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

// Copies every param 1:1 into a fresh builder for the same graph name.
// This records the old -> new ValueId mapping. Here, that mapping is
// the identity, since params are added in the same order first. This
// code still computes it explicitly, instead of assuming it.
GraphBuilder startRebuild(const TypedGraph& graph, std::unordered_map<ValueId, ValueId>& oldToNew) {
  GraphBuilder builder(graph.name);
  for (ValueId p : graph.params) {
    const Value& v = graph.value(p);
    const auto* tt = static_cast<const TensorType*>(v.type);
    ValueId newId = (v.origin == Value::Origin::Input) ? builder.addInput(v.debugName, tt, v.span)
                                                        : builder.addWeight(v.debugName, tt, v.span);
    oldToNew[p] = newId;
  }
  return builder;
}

}  // namespace

// --- DeadOpElimination ---
// Precondition: none. This pass is legal because a node whose output
// `return` never uses, even transitively, has no observable effect at
// all. Removing it cannot change the graph's return value (RM 3.21).
class DeadOpEliminationPass : public GraphPass {
 public:
  std::string_view name() const override { return "DeadOpElimination"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    const size_t numValues = graph.values().size();
    std::vector<bool> reachableNode(numNodes, false);
    std::vector<bool> visitedValue(numValues, false);
    std::vector<ValueId> worklist{graph.output};

    while (!worklist.empty()) {
      ValueId v = worklist.back();
      worklist.pop_back();
      if (v >= numValues || visitedValue[v]) continue;
      visitedValue[v] = true;
      NodeId prod = graph.producerOf(v);
      if (prod == kNoProducer || prod >= numNodes || reachableNode[prod]) continue;
      reachableNode[prod] = true;
      for (ValueId in : graph.node(prod).inputs) worklist.push_back(in);
    }

    size_t deadCount = 0;
    for (bool r : reachableNode) {
      if (!r) ++deadCount;
    }
    if (deadCount == 0) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    for (size_t i = 0; i < numNodes; ++i) {
      if (!reachableNode[i]) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(oldToNew.at(in));
      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), node.attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(oldToNew.at(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = deadCount;
    return result;
  }
};

// --- ReshapeCancellation ---
// Pattern: reshape(reshape(x, A), B), where only the outer reshape
// uses the inner one. Precondition: reshape is a pure re-labeling
// operation. It needs no name relationship between the source and
// result axes (RM 3.12). So composing two re-labelings, each already
// checked on its own by semantic analysis, is the same as one reshape
// straight to the final target B.
class ReshapeCancellationPass : public GraphPass {
 public:
  std::string_view name() const override { return "ReshapeCancellation"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    std::unordered_set<NodeId> skip;
    std::unordered_map<ValueId, ValueId> bypass;  // skipped node's output -> its own input

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& outer = graph.node(i);
      if (outer.op != OpKind::Reshape) continue;
      NodeId innerId = graph.producerOf(outer.inputs[0]);
      if (innerId == kNoProducer) continue;
      const GraphNode& inner = graph.node(innerId);
      if (inner.op != OpKind::Reshape) continue;
      if (graph.consumersOf(inner.output).size() != 1) continue;  // inner used elsewhere: cannot drop it
      skip.insert(innerId);
      bypass[inner.output] = inner.inputs[0];
    }
    if (skip.empty()) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    auto resolve = [&](ValueId v) {
      while (bypass.count(v)) v = bypass.at(v);
      return oldToNew.at(v);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      if (skip.count(static_cast<NodeId>(i))) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(resolve(in));
      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), node.attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(resolve(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = skip.size();
    return result;
  }
};

// --- TransposeCancellation ---
// Pattern: transpose(transpose(x, P1), P2), where only the outer
// transpose uses the inner one. Precondition: composing two
// permutations and applying the result once gives the same axis order
// as applying each permutation in sequence (permutation composition is
// associative). If the composed permutation is the identity, the two
// transposes cancel out completely.
class TransposeCancellationPass : public GraphPass {
 public:
  std::string_view name() const override { return "TransposeCancellation"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    std::unordered_set<NodeId> skip;
    std::unordered_map<ValueId, ValueId> bypass;
    std::unordered_map<NodeId, std::vector<uint32_t>> overridePerm;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& outer = graph.node(i);
      if (outer.op != OpKind::Transpose) continue;
      NodeId innerId = graph.producerOf(outer.inputs[0]);
      if (innerId == kNoProducer) continue;
      const GraphNode& inner = graph.node(innerId);
      if (inner.op != OpKind::Transpose) continue;
      if (graph.consumersOf(inner.output).size() != 1) continue;
      if (inner.attrs.perm.size() != outer.attrs.perm.size()) continue;  // defensive; sema guarantees this anyway

      // z[i] = y[P2[i]] = x[P1[P2[i]]]  (P1 = inner.perm, P2 = outer.perm)
      std::vector<uint32_t> composed(outer.attrs.perm.size());
      bool isIdentity = true;
      for (size_t k = 0; k < composed.size(); ++k) {
        composed[k] = inner.attrs.perm[outer.attrs.perm[k]];
        if (composed[k] != k) isIdentity = false;
      }

      skip.insert(innerId);
      bypass[inner.output] = inner.inputs[0];
      if (isIdentity) {
        skip.insert(static_cast<NodeId>(i));
        bypass[outer.output] = inner.inputs[0];
      } else {
        overridePerm[static_cast<NodeId>(i)] = std::move(composed);
      }
    }
    if (skip.empty()) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    auto resolve = [&](ValueId v) {
      while (bypass.count(v)) v = bypass.at(v);
      return oldToNew.at(v);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      NodeId nid = static_cast<NodeId>(i);
      if (skip.count(nid)) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(resolve(in));

      NodeAttrs attrs = node.attrs;
      auto it = overridePerm.find(nid);
      if (it != overridePerm.end()) attrs.perm = it->second;

      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(resolve(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = skip.size();
    return result;
  }
};

// --- ShapeFolding ---
// Pattern: a Reshape or Flatten node whose result type exactly matches
// its own input type. Precondition: such a node is a pure no-op
// re-labeling back to the same names and sizes it started with. So its
// consumers can use its input directly (RM 3.12/3.13 place no
// requirement that a reshape or flatten actually change anything).
class ShapeFoldingPass : public GraphPass {
 public:
  std::string_view name() const override { return "ShapeFolding"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    std::unordered_set<NodeId> skip;
    std::unordered_map<ValueId, ValueId> bypass;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& node = graph.node(i);
      if (node.op != OpKind::Reshape && node.op != OpKind::Flatten) continue;
      const Type* inType = graph.value(node.inputs[0]).type;
      const Type* outType = graph.value(node.output).type;
      if (!typesEqual(inType, outType)) continue;
      skip.insert(static_cast<NodeId>(i));
      bypass[node.output] = node.inputs[0];
    }
    if (skip.empty()) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    auto resolve = [&](ValueId v) {
      while (bypass.count(v)) v = bypass.at(v);
      return oldToNew.at(v);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      if (skip.count(static_cast<NodeId>(i))) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(resolve(in));
      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), node.attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(resolve(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = skip.size();
    return result;
  }
};

// --- ElementwiseFusion ---
// Pattern: the longest chain of two or more elementwise ops (Add, Sub,
// Mul, Div), where the next op in the chain is the only consumer of
// each intermediate result. Precondition: none is needed just to *tag*
// the chain, since this pass only records a fact; it does not change
// the graph's structure or result. Lowering time decides whether it is
// legal to actually fuse a tagged chain's codegen into one
// linalg.generic, and lowering is where it reads the tag (IR.md §4:
// facts let a later phase use a proven property "without re-deriving
// them").
class ElementwiseFusionPass : public GraphPass {
 public:
  std::string_view name() const override { return "ElementwiseFusion"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    size_t groupCounter = 0;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& node = graph.node(i);
      if (!isElementwise(node.op)) continue;
      if (hasFactPrefix(node, "fusion_group_")) continue;  // already tagged in an earlier sweep

      NodeId pred = graph.producerOf(node.inputs[0]);
      bool isChainStart = !(pred != kNoProducer && isElementwise(graph.node(pred).op) &&
                            graph.consumersOf(node.inputs[0]).size() == 1);
      if (!isChainStart) continue;

      std::vector<NodeId> chain{static_cast<NodeId>(i)};
      NodeId cur = static_cast<NodeId>(i);
      while (graph.consumersOf(graph.node(cur).output).size() == 1) {
        NodeId next = graph.consumersOf(graph.node(cur).output)[0];
        if (!isElementwise(graph.node(next).op)) break;
        chain.push_back(next);
        cur = next;
      }
      if (chain.size() < 2) continue;

      std::string tag = "fusion_group_" + std::to_string(groupCounter++);
      for (NodeId n : chain) graph.addFact(n, tag);
      result.changed = true;
      ++result.stats.numChanges;
    }
    return result;
  }
};

// --- LinearBiasActivationFusion ---
// Pattern: a Relu node whose one operand is a Linear node's output,
// where that Relu is the Linear's only consumer. Precondition: the
// same as ElementwiseFusion -- this pass only tags the pair (RM 3.9's
// Linear already includes the bias-add, so "Linear+Bias+Activation"
// means exactly "Linear immediately followed by an activation"). A
// later lowering pass fuses the codegen.
class LinearBiasActivationFusionPass : public GraphPass {
 public:
  std::string_view name() const override { return "LinearBiasActivationFusion"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    size_t groupCounter = 0;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& relu = graph.node(i);
      if (relu.op != OpKind::Relu) continue;
      NodeId pred = graph.producerOf(relu.inputs[0]);
      if (pred == kNoProducer) continue;
      const GraphNode& linear = graph.node(pred);
      if (linear.op != OpKind::Linear) continue;
      if (graph.consumersOf(linear.output).size() != 1) continue;
      if (hasFactPrefix(linear, "fused_linear_relu_")) continue;  // already tagged in an earlier sweep

      std::string tag = "fused_linear_relu_" + std::to_string(groupCounter++);
      graph.addFact(pred, tag);
      graph.addFact(static_cast<NodeId>(i), tag);
      result.changed = true;
      ++result.stats.numChanges;
    }
    return result;
  }
};

std::unique_ptr<GraphPass> createShapeFoldingPass() { return std::make_unique<ShapeFoldingPass>(); }
std::unique_ptr<GraphPass> createDeadOpEliminationPass() { return std::make_unique<DeadOpEliminationPass>(); }
std::unique_ptr<GraphPass> createReshapeCancellationPass() { return std::make_unique<ReshapeCancellationPass>(); }
std::unique_ptr<GraphPass> createTransposeCancellationPass() {
  return std::make_unique<TransposeCancellationPass>();
}
std::unique_ptr<GraphPass> createElementwiseFusionPass() { return std::make_unique<ElementwiseFusionPass>(); }
std::unique_ptr<GraphPass> createLinearBiasActivationFusionPass() {
  return std::make_unique<LinearBiasActivationFusionPass>();
}

}  // namespace tensora
