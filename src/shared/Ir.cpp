// This file implements the Typed Computation Graph (TCG).
// Semantic analysis creates the TCG. Optimization and codegen read it.
#include "tensora/shared/Ir.h"

namespace tensora {

const char* opKindName(OpKind op) {
  switch (op) {
    case OpKind::Add: return "add";
    case OpKind::Sub: return "sub";
    case OpKind::Mul: return "mul";
    case OpKind::Div: return "div";
    case OpKind::MatMul: return "matmul";
    case OpKind::Linear: return "linear";
    case OpKind::Relu: return "relu";
    case OpKind::Softmax: return "softmax";
    case OpKind::Reshape: return "reshape";
    case OpKind::Flatten: return "flatten";
    case OpKind::Transpose: return "transpose";
    case OpKind::Sum: return "sum";
    case OpKind::Mean: return "mean";
    case OpKind::Conv2D: return "conv2d";
    case OpKind::MaxPool: return "maxpool";
  }
  return "?";
}

namespace {
void report(DiagnosticList& diags, const SourceSpan& span, std::string msg) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = DiagCode::IrInvariantViolation;
  d.span = span;
  d.message = std::move(msg);
  diags.push_back(std::move(d));
}
}  // namespace

DiagnosticList GraphVerifier::verify(const TypedGraph& g) const {
  DiagnosticList diags;
  const size_t numValues = g.values().size();
  const size_t numNodes = g.nodes().size();

  // GraphBuilder always sets node.id to that node's position in the
  // node list. A hand-built fixture might not, so this checks that
  // fact directly before the rest of this function relies on it.
  for (size_t i = 0; i < numNodes; ++i) {
    if (g.nodes()[i].id != i) {
      report(diags, g.nodes()[i].span, "node id does not match its position in the node list");
    }
  }

  std::vector<bool> isParam(numValues, false);
  for (ValueId p : g.params) {
    if (p < numValues) isParam[p] = true;
  }

  // Every value's producer must match its origin: a param has no
  // producer, and every other value is produced by exactly the node
  // that records it as an output (SSA, INTERFACES.md §3a.1).
  for (size_t vid = 0; vid < numValues; ++vid) {
    const Value& v = g.values()[vid];
    if (v.origin != Value::Origin::Internal) {
      if (v.producer != kNoProducer) {
        report(diags, v.span, "parameter value '" + v.debugName + "' unexpectedly has a producer node");
      }
      if (!isParam[vid]) {
        report(diags, v.span, "input/weight value '" + v.debugName + "' is not listed in the graph's params");
      }
    } else if (v.producer == kNoProducer || v.producer >= numNodes) {
      report(diags, v.span, "internal value '" + v.debugName + "' has no valid producer (dangling value)");
    } else if (g.nodes()[v.producer].output != vid) {
      report(diags, v.span,
             "internal value '" + v.debugName + "' is not the recorded output of its producer node");
    }
  }

  // Checks that the graph is closed, acyclic, and in topological
  // order: every node's operands are either a parameter, or produced
  // by a strictly earlier node (INTERFACES.md §3a.1).
  for (size_t i = 0; i < numNodes; ++i) {
    const GraphNode& node = g.nodes()[i];
    for (ValueId in : node.inputs) {
      if (in >= numValues) {
        report(diags, node.span, std::string("node '") + opKindName(node.op) + "' references a dangling operand");
        continue;
      }
      const Value& operand = g.values()[in];
      if (operand.producer == kNoProducer) {
        if (!isParam[in]) {
          report(diags, node.span,
                 std::string("node '") + opKindName(node.op) +
                     "' operand is neither a parameter nor produced by any node");
        }
        continue;
      }
      if (!(operand.producer < i)) {
        report(diags, node.span,
               std::string("node '") + opKindName(node.op) +
                   "' operand is not produced by a strictly earlier node (cycle or bad order)");
      }
    }
  }

  // Checks for a single, valid output (INTERFACES.md §3a.1). Reachability from
  // params already follows from the closed and topological checks
  // above: every node's operands trace back to a param or an earlier
  // node, transitively. So only a bounds check is needed here.
  if (numValues == 0 || g.output >= numValues) {
    report(diags, SourceSpan{}, "graph '" + g.name + "' has no valid output value");
  }

  return diags;
}

}  // namespace tensora
