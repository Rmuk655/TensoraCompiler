// This file implements the Level-2 tiling decision.
// It picks a matmul tile size from verified shapes and, when available,
// the Level-1 liveness estimate. Codegen applies this decision to the
// lowered MLIR (see MatmulTilingPass.cpp).
#include "tensora/codegen/LoopTiling.h"

#include <algorithm>

namespace tensora {

TilingPlan TilingDecision::decide(const GraphNode& matmul, const TypedGraph& graph,
                                   const LivenessResult* liveness) const {
  TilingPlan plan;
  if (matmul.op != OpKind::MatMul) return plan;  // policy scope: the C = A @ B reference computation

  const auto* lhs = static_cast<const TensorType*>(graph.value(matmul.inputs[0]).type);
  const auto* rhs = static_cast<const TensorType*>(graph.value(matmul.inputs[1]).type);
  if (lhs->rank() != 2 || rhs->rank() != 2) return plan;  // batched matmul: not this policy's scope

  Dimension m = lhs->axes()[0].size;
  Dimension k = lhs->axes()[1].size;
  Dimension n = rhs->axes()[1].size;
  if (m.kind != Dimension::Kind::Const || k.kind != Dimension::Kind::Const || n.kind != Dimension::Kind::Const) {
    return plan;  // symbolic dims: conservative default is "do not tile" (no static loop bound to plan around)
  }

  if (m.constant < policy_.minDimToTile || n.constant < policy_.minDimToTile || k.constant < policy_.minDimToTile) {
    return plan;  // too small: tiling overhead isn't worth it
  }

  int64_t tileSize = policy_.maxTileSize;
  if (liveness && exceedsMemoryBudget(*liveness, policy_.memoryBudgetBytes)) {
    tileSize = policy_.conservativeTileSize;  // a real decision that consumes the peak-memory estimate (LIVE-004)
  }

  plan.tile = true;
  plan.tileSizes = {std::min(m.constant, tileSize), std::min(n.constant, tileSize), std::min(k.constant, tileSize)};
  return plan;
}

}  // namespace tensora
