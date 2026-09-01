#pragma once
#include <array>
#include <cstdint>

#include "tensora/optimization/Liveness.h"
#include "tensora/shared/Ir.h"

// Level-2 loop optimization (ARCH-009; INTERFACES.md §4a). This file
// holds the tiling *decision*. It runs on verified shapes, plus
// optional tensor liveness data, at the Typed Computation Graph level,
// before any MLIR exists. MlirLowerer calls this decision and stamps
// the result onto the lowered `linalg.matmul` op as an attribute.
// MatmulTilingPass (codegen/MatmulTilingPass.h) then applies exactly
// that stamped plan; it does not compute its own policy. This keeps
// one source of truth for the tile sizes, and this decision stays
// unit-testable on its own, without any MLIR dependency
// (ARCHITECTURE.md §2).
namespace tensora {

struct TilingPlan {
  bool tile = false;
  std::array<int64_t, 3> tileSizes = {0, 0, 0};  // M, N, K
};

struct TilingPolicy {
  int64_t minDimToTile = 128;   // Below this size on M, N, or K, tiling is not worth its overhead.
  int64_t maxTileSize = 32;     // The normal tile size, used when memory is not under pressure.
  int64_t conservativeTileSize = 16;  // Used instead when the schedule is near a memory budget.
  int64_t memoryBudgetBytes = 1 << 20;  // 1 MiB: an example default working-set budget.
};

class TilingDecision {
 public:
  explicit TilingDecision(TilingPolicy policy = {}) : policy_(policy) {}

  // `matmul` must be an OpKind::MatMul node with both operands at rank 2
  // (the whitepaper's C = A @ B reference computation, ARCH-010).
  // `graph` gives the operand types. `liveness` is optional; if given,
  // it can make the decision more conservative under memory pressure
  // (a real use of the peak-memory estimate, ARCH-009/LIVE-004).
  TilingPlan decide(const GraphNode& matmul, const TypedGraph& graph, const LivenessResult* liveness = nullptr) const;

 private:
  TilingPolicy policy_;
};

}  // namespace tensora
