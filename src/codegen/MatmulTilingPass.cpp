// This file implements the Level-2 tiling pass.
// The pass applies the TilingDecision (LoopTiling.cpp) to each
// linalg.matmul op in the lowered MLIR, using MLIR's own tiling
// interface to do the mechanical rewrite.
#include "tensora/codegen/MatmulTilingPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace tensora {

using namespace mlir;

namespace {

// See MatmulTilingPass.h for the full pass writeup (name, invariants,
// legality, decision, failure behavior).
class MatmulTilingPass : public PassWrapper<MatmulTilingPass, OperationPass<func::FuncOp>> {
 public:
  MatmulTilingPass() = default;
  // The tile sizes come from the "tensora.tile_sizes" attribute, so the policy is not needed here;
  // the parameter only keeps the factory signature the same as MlirLowerer's.
  explicit MatmulTilingPass(TilingPolicy) {}
  MatmulTilingPass(const MatmulTilingPass& other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "tensora-tile-matmul"; }
  StringRef getDescription() const override {
    return "Tensora-authored shape-guided tiling of linalg.matmul (Level-2, ARCH-009)";
  }

  // Applies exactly the tile sizes TilingDecision::decide()
  // (LoopTiling.cpp) already chose at the TCG level. MlirLowerer stamps
  // those sizes onto the op as "tensora.tile_sizes". This pass does not
  // compute its own policy from raw MLIR shapes.
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<linalg::MatmulOp> candidates;
    func.walk([&](linalg::MatmulOp op) { candidates.push_back(op); });

    for (linalg::MatmulOp op : candidates) {
      auto tileSizesAttr = op->getAttrOfType<DenseI64ArrayAttr>("tensora.tile_sizes");
      if (!tileSizesAttr || tileSizesAttr.size() != 3) continue;  // TilingDecision said not to tile this one

      ArrayRef<int64_t> sizes = tileSizesAttr.asArrayRef();
      IRRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      scf::SCFTilingOptions options;
      options.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), SmallVector<int64_t>(sizes)));

      auto tilingOp = cast<TilingInterface>(op.getOperation());
      FailureOr<scf::SCFTilingResult> result = scf::tileUsingSCF(rewriter, tilingOp, options);
      if (failed(result)) {  // leave unchanged: not a pass failure (see header's "Failure behavior")
        op.emitRemark("tensora-tile-matmul: tiling skipped, tileUsingSCF failed");
        continue;
      }
      rewriter.replaceOp(op, result->replacements);
      ++numTiled;
    }
  }

 private:
  Statistic numTiled{this, "num-tiled", "Number of linalg.matmul ops tiled"};
};

}  // namespace

std::unique_ptr<mlir::Pass> createMatmulTilingPass(TilingPolicy policy) {
  return std::make_unique<MatmulTilingPass>(policy);
}

}  // namespace tensora
