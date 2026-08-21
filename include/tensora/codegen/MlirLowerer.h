#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "tensora/codegen/LoopTiling.h"
#include "tensora/shared/Ir.h"

// Lowers a Typed Computation Graph to MLIR (ARCHITECTURE.md §7.1;
// INTERFACES.md §5). Each TypedGraph becomes a func.func, using the
// func/arith/tensor/linalg dialects. Axis names are erased here
// (ARCH-005): a concrete dim becomes a static tensor dimension, and a
// symbolic dim becomes dynamic (`?`). This class reads the verified
// TypedGraph as-is. It never re-derives a shape or re-checks an
// operator rule that semantic analysis already checked.
namespace tensora {

class MlirLowerer {
 public:
  // `tilingPolicy` must match the TilingPolicy given to
  // createMatmulTilingPass (MatmulTilingPass.h) for the same build.
  // This is what makes TilingDecision the single source of the tile
  // sizes MatmulTilingPass applies, instead of the two computing their
  // own policy that only happens to share default values.
  explicit MlirLowerer(mlir::MLIRContext& ctx, TilingPolicy tilingPolicy = {})
      : ctx_(ctx), tilingDecision_(tilingPolicy) {}

  // One mlir::ModuleOp containing one func.func per graph in `program`,
  // each marked for JIT/native invocation (llvm.emit_c_interface).
  mlir::ModuleOp lower(const TypedProgram& program);

 private:
  mlir::MLIRContext& ctx_;
  TilingDecision tilingDecision_;
};

}  // namespace tensora
