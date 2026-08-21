#pragma once
#include "mlir/IR/MLIRContext.h"

// Registers the shared MLIR dialects and bufferization interfaces
// (INTEG-003: LLVM/MLIR 21). MlirLowerer and the backend/execution path
// both use this function. Every tool in lowering/ builds an MLIRContext
// the same way.
namespace tensora {

mlir::DialectRegistry makeTensoraDialectRegistry();

}  // namespace tensora
