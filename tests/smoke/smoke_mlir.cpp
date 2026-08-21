// SMOKE-MLIR (docs/testing/08_TEST_PLAN.md §2b). This test proves the
// pinned MLIR/LLVM link (INTEG-003) works end to end: build a small
// func.func + linalg.matmul module, lower it through the standard
// MLIR/LLVM pipeline, JIT it, and check the number it returns. This is
// the highest-risk external dependency in the whole project. It must
// pass before the team builds anything in lowering/ on top of it.
#include <cstdio>
#include <cstdlib>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace {

// Builds: func.func @main() -> f32 { %A @ %B, extract [0,0] as f32 }
// A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A@B = [[19,22],[43,50]];
// element [0,0] = 19.
ModuleOp buildMatmulModule(MLIRContext& ctx, OpBuilder& b, Location loc) {
  ModuleOp module = ModuleOp::create(b, loc);
  b.setInsertionPointToEnd(module.getBody());

  auto f32 = b.getF32Type();
  auto aType = RankedTensorType::get({2, 2}, f32);
  auto bType = RankedTensorType::get({2, 2}, f32);
  auto cType = RankedTensorType::get({2, 2}, f32);

  auto funcType = b.getFunctionType({}, {f32});
  auto func = func::FuncOp::create(b, loc, "main", funcType);
  // ExecutionEngine::invoke calls "_mlir_ciface_<name>", which FuncToLLVM
  // only emits for functions marked with this attribute.
  func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), UnitAttr::get(&ctx));
  Block* entry = func.addEntryBlock();
  b.setInsertionPointToStart(entry);

  auto aAttr = DenseElementsAttr::get(aType, ArrayRef<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto bAttr = DenseElementsAttr::get(bType, ArrayRef<float>{5.0f, 6.0f, 7.0f, 8.0f});
  Value aVal = arith::ConstantOp::create(b, loc, aType, aAttr);
  Value bVal = arith::ConstantOp::create(b, loc, bType, bAttr);
  Value empty = tensor::EmptyOp::create(b, loc, cType, ValueRange{});
  // linalg.matmul accumulates into `outs` (computes outs += A@B); an
  // uninitialized tensor.empty() as the accumulator reads garbage memory.
  Value zeroF32 = arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(0.0f));
  Value zeroed = linalg::FillOp::create(b, loc, TypeRange{cType}, ValueRange{zeroF32}, ValueRange{empty}).getResult(0);

  auto matmul = linalg::MatmulOp::create(b, loc, TypeRange{cType}, ValueRange{aVal, bVal}, ValueRange{zeroed});
  Value result = matmul.getResult(0);

  // ConstantIndexOp only exposes build(), not the newer create() static
  // factory used everywhere else here; go through the base ConstantOp
  // with an explicit index attribute instead.
  Value c0 = arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value elem = tensor::ExtractOp::create(b, loc, result, ValueRange{c0, c0});
  func::ReturnOp::create(b, loc, ValueRange{elem});

  return module;
}

}  // namespace

int main() {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
                  LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();
  ModuleOp module = buildMatmulModule(ctx, builder, loc);

  if (failed(verify(module))) {
    fprintf(stderr, "SMOKE-MLIR FAIL: initial module does not verify\n");
    module.dump();
    return 1;
  }

  PassManager pm(&ctx);
  pm.addPass(bufferization::createOneShotBufferizePass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module))) {
    fprintf(stderr, "SMOKE-MLIR FAIL: lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto maybeEngine = ExecutionEngine::create(module);
  if (!maybeEngine) {
    fprintf(stderr, "SMOKE-MLIR FAIL: ExecutionEngine::create failed\n");
    return 1;
  }
  std::unique_ptr<ExecutionEngine> engine = std::move(*maybeEngine);

  // ExecutionEngine::invoke()'s Result<T>/ciface convention is designed
  // around out-pointer results (memrefs); a zero-arg function returning a
  // plain scalar uses the ordinary LLVM calling convention instead
  // (return in a register), so go through the raw lowered symbol with
  // its real signature rather than the ciface wrapper.
  auto maybeFn = engine->lookup("main");
  if (!maybeFn) {
    fprintf(stderr, "SMOKE-MLIR FAIL: could not look up 'main'\n");
    return 1;
  }
  auto mainFn = reinterpret_cast<float (*)()>(*maybeFn);
  float result = mainFn();

  if (result != 19.0f) {
    fprintf(stderr, "SMOKE-MLIR FAIL: expected 19.0, got %f\n", result);
    return 1;
  }

  printf("SMOKE-MLIR PASS: matmul([[1,2],[3,4]], [[5,6],[7,8]])[0][0] = %f (native JIT execution)\n", result);
  return 0;
}
