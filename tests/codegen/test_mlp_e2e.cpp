// LOW-E2E-MLP: builds a small MLP-shaped TypedGraph by hand (linear ->
// relu -> linear, matching whitepaper 10.1's structure). The test lowers
// it through the real MlirLowerer, wraps the lowered func.func with a
// small harness that gives it chosen constant inputs and weights, runs
// the standard MLIR/LLVM pipeline, and runs it with JIT execution. It
// then checks the result against a hand-computed reference. This is the
// main proof that MlirLowerer's matmul/linear/relu/bias-fusion lowering
// gives the right numbers, not just the right structure.
#include <cstdio>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MlirLowerer.h"
#include "tensora/optimization/GraphPass.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

const tns::TensorType* buildMlpGraph(tns::TypeContext& types, tns::GraphBuilder& b, tns::ValueId& outValue) {
  using tns::Axis;
  using tns::Dimension;
  const tns::TensorType* xT =
      types.tensor({Axis{"batch", Dimension::makeConst(1)}, Axis{"in", Dimension::makeConst(2)}});
  const tns::TensorType* w1T =
      types.tensor({Axis{"in", Dimension::makeConst(2)}, Axis{"hidden", Dimension::makeConst(2)}});
  const tns::TensorType* b1T = types.tensor({Axis{"hidden", Dimension::makeConst(2)}});
  const tns::TensorType* hT =
      types.tensor({Axis{"batch", Dimension::makeConst(1)}, Axis{"hidden", Dimension::makeConst(2)}});
  const tns::TensorType* w2T =
      types.tensor({Axis{"hidden", Dimension::makeConst(2)}, Axis{"out", Dimension::makeConst(1)}});
  const tns::TensorType* b2T = types.tensor({Axis{"out", Dimension::makeConst(1)}});
  const tns::TensorType* yT =
      types.tensor({Axis{"batch", Dimension::makeConst(1)}, Axis{"out", Dimension::makeConst(1)}});

  tns::ValueId x = b.addInput("x", xT, {});
  tns::ValueId w1 = b.addWeight("W1", w1T, {});
  tns::ValueId b1 = b.addWeight("b1", b1T, {});
  tns::ValueId w2 = b.addWeight("W2", w2T, {});
  tns::ValueId b2 = b.addWeight("b2", b2T, {});

  tns::ValueId h = b.emit(tns::OpKind::Linear, {x, w1, b1}, {}, hT, {}, "h");
  tns::ValueId hRelu = b.emit(tns::OpKind::Relu, {h}, {}, hT, {}, "h_relu");
  tns::ValueId y = b.emit(tns::OpKind::Linear, {hRelu, w2, b2}, {}, yT, {}, "y");
  b.setOutput(y);
  outValue = y;
  return yT;
}

MlirValue constTensor(OpBuilder& b, Location loc, RankedTensorType type, ArrayRef<float> values) {
  auto attr = DenseElementsAttr::get(type, values);
  return arith::ConstantOp::create(b, loc, type, attr);
}

}  // namespace

int main() {
  // --- Build the TypedGraph (mirrors what SemanticAnalyzer would produce). ---
  tns::TypeContext types;
  tns::GraphBuilder builder("mlp");
  tns::ValueId outValue;
  buildMlpGraph(types, builder, outValue);
  tns::TypedGraph graph = builder.finish();

  tns::GraphVerifier verifier;
  if (!verifier.verify(graph).empty()) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: hand-built graph is not GraphVerifier-clean\n");
    return 1;
  }

  // --- Level-1 optimization (real team-authored passes run before lowering). ---
  tns::DiagnosticEngine diags;
  tns::PassContext passCtx{types, diags};
  tns::PassManager pm;
  pm.add(tns::createLinearBiasActivationFusionPass());
  pm.add(tns::createDeadOpEliminationPass());
  pm.runAll(graph, passCtx);

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));

  // --- MLIR lowering. ---
  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: lowered module does not verify\n");
    module.dump();
    return 1;
  }

  // --- Harness: a zero-arg function supplying chosen constants, calling
  // the lowered "mlp" function, and extracting a scalar result.
  // x=[[1,2]], W1=identity, b1=[0,0] -> h=[1,2]; relu(h)=[1,2] (unchanged,
  // already positive); W2=[[1],[1]], b2=[0] -> y = 1*1+2*1+0 = 3.
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto f32 = b.getF32Type();
  auto harnessType = b.getFunctionType({}, {f32});
  auto harness = func::FuncOp::create(b, loc, "test_main", harnessType);
  harness->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), b.getUnitAttr());
  Block* entry = harness.addEntryBlock();
  OpBuilder hb(entry, entry->begin());

  MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 2}, f32), {1.0f, 2.0f});
  MlirValue w1 = constTensor(hb, loc, RankedTensorType::get({2, 2}, f32), {1.0f, 0.0f, 0.0f, 1.0f});
  MlirValue b1 = constTensor(hb, loc, RankedTensorType::get({2}, f32), {0.0f, 0.0f});
  MlirValue w2 = constTensor(hb, loc, RankedTensorType::get({2, 1}, f32), {1.0f, 1.0f});
  MlirValue b2 = constTensor(hb, loc, RankedTensorType::get({1}, f32), {0.0f});

  auto mlpFunc = module.lookupSymbol<func::FuncOp>("mlp");
  if (!mlpFunc) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: lowered module has no 'mlp' function\n");
    return 1;
  }
  auto call = func::CallOp::create(hb, loc, mlpFunc, ValueRange{x, w1, b1, w2, b2});
  MlirValue yTensor = call.getResult(0);
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, yTensor, ValueRange{c0, c0});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: module with harness does not verify\n");
    module.dump();
    return 1;
  }

  // --- Standard MLIR/LLVM lowering pipeline. ---
  mlir::PassManager mlirPm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;  // needed once there's more than one func + a call between them
  mlirPm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  mlirPm.addPass(createConvertLinalgToLoopsPass());
  mlirPm.addPass(createSCFToControlFlowPass());
  mlirPm.addPass(createConvertControlFlowToLLVMPass());
  mlirPm.addPass(createArithToLLVMConversionPass());
  mlirPm.addPass(createFinalizeMemRefToLLVMConversionPass());
  mlirPm.addPass(createConvertFuncToLLVMPass());
  mlirPm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(mlirPm.run(module))) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto maybeEngine = ExecutionEngine::create(module);
  if (!maybeEngine) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: ExecutionEngine::create failed\n");
    return 1;
  }
  auto engine = std::move(*maybeEngine);

  auto maybeFn = engine->lookup("test_main");
  if (!maybeFn) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: could not look up 'test_main'\n");
    return 1;
  }
  auto fn = reinterpret_cast<float (*)()>(*maybeFn);
  float result = fn();

  if (result != 3.0f) {
    fprintf(stderr, "LOW-E2E-MLP FAIL: expected 3.0, got %f\n", result);
    return 1;
  }
  printf("LOW-E2E-MLP PASS: MlirLowerer(linear->relu->linear) native execution = %f (expected 3.0)\n", result);
  return 0;
}
