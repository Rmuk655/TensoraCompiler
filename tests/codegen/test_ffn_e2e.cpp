// E2E-003: a Transformer feed-forward sublayer (position-wise
// linear -> relu -> linear, applied the same way to every sequence
// position through one matmul). This is the core FFN block, without
// the residual connection or layernorm, since neither is a lowered op
// yet. This test uses the same MlirLowerer::lowerLinear/lowerRelu path
// already proven by LOW-E2E-MLP, at FFN-shaped dimensions (seq_len x
// d_model -> hidden -> seq_len x d_model). This shows that this
// workload needed no new lowering work.
//
// x: 2x4 (seq_len=2, d_model=4) = [[1,2,3,4],[5,6,7,8]]
// W1: 4x8, uniform 0.1; b1 = 0 -> h[i] = 0.1 * sum(x_row_i), broadcast
// across all 8 hidden units: h = [[1.0]*8, [2.6]*8] (relu: unchanged,
// both positive)
// W2: 8x4, uniform 0.1; b2 = 0 -> out[i] = 0.1 * sum(h_row_i) = 0.1 *
// (h_row_i * 8): out = [[0.8]*4, [2.08]*4]
#include <cmath>
#include <cstdio>
#include <vector>

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

MlirValue constTensor(OpBuilder& b, Location loc, RankedTensorType type, ArrayRef<float> values) {
  auto attr = DenseElementsAttr::get(type, values);
  return arith::ConstantOp::create(b, loc, type, attr);
}

tns::TypedGraph buildFfnGraph(tns::TypeContext& types) {
  tns::GraphBuilder builder("ffn");
  using tns::Axis;
  using tns::Dimension;
  const tns::TensorType* xT =
      types.tensor({Axis{"seq", Dimension::makeConst(2)}, Axis{"d_model", Dimension::makeConst(4)}});
  const tns::TensorType* w1T =
      types.tensor({Axis{"d_model", Dimension::makeConst(4)}, Axis{"hidden", Dimension::makeConst(8)}});
  const tns::TensorType* b1T = types.tensor({Axis{"hidden", Dimension::makeConst(8)}});
  const tns::TensorType* hT =
      types.tensor({Axis{"seq", Dimension::makeConst(2)}, Axis{"hidden", Dimension::makeConst(8)}});
  const tns::TensorType* w2T =
      types.tensor({Axis{"hidden", Dimension::makeConst(8)}, Axis{"d_model", Dimension::makeConst(4)}});
  const tns::TensorType* b2T = types.tensor({Axis{"d_model", Dimension::makeConst(4)}});
  const tns::TensorType* yT =
      types.tensor({Axis{"seq", Dimension::makeConst(2)}, Axis{"d_model", Dimension::makeConst(4)}});

  tns::ValueId x = builder.addInput("x", xT, {});
  tns::ValueId w1 = builder.addWeight("W1", w1T, {});
  tns::ValueId b1 = builder.addWeight("b1", b1T, {});
  tns::ValueId w2 = builder.addWeight("W2", w2T, {});
  tns::ValueId b2 = builder.addWeight("b2", b2T, {});

  tns::ValueId h = builder.emit(tns::OpKind::Linear, {x, w1, b1}, {}, hT, {}, "h");
  tns::ValueId hRelu = builder.emit(tns::OpKind::Relu, {h}, {}, hT, {}, "h_relu");
  tns::ValueId y = builder.emit(tns::OpKind::Linear, {hRelu, w2, b2}, {}, yT, {}, "y");
  builder.setOutput(y);
  return builder.finish();
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  tns::TypeContext types;
  tns::TypedGraph graph = buildFfnGraph(types);

  tns::GraphVerifier verifier;
  if (!verifier.verify(graph).empty()) {
    fprintf(stderr, "E2E-003 FAIL: hand-built FFN graph is not GraphVerifier-clean\n");
    return 1;
  }

  tns::DiagnosticEngine diags;
  tns::PassContext passCtx{types, diags};
  tns::PassManager passManager;
  passManager.add(tns::createLinearBiasActivationFusionPass());
  passManager.add(tns::createDeadOpEliminationPass());
  passManager.runAll(graph, passCtx);

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));

  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "E2E-003 FAIL: lowered FFN module does not verify\n");
    module.dump();
    return 1;
  }

  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto f32 = b.getF32Type();
  auto harness = func::FuncOp::create(b, loc, "test_main", b.getFunctionType({}, {f32}));
  Block* entry = harness.addEntryBlock();
  OpBuilder hb(entry, entry->begin());

  MlirValue x = constTensor(hb, loc, RankedTensorType::get({2, 4}, f32), {1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<float> w1Data(4 * 8, 0.1f);
  std::vector<float> b1Data(8, 0.0f);
  std::vector<float> w2Data(8 * 4, 0.1f);
  std::vector<float> b2Data(4, 0.0f);
  MlirValue w1 = constTensor(hb, loc, RankedTensorType::get({4, 8}, f32), w1Data);
  MlirValue b1 = constTensor(hb, loc, RankedTensorType::get({8}, f32), b1Data);
  MlirValue w2 = constTensor(hb, loc, RankedTensorType::get({8, 4}, f32), w2Data);
  MlirValue b2 = constTensor(hb, loc, RankedTensorType::get({4}, f32), b2Data);

  auto fn = module.lookupSymbol<func::FuncOp>("ffn");
  auto call = func::CallOp::create(hb, loc, fn, ValueRange{x, w1, b1, w2, b2});
  MlirValue c1 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(1));
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, call.getResult(0), ValueRange{c1, c0});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "E2E-003 FAIL: module with harness does not verify\n");
    module.dump();
    return 1;
  }

  mlir::PassManager mlirPm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  mlirPm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  mlirPm.addPass(createConvertLinalgToLoopsPass());
  mlirPm.addPass(createSCFToControlFlowPass());
  mlirPm.addPass(createConvertControlFlowToLLVMPass());
  mlirPm.addPass(createArithToLLVMConversionPass());
  mlirPm.addPass(createFinalizeMemRefToLLVMConversionPass());
  mlirPm.addPass(createConvertFuncToLLVMPass());
  mlirPm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(mlirPm.run(module))) {
    fprintf(stderr, "E2E-003 FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  auto maybeEngine = ExecutionEngine::create(module);
  if (!maybeEngine) {
    fprintf(stderr, "E2E-003 FAIL: ExecutionEngine::create failed\n");
    return 1;
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("test_main");
  if (!maybeFn) {
    fprintf(stderr, "E2E-003 FAIL: could not look up 'test_main'\n");
    return 1;
  }
  auto testMain = reinterpret_cast<float (*)()>(*maybeFn);
  float result = testMain();

  if (std::fabs(result - 2.08f) > 1e-4f) {
    fprintf(stderr, "E2E-003 FAIL: expected 2.08, got %f\n", result);
    return 1;
  }
  printf("E2E-003 PASS: FFN(linear->relu->linear, seq_len=2, d_model=4, hidden=8)[1][0] = %f (expected 2.08)\n",
         result);
  return 0;
}
