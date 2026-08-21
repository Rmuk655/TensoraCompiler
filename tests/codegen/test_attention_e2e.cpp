// E2E-004: single-head scaled dot-product attention (matmul -> transpose
// -> scale -> softmax -> matmul). This combines LOW-001/006/008 into one
// real workload. The test lowers it through MlirLowerer, runs it with
// JIT execution, and checks the result against a hand-computed
// reference.
//
// Q, K: 2x2 identity; V = [[1,2],[3,4]]; scale = 1/sqrt(2) (d_k=2).
// scores = Q @ K^T * scale = identity * scale = diag(0.707107)
// softmax(scores, axis=seq_k) row0 = [0.669761, 0.330238] (row1 mirrors
// by symmetry: [0.330238, 0.669761])
// out = attn @ V:
//   out[0] = 0.669761*[1,2] + 0.330238*[3,4] = [1.660477, 2.660477]
//   out[1] = 0.330238*[1,2] + 0.669761*[3,4] = [2.339523, 3.339523]
// (exact values from a Python double-precision reference; this is not a
// NumPy dependency in the C++ suite, just how the expected constants
// below were derived.)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

MlirValue constTensor(OpBuilder& b, Location loc, RankedTensorType type, ArrayRef<float> values) {
  auto attr = DenseElementsAttr::get(type, values);
  return arith::ConstantOp::create(b, loc, type, attr);
}

tns::TypedProgram buildAttentionGraph(tns::TypeContext& types) {
  tns::GraphBuilder builder("attention");
  using tns::Axis;
  using tns::Dimension;

  const tns::TensorType* qT =
      types.tensor({Axis{"seq_q", Dimension::makeConst(2)}, Axis{"d_k", Dimension::makeConst(2)}});
  const tns::TensorType* kT =
      types.tensor({Axis{"seq_k", Dimension::makeConst(2)}, Axis{"d_k", Dimension::makeConst(2)}});
  const tns::TensorType* ktT =
      types.tensor({Axis{"d_k", Dimension::makeConst(2)}, Axis{"seq_k", Dimension::makeConst(2)}});
  const tns::TensorType* scoresT =
      types.tensor({Axis{"seq_q", Dimension::makeConst(2)}, Axis{"seq_k", Dimension::makeConst(2)}});
  const tns::TensorType* vT =
      types.tensor({Axis{"seq_k", Dimension::makeConst(2)}, Axis{"d_v", Dimension::makeConst(2)}});
  const tns::TensorType* outT =
      types.tensor({Axis{"seq_q", Dimension::makeConst(2)}, Axis{"d_v", Dimension::makeConst(2)}});

  tns::ValueId q = builder.addInput("q", qT, {});
  tns::ValueId k = builder.addInput("k", kT, {});
  tns::ValueId v = builder.addInput("v", vT, {});

  tns::NodeAttrs transAttrs;
  transAttrs.perm = {1, 0};
  tns::ValueId kt = builder.emit(tns::OpKind::Transpose, {k}, transAttrs, ktT, {}, "kt");
  tns::ValueId scores = builder.emit(tns::OpKind::MatMul, {q, kt}, {}, scoresT, {}, "scores");

  tns::NodeAttrs scaleAttrs;
  scaleAttrs.scalar = 1.0 / std::sqrt(2.0);
  tns::ValueId scaled = builder.emit(tns::OpKind::Mul, {scores}, scaleAttrs, scoresT, {}, "scaled");

  tns::NodeAttrs smAttrs;
  smAttrs.axisName = "seq_k";
  tns::ValueId probs = builder.emit(tns::OpKind::Softmax, {scaled}, smAttrs, scoresT, {}, "probs");
  tns::ValueId out = builder.emit(tns::OpKind::MatMul, {probs, v}, {}, outT, {}, "out");
  builder.setOutput(out);
  tns::TypedGraph graph = builder.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  tns::TypeContext types;
  tns::TypedProgram program = buildAttentionGraph(types);

  tns::GraphVerifier verifier;
  if (!verifier.verify(program.graphs().front()).empty()) {
    fprintf(stderr, "E2E-004 FAIL: hand-built attention graph is not GraphVerifier-clean\n");
    return 1;
  }

  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "E2E-004 FAIL: lowered attention module does not verify\n");
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

  MlirValue q = constTensor(hb, loc, RankedTensorType::get({2, 2}, f32), {1, 0, 0, 1});
  MlirValue k = constTensor(hb, loc, RankedTensorType::get({2, 2}, f32), {1, 0, 0, 1});
  MlirValue v = constTensor(hb, loc, RankedTensorType::get({2, 2}, f32), {1, 2, 3, 4});

  auto fn = module.lookupSymbol<func::FuncOp>("attention");
  auto call = func::CallOp::create(hb, loc, fn, ValueRange{q, k, v});
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, call.getResult(0), ValueRange{c0, c0});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "E2E-004 FAIL: module with harness does not verify\n");
    module.dump();
    return 1;
  }

  mlir::PassManager pm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.functionBoundaryTypeConversion = bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    fprintf(stderr, "E2E-004 FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "E2E-004 FAIL: ExecutionEngine::create failed\n");
    return 1;
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("test_main");
  if (!maybeFn) {
    fprintf(stderr, "E2E-004 FAIL: could not look up 'test_main'\n");
    return 1;
  }
  auto testMain = reinterpret_cast<float (*)()>(*maybeFn);
  float result = testMain();

  float expected = 1.660477f;
  if (std::fabs(result - expected) > 1e-4f) {
    fprintf(stderr, "E2E-004 FAIL: expected %f, got %f\n", expected, result);
    return 1;
  }
  printf("E2E-004 PASS: attention(Q=K=I, V=[[1,2],[3,4]])[0][0] = %f (expected %f)\n", result, expected);
  return 0;
}
