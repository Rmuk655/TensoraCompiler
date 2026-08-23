// RANK-001..004: checks the numbers for the rank-1/rank>=3 `linear` and
// rank>=2 (with and without broadcasting) `@` lowering added to close a
// gap between the language and the backend. RM 3.7/3.9 allow these
// ranks, but MlirLowerer used to handle only rank 2, and would build an
// invalid linalg.matmul for any other rank. Each case builds a small
// TypedGraph by hand, lowers it through the real MlirLowerer, runs it
// with JIT execution, and checks every output value against a
// hand-computed reference. This follows the same method as
// LOW-E2E-MLP, made to work for outputs of any rank with more than one
// value.
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

// Row-major unravel of `linearIdx` into `shape`'s index attrs.
std::vector<MlirValue> unravelIndex(OpBuilder& b, Location loc, ArrayRef<int64_t> shape, size_t linearIdx) {
  std::vector<int64_t> idx(shape.size());
  for (size_t i = shape.size(); i-- > 0;) {
    idx[i] = linearIdx % shape[i];
    linearIdx /= shape[i];
  }
  std::vector<MlirValue> result;
  for (int64_t v : idx) result.push_back(arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(v)));
  return result;
}

// Builds a one-node TypedGraph, lowers it, wraps it with a harness that
// supplies constant inputs and extracts every output element, JIT-runs it,
// and checks the results. Returns true on success (prints on failure).
bool runAndCheck(const char* caseId, tns::OpKind op, const tns::TensorType* xT, ArrayRef<float> xVals,
                  const tns::TensorType* wT, ArrayRef<float> wVals, const tns::TensorType* bT, ArrayRef<float> bVals,
                  const tns::TensorType* yT, ArrayRef<float> expected) {
  tns::TypeContext types;
  tns::GraphBuilder builder("g");
  tns::ValueId x = builder.addInput("x", xT, {});
  tns::ValueId w = builder.addWeight("w", wT, {});
  std::vector<tns::ValueId> ins = {x, w};
  if (bT != nullptr) ins.push_back(builder.addWeight("bias", bT, {}));
  tns::ValueId y = builder.emit(op, ins, {}, yT, {}, "y");
  builder.setOutput(y);
  tns::TypedGraph graph = builder.finish();

  tns::GraphVerifier verifier;
  if (!verifier.verify(graph).empty()) {
    fprintf(stderr, "%s FAIL: hand-built graph is not GraphVerifier-clean\n", caseId);
    return false;
  }

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));

  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "%s FAIL: lowered module does not verify\n", caseId);
    module.dump();
    return false;
  }

  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto f32 = b.getF32Type();
  size_t n = expected.size();
  std::vector<mlir::Type> resultTypes(n, f32);
  auto harnessType = b.getFunctionType({}, resultTypes);
  auto harness = func::FuncOp::create(b, loc, "test_main", harnessType);
  harness->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), b.getUnitAttr());
  Block* entry = harness.addEntryBlock();
  OpBuilder hb(entry, entry->begin());

  auto xShape = SmallVector<int64_t>(xT->rank());
  for (size_t i = 0; i < xT->rank(); ++i) xShape[i] = xT->axes()[i].size.constant;
  auto wShape = SmallVector<int64_t>(wT->rank());
  for (size_t i = 0; i < wT->rank(); ++i) wShape[i] = wT->axes()[i].size.constant;
  auto yShape = SmallVector<int64_t>(yT->rank());
  for (size_t i = 0; i < yT->rank(); ++i) yShape[i] = yT->axes()[i].size.constant;

  MlirValue xVal = constTensor(hb, loc, RankedTensorType::get(xShape, f32), xVals);
  MlirValue wVal = constTensor(hb, loc, RankedTensorType::get(wShape, f32), wVals);
  std::vector<MlirValue> callArgs = {xVal, wVal};
  if (bT != nullptr) {
    auto biasShape = SmallVector<int64_t>(bT->rank());
    for (size_t i = 0; i < bT->rank(); ++i) biasShape[i] = bT->axes()[i].size.constant;
    callArgs.push_back(constTensor(hb, loc, RankedTensorType::get(biasShape, f32), bVals));
  }

  auto gFunc = module.lookupSymbol<func::FuncOp>("g");
  if (!gFunc) {
    fprintf(stderr, "%s FAIL: lowered module has no 'g' function\n", caseId);
    return false;
  }
  auto call = func::CallOp::create(hb, loc, gFunc, ValueRange(callArgs));
  MlirValue yTensor = call.getResult(0);

  std::vector<MlirValue> elems;
  for (size_t i = 0; i < n; ++i) {
    auto idx = unravelIndex(hb, loc, yShape, i);
    elems.push_back(tensor::ExtractOp::create(hb, loc, yTensor, ValueRange(idx)));
  }
  func::ReturnOp::create(hb, loc, ValueRange(elems));

  if (failed(verify(module))) {
    fprintf(stderr, "%s FAIL: module with harness does not verify\n", caseId);
    module.dump();
    return false;
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
    fprintf(stderr, "%s FAIL: standard lowering pipeline failed\n", caseId);
    module.dump();
    return false;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto maybeEngine = ExecutionEngine::create(module);
  if (!maybeEngine) {
    fprintf(stderr, "%s FAIL: ExecutionEngine::create failed\n", caseId);
    return false;
  }
  auto engine = std::move(*maybeEngine);

  std::vector<float> results(n, 0.0f);
  std::vector<void*> argPtrs;
  for (size_t i = 0; i < n; ++i) argPtrs.push_back(&results[i]);
  if (engine->invokePacked("test_main", argPtrs)) {
    fprintf(stderr, "%s FAIL: invokePacked failed\n", caseId);
    return false;
  }

  bool ok = true;
  for (size_t i = 0; i < n; ++i) {
    if (results[i] != expected[i]) {
      fprintf(stderr, "%s FAIL: element[%zu] = %f, expected %f\n", caseId, i, results[i], expected[i]);
      ok = false;
    }
  }
  if (ok) printf("%s PASS\n", caseId);
  return ok;
}

}  // namespace

int run_rank_e2e() {
  using tns::Axis;
  using tns::Dimension;
  tns::TypeContext types;
  bool allOk = true;

  // RANK-001: linear, rank(x) == 1 (RM 3.9's rank>=1 case, no leading
  // batch axis at all). x=[1,2,3], W=[[1,0],[0,1],[1,1]], b=[10,20]
  // -> y = [1*1+2*0+3*1+10, 1*0+2*1+3*1+20] = [14, 25].
  {
    const tns::TensorType* xT = types.tensor({Axis{"k", Dimension::makeConst(3)}});
    const tns::TensorType* wT =
        types.tensor({Axis{"in", Dimension::makeConst(3)}, Axis{"out", Dimension::makeConst(2)}});
    const tns::TensorType* bT = types.tensor({Axis{"out", Dimension::makeConst(2)}});
    const tns::TensorType* yT = types.tensor({Axis{"out", Dimension::makeConst(2)}});
    allOk &= runAndCheck("RANK-001", tns::OpKind::Linear, xT, {1, 2, 3}, wT, {1, 0, 0, 1, 1, 1}, bT, {10, 20}, yT,
                         {14, 25});
  }

  // RANK-002: linear, rank(x) == 3 (batched, W shared across batch: RM
  // 3.9's rank>=3 case). W = identity(2), b = [100, 200] -> y[...,0] =
  // x[...,0] + 100, y[...,1] = x[...,1] + 200.
  {
    const tns::TensorType* xT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"seq", Dimension::makeConst(2)}, Axis{"k", Dimension::makeConst(2)}});
    const tns::TensorType* wT =
        types.tensor({Axis{"in", Dimension::makeConst(2)}, Axis{"out", Dimension::makeConst(2)}});
    const tns::TensorType* bT = types.tensor({Axis{"out", Dimension::makeConst(2)}});
    const tns::TensorType* yT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"seq", Dimension::makeConst(2)}, Axis{"out", Dimension::makeConst(2)}});
    allOk &= runAndCheck("RANK-002", tns::OpKind::Linear, xT, {1, 2, 3, 4, 5, 6, 7, 8}, wT, {1, 0, 0, 1}, bT,
                         {100, 200}, yT, {101, 202, 103, 204, 105, 206, 107, 208});
  }

  // RANK-003: matmul (@), rank 3, equal batch dims, no broadcasting (RM
  // 3.7's rank>=2 general case). A batch0 = I, batch1 = 2*I.
  {
    const tns::TensorType* aT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"m", Dimension::makeConst(2)}, Axis{"k", Dimension::makeConst(2)}});
    const tns::TensorType* bMatT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"k", Dimension::makeConst(2)}, Axis{"n", Dimension::makeConst(2)}});
    const tns::TensorType* cT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"m", Dimension::makeConst(2)}, Axis{"n", Dimension::makeConst(2)}});
    allOk &= runAndCheck("RANK-003", tns::OpKind::MatMul, aT, {1, 0, 0, 1, 2, 0, 0, 2}, bMatT,
                         {5, 6, 7, 8, 9, 10, 11, 12}, nullptr, {}, cT, {5, 6, 7, 8, 18, 20, 22, 24});
  }

  // RANK-004: matmul (@), rank 3, A's batch dim is 1 and broadcasts
  // against B's batch dim of 2 (RM 3.7's broadcastable-leading-axes case).
  {
    const tns::TensorType* aT = types.tensor(
        {Axis{"batch", Dimension::makeConst(1)}, Axis{"m", Dimension::makeConst(2)}, Axis{"k", Dimension::makeConst(2)}});
    const tns::TensorType* bMatT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"k", Dimension::makeConst(2)}, Axis{"n", Dimension::makeConst(2)}});
    const tns::TensorType* cT = types.tensor(
        {Axis{"batch", Dimension::makeConst(2)}, Axis{"m", Dimension::makeConst(2)}, Axis{"n", Dimension::makeConst(2)}});
    allOk &= runAndCheck("RANK-004", tns::OpKind::MatMul, aT, {1, 2, 3, 4}, bMatT, {1, 0, 0, 1, 0, 1, 1, 0}, nullptr,
                         {}, cT, {1, 2, 3, 4, 2, 1, 4, 3});
  }

  return allOk ? 0 : 1;
}
