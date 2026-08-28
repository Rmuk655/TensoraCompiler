// LOW-006..009: checks the numbers for the ops added to test the CNN
// and attention workloads end to end: transpose, flatten, softmax, and
// conv2d. Each op is proven on its own, through real JIT execution,
// before the tests combine them into bigger workloads
// (tests/lowering/test_cnn_e2e.cpp, tests/lowering/test_attention_e2e.cpp).
// This matches how matmul, linear, and relu were each proven on their
// own earlier in this phase.
#include <cmath>
#include <cstdio>
#include <functional>
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
#include "support/JitHarness.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;
using tensora::testing::constTensor;

namespace {


// Runs the standard MLIR/LLVM lowering pipeline (tests/support/JitHarness.h),
// JIT-executes "test_main" (zero args, scalar f32 return), and returns
// the result or NAN on failure. Uses the full JitPipelineOptions profile
// (ExpandStridedMetadata/LowerAffine for collapse_shape/subview-like
// ops, ConvertMathToLLVM for softmax's math.exp, and the C-runner-utils
// shared lib for flatten/collapse_shape/conv2d's memrefCopy calls)
// unconditionally -- harmless no-ops for graphs that don't use them.
float runAndGetScalar(ModuleOp module, MLIRContext& ctx, const char* label) {
  auto jitOpts = tensora::testing::JitPipelineOptions::full();
  if (!tensora::testing::runStandardLoweringPipeline(ctx, module, jitOpts, label)) return NAN;

  auto maybeEngine = tensora::testing::createJitEngine(module, jitOpts, label);
  if (!maybeEngine) return NAN;
  auto maybeFn = tensora::testing::lookupJitEntry<float (*)()>(**maybeEngine, "test_main", label);
  if (!maybeFn) return NAN;
  return (*maybeFn)();
}

// Every test below shares the same shape: build a TypedGraph, lower it,
// build a scalar harness that calls the lowered function with some
// constant-tensor arguments, extracts (or otherwise reduces) one
// element from the result, and runs it. lowerCallAndRun() factors out
// everything but the two genuinely per-test pieces: the arguments (via
// buildArgs) and how the single f32 to check is obtained from the call
// result (via buildResult). extractAt() covers the common "pick one
// element by index" case; LOW-009b uses a custom buildResult because it
// checks two elements at once (see its call site).
using ArgsBuilder = std::function<std::vector<MlirValue>(OpBuilder&, Location, Type)>;
using ResultBuilder = std::function<MlirValue(OpBuilder&, Location, MlirValue)>;

float lowerCallAndRun(tns::TypeContext& types, tns::TypedGraph graph, const std::string& fnName,
                      const ArgsBuilder& buildArgs, const ResultBuilder& buildResult, const char* label) {
  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));

  auto lowered = tensora::testing::lowerProgram(program);
  MLIRContext& ctx = *lowered.ctx;
  ModuleOp module = lowered.module;
  auto f32 = mlir::Float32Type::get(&ctx);
  OpBuilder hb = tensora::testing::appendScalarHarness(ctx, module, f32);
  Location loc = hb.getUnknownLoc();

  std::vector<MlirValue> args = buildArgs(hb, loc, f32);
  auto fn = module.lookupSymbol<func::FuncOp>(fnName);
  auto call = func::CallOp::create(hb, loc, fn, ValueRange(args));
  MlirValue elem = buildResult(hb, loc, call.getResult(0));
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  return runAndGetScalar(module, ctx, label);
}

ResultBuilder extractAt(std::vector<int64_t> indices) {
  return [indices](OpBuilder& hb, Location loc, MlirValue result) -> MlirValue {
    std::vector<MlirValue> idx;
    for (int64_t i : indices) idx.push_back(arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(i)));
    return tensor::ExtractOp::create(hb, loc, result, ValueRange(idx));
  };
}

// --- LOW-013: elementwise add with real tensor-tensor broadcasting ---
// x = [[1,2,3],[4,5,6]] (batch=2,feat=3); y = [10,20,30] (feat=3), a
// shorter-rank operand broadcast against x's leading axis (RM 3.8: a
// shape shorter in rank gets an implicit leading size-1 axis, named
// from the longer operand) -- not the tensor-scalar case every other
// elementwise test in this project already covers. x+y = [[11,22,33],
// [14,25,36]]; element [1][2] = 36.
bool testElementwiseBroadcast() {
  tns::TypeContext types;
  tns::GraphBuilder builder("ew_broadcast_test");
  const tns::TensorType* xT = types.tensor(
      {tns::Axis{"batch", tns::Dimension::makeConst(2)}, tns::Axis{"feat", tns::Dimension::makeConst(3)}});
  const tns::TensorType* yT = types.tensor({tns::Axis{"feat", tns::Dimension::makeConst(3)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::ValueId yVal = builder.addInput("y", yT, {});
  tns::ValueId zVal = builder.emit(tns::OpKind::Add, {xVal, yVal}, {}, xT, {}, "z");
  builder.setOutput(zVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "ew_broadcast_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        MlirValue x = constTensor(hb, loc, RankedTensorType::get({2, 3}, f32), {1, 2, 3, 4, 5, 6});
        MlirValue y = constTensor(hb, loc, RankedTensorType::get({3}, f32), {10, 20, 30});
        return {x, y};
      },
      extractAt({1, 2}), "LOW-013");

  if (result != 36.0f) {
    fprintf(stderr, "LOW-013 FAIL: expected 36.0, got %f\n", result);
    return false;
  }
  printf("LOW-013 PASS: ([[1,2,3],[4,5,6]] + [10,20,30])[1][2] = %f (expected 36.0)\n", result);
  return true;
}

// --- LOW-006: transpose ---
// x = [[1,2,3],[4,5,6]] (2x3); transpose(perm=[1,0]) -> (3x2)
// [[1,4],[2,5],[3,6]]; element [2][0] = 3.
bool testTranspose() {
  tns::TypeContext types;
  tns::GraphBuilder builder("transpose_test");
  const tns::TensorType* xT = types.tensor(
      {tns::Axis{"m", tns::Dimension::makeConst(2)}, tns::Axis{"n", tns::Dimension::makeConst(3)}});
  const tns::TensorType* yT = types.tensor(
      {tns::Axis{"n", tns::Dimension::makeConst(3)}, tns::Axis{"m", tns::Dimension::makeConst(2)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::NodeAttrs attrs;
  attrs.perm = {1, 0};
  tns::ValueId yVal = builder.emit(tns::OpKind::Transpose, {xVal}, attrs, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "transpose_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        return {constTensor(hb, loc, RankedTensorType::get({2, 3}, f32), {1, 2, 3, 4, 5, 6})};
      },
      extractAt({2, 0}), "LOW-006");

  if (result != 3.0f) {
    fprintf(stderr, "LOW-006 FAIL: expected 3.0, got %f\n", result);
    return false;
  }
  printf("LOW-006 PASS: transpose([[1,2,3],[4,5,6]], perm=[1,0])[2][0] = %f (expected 3.0)\n", result);
  return true;
}

// --- LOW-007: flatten ---
// x: (2,2,2) = [[[1,2],[3,4]],[[5,6],[7,8]]] -> flatten -> (2,4)
// [[1,2,3,4],[5,6,7,8]]; element [1][2] = 7.
bool testFlatten() {
  tns::TypeContext types;
  tns::GraphBuilder builder("flatten_test");
  const tns::TensorType* xT = types.tensor({tns::Axis{"b", tns::Dimension::makeConst(2)},
                                            tns::Axis{"h", tns::Dimension::makeConst(2)},
                                            tns::Axis{"w", tns::Dimension::makeConst(2)}});
  const tns::TensorType* yT =
      types.tensor({tns::Axis{"b", tns::Dimension::makeConst(2)}, tns::Axis{"flat", tns::Dimension::makeConst(4)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::ValueId yVal = builder.emit(tns::OpKind::Flatten, {xVal}, {}, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "flatten_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        return {constTensor(hb, loc, RankedTensorType::get({2, 2, 2}, f32), {1, 2, 3, 4, 5, 6, 7, 8})};
      },
      extractAt({1, 2}), "LOW-007");

  if (result != 7.0f) {
    fprintf(stderr, "LOW-007 FAIL: expected 7.0, got %f\n", result);
    return false;
  }
  printf("LOW-007 PASS: flatten((2,2,2))[1][2] = %f (expected 7.0)\n", result);
  return true;
}

// --- LOW-008: softmax ---
// x = [[1, 2, 3]] (batch=1, n=3) along axis "n" -- a non-trivial max
// (3, at index 2) so the max-shift step actually shifts something,
// unlike an all-zero input. exp(1-3)=0.135335, exp(2-3)=0.367879,
// exp(3-3)=1; sum=1.503214; softmax = [0.090031, 0.244728, 0.665241].
bool testSoftmax() {
  tns::TypeContext types;
  tns::GraphBuilder builder("softmax_test");
  const tns::TensorType* xT =
      types.tensor({tns::Axis{"batch", tns::Dimension::makeConst(1)}, tns::Axis{"n", tns::Dimension::makeConst(3)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::NodeAttrs attrs;
  attrs.axisName = "n";
  tns::ValueId yVal = builder.emit(tns::OpKind::Softmax, {xVal}, attrs, xT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "softmax_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        return {constTensor(hb, loc, RankedTensorType::get({1, 3}, f32), {1.0f, 2.0f, 3.0f})};
      },
      extractAt({0, 2}), "LOW-008");

  float expected = 0.665241f;
  if (std::fabs(result - expected) > 1e-4f) {
    fprintf(stderr, "LOW-008 FAIL: expected %f, got %f\n", expected, result);
    return false;
  }
  printf("LOW-008 PASS: softmax([1,2,3])[0][2] = %f (expected %f)\n", result, expected);
  return true;
}

// --- LOW-009: conv2d ---
// x: 1x1x3x3 (NCHW) = [[1,2,3],[4,5,6],[7,8,9]]; w: 1x1x2x2 (FCHW),
// all-ones kernel; bias = [10]; stride=1, padding=0 -> output 1x1x2x2,
// each element = sum of a 2x2 window + bias.
// window[0][0]=1+2+4+5=12 -> +10=22; window[0][1]=2+3+5+6=16 -> +10=26;
// window[1][0]=4+5+7+8=24 -> +10=34; window[1][1]=5+6+8+9=28 -> +10=38.
bool testConv2D() {
  tns::TypeContext types;
  tns::GraphBuilder builder("conv2d_test");
  const tns::TensorType* xT = types.tensor({tns::Axis{"n", tns::Dimension::makeConst(1)},
                                            tns::Axis{"c", tns::Dimension::makeConst(1)},
                                            tns::Axis{"h", tns::Dimension::makeConst(3)},
                                            tns::Axis{"w", tns::Dimension::makeConst(3)}});
  const tns::TensorType* wT = types.tensor({tns::Axis{"oc", tns::Dimension::makeConst(1)},
                                            tns::Axis{"ic", tns::Dimension::makeConst(1)},
                                            tns::Axis{"kh", tns::Dimension::makeConst(2)},
                                            tns::Axis{"kw", tns::Dimension::makeConst(2)}});
  const tns::TensorType* biasT = types.tensor({tns::Axis{"oc", tns::Dimension::makeConst(1)}});
  const tns::TensorType* yT = types.tensor({tns::Axis{"n", tns::Dimension::makeConst(1)},
                                            tns::Axis{"c", tns::Dimension::makeConst(1)},
                                            tns::Axis{"h", tns::Dimension::makeConst(2)},
                                            tns::Axis{"w", tns::Dimension::makeConst(2)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::ValueId wVal = builder.addWeight("w", wT, {});
  tns::ValueId biasVal = builder.addWeight("bias", biasT, {});
  tns::NodeAttrs attrs;
  attrs.stride = 1;
  attrs.padding = 0;
  tns::ValueId yVal = builder.emit(tns::OpKind::Conv2D, {xVal, wVal, biasVal}, attrs, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "conv2d_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 1, 3, 3}, f32), {1, 2, 3, 4, 5, 6, 7, 8, 9});
        MlirValue w = constTensor(hb, loc, RankedTensorType::get({1, 1, 2, 2}, f32), {1, 1, 1, 1});
        MlirValue bias = constTensor(hb, loc, RankedTensorType::get({1}, f32), {10.0f});
        return {x, w, bias};
      },
      extractAt({0, 0, 1, 1}), "LOW-009");

  if (result != 38.0f) {
    fprintf(stderr, "LOW-009 FAIL: expected 38.0, got %f\n", result);
    return false;
  }
  printf("LOW-009 PASS: conv2d(3x3 input, 2x2 all-ones kernel, bias=10)[0][0][1][1] = %f (expected 38.0)\n", result);
  return true;
}

// --- LOW-009b: conv2d with padding>0 ---
// x: 1x1x2x2, all ones; w: 1x1x1x1 = [2]; bias=[0]; stride=1,
// padding=1 -> padded input is 4x4 with a ring of zeros around the
// original 2x2 of ones; hout = (2+2*1-1)/1+1 = 4, so output is 1x1x4x4.
// A 1x1 kernel just scales every element by 2: padded-zero positions
// (e.g. [0][0]) -> 0*2=0; positions over the original data (e.g.
// [1][1], the top-left of the real 2x2 region once shifted by the
// padding) -> 1*2=2. This is the only path that exercises `tensor.pad`
// -- LOW-009 alone (padding=0) never emits it.
bool testConv2DPadding() {
  tns::TypeContext types;
  tns::GraphBuilder builder("conv2d_pad_test");
  const tns::TensorType* xT = types.tensor({tns::Axis{"n", tns::Dimension::makeConst(1)},
                                            tns::Axis{"c", tns::Dimension::makeConst(1)},
                                            tns::Axis{"h", tns::Dimension::makeConst(2)},
                                            tns::Axis{"w", tns::Dimension::makeConst(2)}});
  const tns::TensorType* wT = types.tensor({tns::Axis{"oc", tns::Dimension::makeConst(1)},
                                            tns::Axis{"ic", tns::Dimension::makeConst(1)},
                                            tns::Axis{"kh", tns::Dimension::makeConst(1)},
                                            tns::Axis{"kw", tns::Dimension::makeConst(1)}});
  const tns::TensorType* biasT = types.tensor({tns::Axis{"oc", tns::Dimension::makeConst(1)}});
  const tns::TensorType* yT = types.tensor({tns::Axis{"n", tns::Dimension::makeConst(1)},
                                            tns::Axis{"c", tns::Dimension::makeConst(1)},
                                            tns::Axis{"h", tns::Dimension::makeConst(4)},
                                            tns::Axis{"w", tns::Dimension::makeConst(4)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::ValueId wVal = builder.addWeight("w", wT, {});
  tns::ValueId biasVal = builder.addWeight("bias", biasT, {});
  tns::NodeAttrs attrs;
  attrs.stride = 1;
  attrs.padding = 1;
  tns::ValueId yVal = builder.emit(tns::OpKind::Conv2D, {xVal, wVal, biasVal}, attrs, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  // Both a padding-introduced-zero corner ([0][0]) and a real-data
  // position ([1][1]) are checked, so a bug that pads with the wrong
  // value (or doesn't shift the real data at all) would be caught.
  // Returns elemPad + elemReal so both are exercised through the single
  // scalar-returning harness convention used throughout this file.
  float result = lowerCallAndRun(
      types, std::move(graph), "conv2d_pad_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 1, 2, 2}, f32), {1, 1, 1, 1});
        MlirValue w = constTensor(hb, loc, RankedTensorType::get({1, 1, 1, 1}, f32), {2.0f});
        MlirValue bias = constTensor(hb, loc, RankedTensorType::get({1}, f32), {0.0f});
        return {x, w, bias};
      },
      [](OpBuilder& hb, Location loc, MlirValue result) -> MlirValue {
        MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
        MlirValue c1 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(1));
        MlirValue elemPad = tensor::ExtractOp::create(hb, loc, result, ValueRange{c0, c0, c0, c0});
        MlirValue elemReal = tensor::ExtractOp::create(hb, loc, result, ValueRange{c0, c0, c1, c1});
        return arith::AddFOp::create(hb, loc, elemPad, elemReal);
      },
      "LOW-009b");

  float expected = 0.0f + 2.0f;
  if (result != expected) {
    fprintf(stderr, "LOW-009b FAIL: expected %f (0 pad + 2 real), got %f\n", expected, result);
    return false;
  }
  printf("LOW-009b PASS: conv2d w/ padding=1: pad-corner[0][0]=0, real[1][1]=2, sum = %f (expected %f)\n", result,
         expected);
  return true;
}

// --- LOW-010: sum ---
// x = [[1,2,3],[4,5,6]] (2x3), reduce axis "n" (size 3) -> rank-1 (2,)
// = [1+2+3, 4+5+6] = [6, 15]; element [1] = 15.
bool testSum() {
  tns::TypeContext types;
  tns::GraphBuilder builder("sum_test");
  const tns::TensorType* xT = types.tensor(
      {tns::Axis{"m", tns::Dimension::makeConst(2)}, tns::Axis{"n", tns::Dimension::makeConst(3)}});
  const tns::TensorType* yT = types.tensor({tns::Axis{"m", tns::Dimension::makeConst(2)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::NodeAttrs attrs;
  attrs.axisName = "n";
  tns::ValueId yVal = builder.emit(tns::OpKind::Sum, {xVal}, attrs, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "sum_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        return {constTensor(hb, loc, RankedTensorType::get({2, 3}, f32), {1, 2, 3, 4, 5, 6})};
      },
      extractAt({1}), "LOW-010");

  if (result != 15.0f) {
    fprintf(stderr, "LOW-010 FAIL: expected 15.0, got %f\n", result);
    return false;
  }
  printf("LOW-010 PASS: sum([[1,2,3],[4,5,6]], axis=n)[1] = %f (expected 15.0)\n", result);
  return true;
}

// --- LOW-011: mean ---
// Same x as LOW-010, reduce axis "n" via mean -> [(1+2+3)/3, (4+5+6)/3]
// = [2, 5]; element [1] = 5.
bool testMean() {
  tns::TypeContext types;
  tns::GraphBuilder builder("mean_test");
  const tns::TensorType* xT = types.tensor(
      {tns::Axis{"m", tns::Dimension::makeConst(2)}, tns::Axis{"n", tns::Dimension::makeConst(3)}});
  const tns::TensorType* yT = types.tensor({tns::Axis{"m", tns::Dimension::makeConst(2)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::NodeAttrs attrs;
  attrs.axisName = "n";
  tns::ValueId yVal = builder.emit(tns::OpKind::Mean, {xVal}, attrs, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "mean_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        return {constTensor(hb, loc, RankedTensorType::get({2, 3}, f32), {1, 2, 3, 4, 5, 6})};
      },
      extractAt({1}), "LOW-011");

  if (result != 5.0f) {
    fprintf(stderr, "LOW-011 FAIL: expected 5.0, got %f\n", result);
    return false;
  }
  printf("LOW-011 PASS: mean([[1,2,3],[4,5,6]], axis=n)[1] = %f (expected 5.0)\n", result);
  return true;
}

// --- LOW-012: maxpool ---
// x: 1x1x4x4 (NCHW) = rows [1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16];
// size=2, stride=2 (non-overlapping windows) -> 1x1x2x2:
// [[max(1,2,5,6), max(3,4,7,8)], [max(9,10,13,14), max(11,12,15,16)]]
// = [[6,8],[14,16]]; element [0][0][1][1] = 16.
bool testMaxPool() {
  tns::TypeContext types;
  tns::GraphBuilder builder("maxpool_test");
  const tns::TensorType* xT = types.tensor({tns::Axis{"n", tns::Dimension::makeConst(1)},
                                            tns::Axis{"c", tns::Dimension::makeConst(1)},
                                            tns::Axis{"h", tns::Dimension::makeConst(4)},
                                            tns::Axis{"w", tns::Dimension::makeConst(4)}});
  const tns::TensorType* yT = types.tensor({tns::Axis{"n", tns::Dimension::makeConst(1)},
                                            tns::Axis{"c", tns::Dimension::makeConst(1)},
                                            tns::Axis{"h", tns::Dimension::makeConst(2)},
                                            tns::Axis{"w", tns::Dimension::makeConst(2)}});
  tns::ValueId xVal = builder.addInput("x", xT, {});
  tns::NodeAttrs attrs;
  attrs.poolSize = 2;
  attrs.stride = 2;
  tns::ValueId yVal = builder.emit(tns::OpKind::MaxPool, {xVal}, attrs, yT, {}, "y");
  builder.setOutput(yVal);
  tns::TypedGraph graph = builder.finish();

  float result = lowerCallAndRun(
      types, std::move(graph), "maxpool_test",
      [](OpBuilder& hb, Location loc, Type f32) -> std::vector<MlirValue> {
        return {constTensor(hb, loc, RankedTensorType::get({1, 1, 4, 4}, f32),
                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})};
      },
      extractAt({0, 0, 1, 1}), "LOW-012");

  if (result != 16.0f) {
    fprintf(stderr, "LOW-012 FAIL: expected 16.0, got %f\n", result);
    return false;
  }
  printf("LOW-012 PASS: maxpool(4x4 input, size=2, stride=2)[0][0][1][1] = %f (expected 16.0)\n", result);
  return true;
}

}  // namespace

int run_new_ops_e2e() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  bool ok = true;
  ok &= testElementwiseBroadcast();
  ok &= testTranspose();
  ok &= testFlatten();
  ok &= testSoftmax();
  ok &= testConv2D();
  ok &= testConv2DPadding();
  ok &= testSum();
  ok &= testMean();
  ok &= testMaxPool();
  return ok ? 0 : 1;
}
