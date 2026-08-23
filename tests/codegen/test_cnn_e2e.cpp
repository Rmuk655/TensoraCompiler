// E2E-002: a small CNN classifier (conv2d -> relu -> flatten -> linear
// -> softmax). This combines LOW-001/003/007/008/009 into one real
// workload. The test lowers it through MlirLowerer, runs it with JIT
// execution, and checks the result against a hand-computed reference.
// This is the CNN counterpart to LOW-E2E-MLP.
//
// Shapes: x: 1x1x4x4 (NCHW) -> conv2d(1x1x2x2 kernel, stride=2,
// padding=0, bias=0) -> 1x1x2x2 -> relu (no-op, all positive) ->
// flatten -> 1x4 -> linear(W: 4x2, b: 2) -> 1x2 -> softmax(axis
// "classes") -> 1x2.
//
// x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]; kernel = all
// ones (2x2); stride=2 means the two 2x2 windows don't overlap:
// conv = [[1+2+5+6, 3+4+7+8], [9+10+13+14, 11+12+15+16]]
//      = [[14, 22], [46, 54]]
// flatten (row-major over c,h,w) = [14, 22, 46, 54]
// W (each input feature routes to exactly one of 2 classes, scaled by
// 0.01): row0=[0.01,0], row1=[0,0.01], row2=[0.01,0], row3=[0,0.01]
// linear = [0.01*(14+46), 0.01*(22+54)] = [0.60, 0.76]
// softmax([0.60, 0.76]): e^0.60=1.822119, e^0.76=2.138264,
// sum=3.960382 -> [0.460098, 0.539902]
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
#include "tensora/optimization/GraphPass.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

MlirValue constTensor(OpBuilder& b, Location loc, RankedTensorType type, ArrayRef<float> values) {
  auto attr = DenseElementsAttr::get(type, values);
  return arith::ConstantOp::create(b, loc, type, attr);
}

tns::TypedProgram buildCnnGraph(tns::TypeContext& types) {
  tns::GraphBuilder builder("cnn");
  using tns::Axis;
  using tns::Dimension;

  const tns::TensorType* xT = types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"c", Dimension::makeConst(1)},
                                            Axis{"h", Dimension::makeConst(4)}, Axis{"w", Dimension::makeConst(4)}});
  const tns::TensorType* convWT = types.tensor({Axis{"oc", Dimension::makeConst(1)}, Axis{"ic", Dimension::makeConst(1)},
                                                Axis{"kh", Dimension::makeConst(2)}, Axis{"kw", Dimension::makeConst(2)}});
  const tns::TensorType* convBT = types.tensor({Axis{"oc", Dimension::makeConst(1)}});
  const tns::TensorType* convOutT =
      types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"c", Dimension::makeConst(1)},
                     Axis{"h", Dimension::makeConst(2)}, Axis{"w", Dimension::makeConst(2)}});
  const tns::TensorType* flatT =
      types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"flat", Dimension::makeConst(4)}});
  const tns::TensorType* fcWT =
      types.tensor({Axis{"flat", Dimension::makeConst(4)}, Axis{"classes", Dimension::makeConst(2)}});
  const tns::TensorType* fcBT = types.tensor({Axis{"classes", Dimension::makeConst(2)}});
  const tns::TensorType* logitsT =
      types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"classes", Dimension::makeConst(2)}});

  tns::ValueId x = builder.addInput("x", xT, {});
  tns::ValueId convW = builder.addWeight("convW", convWT, {});
  tns::ValueId convB = builder.addWeight("convB", convBT, {});
  tns::ValueId fcW = builder.addWeight("fcW", fcWT, {});
  tns::ValueId fcB = builder.addWeight("fcB", fcBT, {});

  tns::NodeAttrs convAttrs;
  convAttrs.stride = 2;
  convAttrs.padding = 0;
  tns::ValueId conv = builder.emit(tns::OpKind::Conv2D, {x, convW, convB}, convAttrs, convOutT, {}, "conv");
  tns::ValueId reluOut = builder.emit(tns::OpKind::Relu, {conv}, {}, convOutT, {}, "relu");
  tns::ValueId flat = builder.emit(tns::OpKind::Flatten, {reluOut}, {}, flatT, {}, "flat");
  tns::ValueId logits = builder.emit(tns::OpKind::Linear, {flat, fcW, fcB}, {}, logitsT, {}, "logits");
  tns::NodeAttrs smAttrs;
  smAttrs.axisName = "classes";
  tns::ValueId probs = builder.emit(tns::OpKind::Softmax, {logits}, smAttrs, logitsT, {}, "probs");
  builder.setOutput(probs);
  tns::TypedGraph graph = builder.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

}  // namespace

int run_cnn_e2e() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  tns::TypeContext types;
  tns::TypedProgram program = buildCnnGraph(types);

  tns::GraphVerifier verifier;
  if (!verifier.verify(program.graphs().front()).empty()) {
    fprintf(stderr, "E2E-002 FAIL: hand-built CNN graph is not GraphVerifier-clean\n");
    return 1;
  }

  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "E2E-002 FAIL: lowered CNN module does not verify\n");
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

  MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 1, 4, 4}, f32),
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  MlirValue convW = constTensor(hb, loc, RankedTensorType::get({1, 1, 2, 2}, f32), {1, 1, 1, 1});
  MlirValue convB = constTensor(hb, loc, RankedTensorType::get({1}, f32), {0.0f});
  MlirValue fcW = constTensor(hb, loc, RankedTensorType::get({4, 2}, f32),
                              {0.01f, 0.0f, 0.0f, 0.01f, 0.01f, 0.0f, 0.0f, 0.01f});
  MlirValue fcB = constTensor(hb, loc, RankedTensorType::get({2}, f32), {0.0f, 0.0f});

  auto fn = module.lookupSymbol<func::FuncOp>("cnn");
  auto call = func::CallOp::create(hb, loc, fn, ValueRange{x, convW, convB, fcW, fcB});
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue c1 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(1));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, call.getResult(0), ValueRange{c0, c1});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "E2E-002 FAIL: module with harness does not verify\n");
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
    fprintf(stderr, "E2E-002 FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "E2E-002 FAIL: ExecutionEngine::create failed\n");
    return 1;
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("test_main");
  if (!maybeFn) {
    fprintf(stderr, "E2E-002 FAIL: could not look up 'test_main'\n");
    return 1;
  }
  auto testMain = reinterpret_cast<float (*)()>(*maybeFn);
  float result = testMain();

  float expected = 0.539902f;
  if (std::fabs(result - expected) > 1e-4f) {
    fprintf(stderr, "E2E-002 FAIL: expected %f, got %f\n", expected, result);
    return 1;
  }
  printf(
      "E2E-002 PASS: CNN(conv2d->relu->flatten->linear->softmax) class-1 probability = %f (expected %f)\n", result,
      expected);
  return 0;
}
