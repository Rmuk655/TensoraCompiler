// Benchmarks MatmulTilingPass across a few candidate tile sizes on the
// same 1024x1024x1024 matmul workload test_tiling_e2e.cpp uses, plus an
// untiled baseline, and prints the measured wall times. This is a
// stand-alone reporting tool, not a CTest case: LOOP-002/003/004
// already check tiling's structure and numerical correctness; this
// tool exists only to produce real, machine-measured timing evidence
// for the tile-size choice, instead of an assumed or invented number.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MatmulTilingPass.h"
#include "tensora/codegen/MlirLowerer.h"

namespace tns = tensora;
using namespace mlir;

namespace {

constexpr int64_t kDim = 1024;
constexpr int kReps = 5;  // report the best of kReps runs, to cut JIT/OS-noise outliers

tns::TypedProgram buildMatmulProgram(tns::TypeContext& types) {
  tns::GraphBuilder builder("mm");
  const tns::TensorType* aT =
      types.tensor({tns::Axis{"m", tns::Dimension::makeConst(kDim)}, tns::Axis{"k", tns::Dimension::makeConst(kDim)}});
  const tns::TensorType* bT =
      types.tensor({tns::Axis{"k", tns::Dimension::makeConst(kDim)}, tns::Axis{"n", tns::Dimension::makeConst(kDim)}});
  const tns::TensorType* cT =
      types.tensor({tns::Axis{"m", tns::Dimension::makeConst(kDim)}, tns::Axis{"n", tns::Dimension::makeConst(kDim)}});
  tns::ValueId a = builder.addInput("a", aT, {});
  tns::ValueId bIn = builder.addInput("b", bT, {});
  tns::ValueId c = builder.emit(tns::OpKind::MatMul, {a, bIn}, {}, cT, {}, "c");
  builder.setOutput(c);
  tns::TypedGraph graph = builder.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

StridedMemRefType<float, 2> makeMemRef(float* data) {
  StridedMemRefType<float, 2> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = kDim;
  m.sizes[1] = kDim;
  m.strides[0] = kDim;
  m.strides[1] = 1;
  return m;
}

// tileSize == 0 means "no tiling pass applied at all." Returns
// {result, best-of-kReps wall time in seconds}, or {NAN, 0} on failure.
std::pair<float, double> benchmarkOneConfig(int64_t tileSize) {
  tns::TypeContext types;
  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(buildMatmulProgram(types));
  if (failed(verify(module))) {
    fprintf(stderr, "module does not verify\n");
    return {NAN, 0.0};
  }

  if (tileSize > 0) {
    PassManager tilingPm(&ctx);
    tilingPm.addNestedPass<func::FuncOp>(
        tns::createMatmulTilingPass(tns::TilingPolicy{/*minDimToTile=*/128, /*maxTileSize=*/tileSize}));
    if (failed(tilingPm.run(module))) {
      fprintf(stderr, "tiling pass failed for tileSize=%lld\n", static_cast<long long>(tileSize));
      return {NAN, 0.0};
    }
  }

  PassManager pm(&ctx);
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
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    fprintf(stderr, "standard lowering pipeline failed\n");
    return {NAN, 0.0};
  }

  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "ExecutionEngine::create failed\n");
    return {NAN, 0.0};
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("_mlir_ciface_mm");
  if (!maybeFn) {
    fprintf(stderr, "could not look up '_mlir_ciface_mm'\n");
    return {NAN, 0.0};
  }
  using CIfaceMatmul = void (*)(StridedMemRefType<float, 2>*, StridedMemRefType<float, 2>*,
                                 StridedMemRefType<float, 2>*);
  auto fn = reinterpret_cast<CIfaceMatmul>(*maybeFn);

  std::vector<float> aData(kDim * kDim, 1.0f);
  std::vector<float> bData(kDim * kDim, 1.0f);
  auto aMem = makeMemRef(aData.data());
  auto bMem = makeMemRef(bData.data());

  float result = 0.0f;
  double best = 1e18;
  for (int rep = 0; rep < kReps; ++rep) {
    StridedMemRefType<float, 2> cMem = {};
    auto t0 = std::chrono::steady_clock::now();
    fn(&cMem, &aMem, &bMem);
    auto t1 = std::chrono::steady_clock::now();
    result = cMem.data[cMem.offset];
    std::free(cMem.basePtr);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed < best) best = elapsed;
  }
  return {result, best};
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  const float expected = static_cast<float>(kDim);

  struct Config {
    const char* label;
    int64_t tileSize;
  };
  const std::vector<Config> configs = {
      {"untiled", 0},
      {"16x16x16", 16},
      {"32x32x32", 32},
      {"64x64x64", 64},
  };

  printf("workload: %lldx%lldx%lld matmul, f32, all-ones operands, best of %d runs per config\n",
         static_cast<long long>(kDim), static_cast<long long>(kDim), static_cast<long long>(kDim), kReps);
  printf("%-12s %12s %10s\n", "config", "seconds", "correct");

  double untiledTime = -1.0;
  for (const Config& c : configs) {
    auto [result, time] = benchmarkOneConfig(c.tileSize);
    bool correct = (result == expected);
    printf("%-12s %12.6f %10s", c.label, time, correct ? "yes" : "NO");
    if (c.tileSize == 0) {
      untiledTime = time;
    } else if (untiledTime > 0) {
      printf("   (%.2fx untiled)", untiledTime / time);
    }
    printf("\n");
    if (!correct) {
      fprintf(stderr, "BENCHMARK FAIL: config %s produced %f, expected %f\n", c.label, result, expected);
      return 1;
    }
  }
  return 0;
}
