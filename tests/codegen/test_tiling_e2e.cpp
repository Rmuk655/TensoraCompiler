// LOOP-002/003/004: checks the structure and the numbers of the
// project-authored MatmulTilingPass, applied to a real
// MlirLowerer-produced module. It also reports timing, but does not
// assert a timing value.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/Passes.h"
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
using MlirValue = mlir::Value;

namespace {

// Large enough to show a real, dramatic tiled-vs-untiled speedup (cache
// blocking actually matters at this size, not just JIT-call noise);
// still small enough to fill/verify by hand (every output element =
// kDim for all-ones operands) and to add only a few seconds to `ctest`,
// not minutes.
constexpr int64_t kDim = 1024;

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

// Builds a StridedMemRefType view over caller-owned, row-major kDim x
// kDim data -- the layout the standard lowering pipeline's LLVM-level
// ABI expects for a tensor<kDim x kDim x f32> argument/result.
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

// Returns {result, wall_time_seconds}, or {NAN, 0} on failure. Calls the
// lowered "mm" function directly through its emitted C interface
// (_mlir_ciface_mm), passing memref descriptors for two all-ones kDim x
// kDim operands -- this avoids routing through a second, calling
// function entirely, so the module under test (and under
// bufferization) is always the single function MlirLowerer produced
// plus whatever MatmulTilingPass did to it, nothing more.
std::pair<float, double> runPipeline(ModuleOp module, MLIRContext& ctx, bool applyTiling, bool* sawTiledLoop) {
  if (applyTiling) {
    PassManager tilingPm(&ctx);
    // 128 is a reasonable cache-blocking tile size for a 1024x1024x1024
    // matmul (8 tiles per dimension), not just the smallest size that
    // still exercises the pass.
    tilingPm.addNestedPass<func::FuncOp>(
        tns::createMatmulTilingPass(tns::TilingPolicy{/*minDimToTile=*/128, /*maxTileSize=*/128}));
    if (failed(tilingPm.run(module))) {
      fprintf(stderr, "tiling pass failed\n");
      return {NAN, 0.0};
    }
  }
  if (sawTiledLoop) {
    *sawTiledLoop = false;
    module.walk([&](scf::ForOp) { *sawTiledLoop = true; });
  }

  PassManager pm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;  // tiling's scf.for loops carry tensor iter_args/results
  // The default InferLayoutMap tries to compute a precise (possibly
  // static-stride) layout per function boundary; the subview-derived
  // layouts inside a tiled loop nest defeat that inference. Forcing
  // fully-dynamic-strided memrefs at every function boundary sidesteps
  // the inference entirely (still correct, just less precise typing).
  bufOpts.functionBoundaryTypeConversion = bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  // memref.subview (introduced by tiling) lowers to LLVM only after its
  // offset/stride metadata has been expanded into explicit index
  // arithmetic; this must run *before* ArithToLLVM so that the
  // newly-introduced index-typed arithmetic gets converted to i64 in the
  // same sweep as everything else, instead of being stranded afterward.
  pm.addPass(memref::createExpandStridedMetadataPass());
  // ExpandStridedMetadata computes subview offsets via affine.apply;
  // that needs lowering to arith before ArithToLLVM can see it.
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    fprintf(stderr, "standard lowering pipeline failed\n");
    module.dump();
    return {NAN, 0.0};
  }

  // Tiling's per-tile memref.copy on non-identity-layout subviews lowers
  // to a call to the `memrefCopy` runtime helper (a strided element-wise
  // copy loop; not something the LLVM conversion inlines), which lives
  // in MLIR's own runner-support shared library, not in this process.
  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "ExecutionEngine::create failed\n");
    module.dump();
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
  // "mm" returns its result by value; the ciface wrapper mallocs the
  // backing storage itself and overwrites this descriptor's fields (data
  // /basePtr included) with the new allocation -- this is purely an
  // out-parameter for the wrapper to fill in, not caller-owned storage.
  StridedMemRefType<float, 2> cMem = {};

  auto t0 = std::chrono::steady_clock::now();
  fn(&cMem, &aMem, &bMem);
  auto t1 = std::chrono::steady_clock::now();
  float result = cMem.data[cMem.offset];
  std::free(cMem.basePtr);
  return {result, std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  const float expected = static_cast<float>(kDim);  // all-ones 1024x1024 @ 1024x1024 -> every element = 1024

  // --- Untiled baseline. ---
  tns::TypeContext types1;
  DialectRegistry registry1 = tns::makeTensoraDialectRegistry();
  MLIRContext ctx1(registry1);
  ctx1.loadAllAvailableDialects();
  tns::MlirLowerer lowerer1(ctx1);
  ModuleOp moduleUntiled = lowerer1.lower(buildMatmulProgram(types1));
  if (failed(verify(moduleUntiled))) {
    fprintf(stderr, "LOOP FAIL: untiled module does not verify\n");
    return 1;
  }
  bool untiledSawLoop = false;
  auto [untiledResult, untiledTime] = runPipeline(moduleUntiled, ctx1, /*applyTiling=*/false, &untiledSawLoop);

  // --- Tiled (MatmulTilingPass applied). ---
  tns::TypeContext types2;
  DialectRegistry registry2 = tns::makeTensoraDialectRegistry();
  MLIRContext ctx2(registry2);
  ctx2.loadAllAvailableDialects();
  tns::MlirLowerer lowerer2(ctx2);
  ModuleOp moduleTiled = lowerer2.lower(buildMatmulProgram(types2));
  if (failed(verify(moduleTiled))) {
    fprintf(stderr, "LOOP FAIL: tiled module does not verify\n");
    return 1;
  }
  bool tiledSawLoop = false;
  auto [tiledResult, tiledTime] = runPipeline(moduleTiled, ctx2, /*applyTiling=*/true, &tiledSawLoop);

  bool ok = true;

  // LOOP-002: structural -- the tiled copy has scf.for loops from
  // tiling (introduced before the generic linalg->scf.for conversion
  // pass runs at all); the untiled copy has none at that point.
  if (!tiledSawLoop) {
    fprintf(stderr, "LOOP-002 FAIL: expected scf.for loops after MatmulTilingPass, found none\n");
    ok = false;
  }
  if (untiledSawLoop) {
    fprintf(stderr, "LOOP-002 FAIL: untiled copy unexpectedly has scf.for before the generic loop pass ran\n");
    ok = false;
  }

  // LOOP-003: numerical equivalence.
  if (untiledResult != expected) {
    fprintf(stderr, "LOOP-003 FAIL: untiled result %f != expected %f\n", untiledResult, expected);
    ok = false;
  }
  if (tiledResult != expected) {
    fprintf(stderr, "LOOP-003 FAIL: tiled result %f != expected %f\n", tiledResult, expected);
    ok = false;
  }
  if (tiledResult != untiledResult) {
    fprintf(stderr, "LOOP-003 FAIL: tiled (%f) != untiled (%f)\n", tiledResult, untiledResult);
    ok = false;
  }

  if (!ok) return 1;

  // LOOP-004: reported, not asserted. At 1024x1024x1024 the wall times
  // below are real, measurable numbers (seconds), not JIT-call noise --
  // but this single untuned run is still not a rigorous performance
  // claim on its own (no warmup, no repetition, no machine info).
  // `tools/benchmark_matmul_tiling.cpp` (best-of-5, sweeps 16/32/64) is
  // the actual rigorous measurement behind the tiling policy's default
  // tile size -- see docs/testing/TEST_MATRIX.md's Loop optimization
  // section for the measured numbers.
  printf(
      "LOOP-002/003/004 PASS: tiled and untiled both compute %.1f (expected %.1f); tiled has scf.for "
      "loops, untiled does not; wall time untiled=%.6fs tiled=%.6fs (workload=%lldx%lldx%lld, single run, "
      "not a rigorous performance claim)\n",
      untiledResult, expected, untiledTime, tiledTime, static_cast<long long>(kDim), static_cast<long long>(kDim),
      static_cast<long long>(kDim));
  return 0;
}
