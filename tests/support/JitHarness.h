#pragma once
// Shared "run this lowered module through the standard MLIR/LLVM
// pipeline, then JIT-execute it" helper for the codegen e2e tests
// (tests/codegen/test_*_e2e.cpp). Every one of those tests used to
// duplicate this same bufferize -> linalg-to-loops -> ... -> LLVM
// pipeline and ExecutionEngine setup by hand (26-46 near-identical
// lines each). This header is that pipeline, written once, with the
// small amount of real per-test variation (see JitPipelineOptions)
// pulled out as flags instead of copy-pasted pass lists.
//
// This header is included by several .cpp files that all link into one
// binary (test_codegen_e2e); every function here is `inline` so that is
// safe (no multiple-definition link errors).
#include <cstdio>
#include <memory>
#include <optional>
#include <string>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"

namespace tensora::testing {

// The only real variation across the e2e tests' pipelines (confirmed by
// reading every test_*_e2e.cpp's pipeline block): whether bufferization
// must tolerate tensor-typed loop results/allocations, whether
// function-boundary layouts must be forced fully-dynamic (both needed
// once a graph produces `scf.for`-nested or reshaped tensors, e.g.
// tiling/conv/attention), whether `memref.subview` metadata needs
// expanding (paired 1:1 with lowering `affine.apply`, so both flags
// move together), whether the graph lowers any `math.*` op (softmax's
// exp, attention's exp), and whether the JIT engine needs the MLIR
// C-runner-utils shared library on its search path (needed once the
// bufferized program calls into it, which tracks the same tests that
// need the fully-dynamic-layout/alloc-tolerant bufferization above).
struct JitPipelineOptions {
  bool allowReturnAllocsFromLoops = false;
  bool fullyDynamicFunctionBoundaryLayout = false;
  bool expandStridedMetadataAndLowerAffine = false;
  bool convertMathToLLVM = false;
  bool useCRunnerUtilsSharedLib = false;
};

// Runs the standard Tensora-to-LLVM lowering pipeline on `module` in
// place: one-shot bufferization, linalg-to-loops, scf-to-cf, then every
// dialect down to LLVM, finishing with unrealized-cast reconciliation.
// `label` prefixes the FAIL message (so failures still read like
// "E2E-002 FAIL: ...") and the module is dumped to stderr on failure, so
// a failing case shows the exact IR the pipeline choked on. Returns
// true on success.
inline bool runStandardLoweringPipeline(mlir::MLIRContext& ctx, mlir::ModuleOp module,
                                         const JitPipelineOptions& opts, const char* label) {
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = opts.allowReturnAllocsFromLoops;
  if (opts.fullyDynamicFunctionBoundaryLayout) {
    bufOpts.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  }

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  if (opts.expandStridedMetadataAndLowerAffine) {
    // memref.subview (from tiling/reshape) lowers to LLVM only after its
    // offset/stride metadata is expanded into explicit index arithmetic
    // (memref::createExpandStridedMetadataPass), and that expansion
    // computes offsets via affine.apply, so LowerAffine must follow it
    // and both must run before ArithToLLVM sees the new arithmetic.
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
  }
  if (opts.convertMathToLLVM) pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module))) {
    fprintf(stderr, "%s FAIL: standard lowering pipeline failed\n", label);
    module.dump();
    return false;
  }
  return true;
}

// Runs `module` (already through runStandardLoweringPipeline) through
// the JIT and returns the resulting engine, or std::nullopt (with a FAIL
// message on stderr) if engine creation fails. The caller owns the
// returned engine and must keep it alive for as long as it calls any
// function looked up from it -- the same lifetime rule
// mlir::ExecutionEngine::create itself has, just not reimplemented at
// every call site.
inline std::optional<std::unique_ptr<mlir::ExecutionEngine>> createJitEngine(mlir::ModuleOp module,
                                                                              const JitPipelineOptions& opts,
                                                                              const char* label) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  if (opts.useCRunnerUtilsSharedLib) engineOpts.sharedLibPaths = kSharedLibs;

  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "%s FAIL: ExecutionEngine::create failed\n", label);
    return std::nullopt;
  }
  return std::move(*maybeEngine);
}

// Looks up `entryName` in `engine` and casts it to `Fn` (a zero-argument
// function-pointer type, e.g. `float (*)()`). Returns std::nullopt (with
// a FAIL message on stderr) if the symbol lookup fails.
template <typename Fn>
inline std::optional<Fn> lookupJitEntry(mlir::ExecutionEngine& engine, const std::string& entryName,
                                         const char* label) {
  auto maybeFn = engine.lookup(entryName);
  if (!maybeFn) {
    fprintf(stderr, "%s FAIL: could not look up '%s'\n", label, entryName.c_str());
    return std::nullopt;
  }
  return reinterpret_cast<Fn>(*maybeFn);
}

}  // namespace tensora::testing
