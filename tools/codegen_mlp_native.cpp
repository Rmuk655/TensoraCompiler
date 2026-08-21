// CodegenDriver (ARCH-009/ARCHITECTURE.md "BackendPipeline"). This
// compiles the same hand-built MLP TypedGraph LOW-E2E-MLP uses, all
// the way to a real native object file, through LLVM's TargetMachine,
// not the JIT. This lets NATIVE-001 link it and run it as an ahead-
// of-time-compiled executable, proving the pipeline produces real
// machine code, not just something an in-process JIT happens to run.
//
// Usage: codegen_mlp_native <output-object-file-path>
#include <cstdio>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MlirLowerer.h"
#include "tensora/optimization/GraphPass.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

// The same graph shape as test_mlp_e2e.cpp's LOW-E2E-MLP (linear ->
// relu -> linear). This project keeps the two in sync on purpose, so
// NATIVE-001 tests the same workload through a different backend
// path as that JIT test, not a new numerical scenario.
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

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <output.o>\n", argv[0]);
    return 1;
  }
  const char* outputPath = argv[1];

  // --- Build and optimize the TypedGraph (same as LOW-E2E-MLP). ---
  tns::TypeContext types;
  tns::GraphBuilder builder("mlp");
  tns::ValueId outValue;
  buildMlpGraph(types, builder, outValue);
  tns::TypedGraph graph = builder.finish();

  tns::GraphVerifier verifier;
  if (!verifier.verify(graph).empty()) {
    fprintf(stderr, "codegen_mlp_native FAIL: hand-built graph is not GraphVerifier-clean\n");
    return 1;
  }

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
    fprintf(stderr, "codegen_mlp_native FAIL: lowered module does not verify\n");
    return 1;
  }

  // --- Harness: a zero-arg, scalar-returning entry point, exactly
  // like LOW-E2E-MLP's "test_main". Its ABI is deliberately trivial
  // (no memref marshalling), so the native driver can call it as a
  // plain C function. ---
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto f32 = b.getF32Type();
  auto harnessType = b.getFunctionType({}, {f32});
  auto harness = func::FuncOp::create(b, loc, "tensora_entry", harnessType);
  Block* entry = harness.addEntryBlock();
  OpBuilder hb(entry, entry->begin());

  MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 2}, f32), {1.0f, 2.0f});
  MlirValue w1 = constTensor(hb, loc, RankedTensorType::get({2, 2}, f32), {1.0f, 0.0f, 0.0f, 1.0f});
  MlirValue b1 = constTensor(hb, loc, RankedTensorType::get({2}, f32), {0.0f, 0.0f});
  MlirValue w2 = constTensor(hb, loc, RankedTensorType::get({2, 1}, f32), {1.0f, 1.0f});
  MlirValue b2 = constTensor(hb, loc, RankedTensorType::get({1}, f32), {0.0f});

  auto mlpFunc = module.lookupSymbol<func::FuncOp>("mlp");
  if (!mlpFunc) {
    fprintf(stderr, "codegen_mlp_native FAIL: lowered module has no 'mlp' function\n");
    return 1;
  }
  auto call = func::CallOp::create(hb, loc, mlpFunc, ValueRange{x, w1, b1, w2, b2});
  MlirValue yTensor = call.getResult(0);
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, yTensor, ValueRange{c0, c0});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "codegen_mlp_native FAIL: module with harness does not verify\n");
    return 1;
  }

  // --- Standard MLIR/LLVM lowering pipeline (same as LOW-E2E-MLP). ---
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
    fprintf(stderr, "codegen_mlp_native FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  // --- Translate to LLVM IR (standard MLIR->LLVM IR export, not a
  // Tensora-authored step). ---
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext, "tensora_mlp");
  if (!llvmModule) {
    fprintf(stderr, "codegen_mlp_native FAIL: MLIR->LLVM IR translation failed\n");
    return 1;
  }

  // --- Emit a real native object file via LLVM's TargetMachine (ahead-
  // of-time codegen, not the ExecutionEngine's in-process JIT). ---
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  llvmModule->setTargetTriple(targetTriple);

  std::string lookupError;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple.str(), lookupError);
  if (!target) {
    fprintf(stderr, "codegen_mlp_native FAIL: TargetRegistry::lookupTarget: %s\n", lookupError.c_str());
    return 1;
  }

  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> targetMachine(target->createTargetMachine(
      targetTriple, /*CPU=*/"generic", /*Features=*/"", targetOptions, llvm::Reloc::PIC_));
  if (!targetMachine) {
    fprintf(stderr, "codegen_mlp_native FAIL: createTargetMachine returned null\n");
    return 1;
  }
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  std::error_code ec;
  llvm::raw_fd_ostream dest(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    fprintf(stderr, "codegen_mlp_native FAIL: could not open '%s': %s\n", outputPath, ec.message().c_str());
    return 1;
  }

  llvm::legacy::PassManager codegenPm;
  if (targetMachine->addPassesToEmitFile(codegenPm, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    fprintf(stderr, "codegen_mlp_native FAIL: TargetMachine can't emit an object file for this target\n");
    return 1;
  }
  codegenPm.run(*llvmModule);
  dest.flush();

  printf("codegen_mlp_native: wrote native object file to %s (entry point: tensora_entry)\n", outputPath);
  return 0;
}
