// LOOP-005: proves MlirLowerer computes a real LivenessResult for the
// graph it is lowering and threads it into TilingDecision::decide(),
// instead of always taking decide()'s liveness=nullptr default.
// LOOP-001-F already proves TilingDecision itself picks the
// conservative tile size when given a LivenessResult over budget --
// but that test hand-builds the LivenessResult directly. This test
// builds only a TypedGraph (no liveness value anywhere), lowers it
// through the real MlirLowerer, and reads the "tensora.tile_sizes"
// attribute MlirLowerer stamps on the linalg.matmul it produces. The
// only way that attribute can show the conservative size is if
// MlirLowerer's own internal liveness computation (LivenessAnalysis,
// not a fixture) found the same memory pressure LOOP-001-F fabricates
// by hand.
#include <array>
#include <cstdio>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MlirLowerer.h"

namespace tns = tensora;
using namespace mlir;

namespace {

// Builds `A(m,k) @ B(k,n) -> C(m,n)`, returning C. When `withBigWeight`
// is true, also declares an unused weight big enough on its own to
// exceed TilingPolicy{}'s default 1 MiB memory budget (600*600*4 bytes
// ~= 1.37 MiB): every TypedGraph param is live from the graph's entry
// (Liveness.cpp), so this alone raises the graph's peak live-byte
// estimate past the budget, regardless of whether anything ever reads
// it.
tns::TypedProgram buildProgram(tns::TypeContext& types, bool withBigWeight) {
  using tns::Axis;
  using tns::Dimension;
  tns::GraphBuilder b("g");
  const tns::TensorType* lhsT =
      types.tensor({Axis{"m", Dimension::makeConst(128)}, Axis{"k", Dimension::makeConst(128)}});
  const tns::TensorType* rhsT =
      types.tensor({Axis{"k", Dimension::makeConst(128)}, Axis{"n", Dimension::makeConst(128)}});
  const tns::TensorType* outT =
      types.tensor({Axis{"m", Dimension::makeConst(128)}, Axis{"n", Dimension::makeConst(128)}});
  tns::ValueId a = b.addInput("a", lhsT, {});
  tns::ValueId w = b.addWeight("w", rhsT, {});
  if (withBigWeight) {
    const tns::TensorType* bigT =
        types.tensor({Axis{"p", Dimension::makeConst(600)}, Axis{"q", Dimension::makeConst(600)}});
    b.addWeight("big", bigT, {});
  }
  tns::ValueId c = b.emit(tns::OpKind::MatMul, {a, w}, {}, outT, {}, "c");
  b.setOutput(c);
  tns::TypedGraph graph = b.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

// A@B -> C where A, B, and C are each dim x dim: no extra weight, no
// fixture -- the matmul's own operands are what create memory
// pressure, once dim is large enough.
tns::TypedProgram buildSquareMatmulProgram(tns::TypeContext& types, int64_t dim) {
  using tns::Axis;
  using tns::Dimension;
  tns::GraphBuilder b("g");
  const tns::TensorType* aT =
      types.tensor({Axis{"m", Dimension::makeConst(dim)}, Axis{"k", Dimension::makeConst(dim)}});
  const tns::TensorType* bT =
      types.tensor({Axis{"k", Dimension::makeConst(dim)}, Axis{"n", Dimension::makeConst(dim)}});
  const tns::TensorType* cT =
      types.tensor({Axis{"m", Dimension::makeConst(dim)}, Axis{"n", Dimension::makeConst(dim)}});
  tns::ValueId a = b.addInput("a", aT, {});
  tns::ValueId w = b.addWeight("b", bT, {});
  tns::ValueId c = b.emit(tns::OpKind::MatMul, {a, w}, {}, cT, {}, "c");
  b.setOutput(c);
  tns::TypedGraph graph = b.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

// Lowers `program` under `policy` and returns the tile sizes
// MlirLowerer stamped on the one linalg.matmul it produces, or
// {-1,-1,-1} if none is found.
std::array<int64_t, 3> lowerAndReadTileSizes(tns::TypedProgram& program, tns::TilingPolicy policy = {}) {
  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  tns::MlirLowerer lowerer(ctx, policy);
  ModuleOp module = lowerer.lower(program);

  std::array<int64_t, 3> sizes{-1, -1, -1};
  module.walk([&](linalg::MatmulOp op) {
    auto attr = op->getAttrOfType<DenseI64ArrayAttr>("tensora.tile_sizes");
    if (attr && attr.size() == 3) {
      for (int i = 0; i < 3; ++i) sizes[i] = attr.asArrayRef()[i];
    }
  });
  return sizes;
}

}  // namespace

int main() {
  bool allOk = true;

  {
    tns::TypeContext types;
    tns::TypedProgram program = buildProgram(types, /*withBigWeight=*/false);
    std::array<int64_t, 3> sizes = lowerAndReadTileSizes(program);
    bool ok = sizes[0] == 32 && sizes[1] == 32 && sizes[2] == 32;
    if (!ok) fprintf(stderr, "LOOP-005-A FAIL: got {%ld,%ld,%ld}, expected {32,32,32}\n", (long)sizes[0],
                      (long)sizes[1], (long)sizes[2]);
    printf("LOOP-005-A %s: no memory pressure -> default tile size 32 (real MlirLowerer, real Liveness)\n",
           ok ? "PASS" : "FAIL");
    allOk &= ok;
  }

  {
    tns::TypeContext types;
    tns::TypedProgram program = buildProgram(types, /*withBigWeight=*/true);
    std::array<int64_t, 3> sizes = lowerAndReadTileSizes(program);
    bool ok = sizes[0] == 16 && sizes[1] == 16 && sizes[2] == 16;
    if (!ok) fprintf(stderr, "LOOP-005-B FAIL: got {%ld,%ld,%ld}, expected {16,16,16}\n", (long)sizes[0],
                      (long)sizes[1], (long)sizes[2]);
    printf(
        "LOOP-005-B %s: an unused 1.37 MiB weight alone pushes peak live bytes over budget -> conservative tile "
        "size 16, chosen by MlirLowerer's own LivenessAnalysis call, not a hand-built LivenessResult\n",
        ok ? "PASS" : "FAIL");
    allOk &= ok;
  }

  {
    // LOOP-005-C: a realistic-sized matmul, no artificial extra weight
    // anywhere. A 512x512 f32 tensor is 512*512*4 = 1,048,576 bytes (1
    // MiB) on its own; A + B + C together are already 3x the default
    // 1 MiB budget. Memory pressure here comes purely from the
    // matmul's own operand sizes -- exactly the "a much larger matrix
    // will cause memory/cache pressure" case, not a contrived fixture.
    tns::TypeContext types;
    tns::TypedProgram program = buildSquareMatmulProgram(types, /*dim=*/512);
    std::array<int64_t, 3> sizes = lowerAndReadTileSizes(program);
    bool ok = sizes[0] == 16 && sizes[1] == 16 && sizes[2] == 16;
    if (!ok) fprintf(stderr, "LOOP-005-C FAIL: got {%ld,%ld,%ld}, expected {16,16,16}\n", (long)sizes[0],
                      (long)sizes[1], (long)sizes[2]);
    printf(
        "LOOP-005-C %s: a plain 512x512x512 matmul already exceeds the default 1 MiB budget on its own operand "
        "sizes (3 MiB total) -> conservative tile size 16, no extra weight needed\n",
        ok ? "PASS" : "FAIL");
    allOk &= ok;
  }

  {
    // LOOP-005-D: the same 512x512x512 matmul as LOOP-005-C, but under
    // a larger memory budget (4 MiB instead of the default 1 MiB) --
    // e.g. modeling a target with more cache/memory available for this
    // working set. TilingDecision's memory check is a static estimate
    // (verified shapes x 4 bytes/element) compared against a
    // TilingPolicy field, not a runtime memory measurement -- so
    // "simulating" a machine with more or less available memory for
    // this decision is exactly this: pass a different
    // TilingPolicy::memoryBudgetBytes, not an OS-level memory limit
    // (cgroups/ulimit) on the process. No such mechanism is needed or
    // used here.
    tns::TypeContext types;
    tns::TypedProgram program = buildSquareMatmulProgram(types, /*dim=*/512);
    tns::TilingPolicy roomyPolicy;
    roomyPolicy.memoryBudgetBytes = 4 * (1 << 20);  // 4 MiB: 3 MiB peak now fits comfortably
    std::array<int64_t, 3> sizes = lowerAndReadTileSizes(program, roomyPolicy);
    bool ok = sizes[0] == 32 && sizes[1] == 32 && sizes[2] == 32;
    if (!ok) fprintf(stderr, "LOOP-005-D FAIL: got {%ld,%ld,%ld}, expected {32,32,32}\n", (long)sizes[0],
                      (long)sizes[1], (long)sizes[2]);
    printf(
        "LOOP-005-D %s: the identical matmul, under a 4 MiB budget instead of the 1 MiB default, no longer counts "
        "as memory-pressured -> back to the default tile size 32\n",
        ok ? "PASS" : "FAIL");
    allOk &= ok;
  }

  return allOk ? 0 : 1;
}
