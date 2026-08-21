// LOOP-001: checks that TilingDecision is correct (ARCH-009;
// INTERFACES.md §4a). This test is plain C++, with no MLIR dependency,
// matching LoopTiling.h/.cpp's own design: you can unit-test the
// decision on its own, without MLIR.
#include "tensora/codegen/LoopTiling.h"
#include "support/TestHarness.h"

using namespace tensora;
using tensora::testing::Reporter;
#define CASE TEST_CASE

namespace {

GraphNode makeMatMulNode(TypedGraph& graph, TypeContext& types, int64_t m, int64_t k, int64_t n) {
  GraphBuilder b(graph.name);
  const TensorType* lhsT = types.tensor({Axis{"m", Dimension::makeConst(m)}, Axis{"k", Dimension::makeConst(k)}});
  const TensorType* rhsT = types.tensor({Axis{"k", Dimension::makeConst(k)}, Axis{"n", Dimension::makeConst(n)}});
  const TensorType* outT = types.tensor({Axis{"m", Dimension::makeConst(m)}, Axis{"n", Dimension::makeConst(n)}});
  ValueId lhs = b.addInput("a", lhsT, {});
  ValueId rhs = b.addInput("b", rhsT, {});
  ValueId out = b.emit(OpKind::MatMul, {lhs, rhs}, {}, outT, {}, "c");
  b.setOutput(out);
  graph = b.finish();
  return graph.node(0);
}

}  // namespace

CASE(LOOP_001_A, "LOOP-001-A", "TilingDecision: a large matmul (>= threshold on all dims) is tiled") {
  TypeContext types;
  TypedGraph graph;
  graph.name = "g";
  GraphNode node = makeMatMulNode(graph, types, 256, 256, 256);

  TilingDecision decision;
  TilingPlan plan = decision.decide(node, graph);
  R.check(plan.tile, "256x256x256 exceeds the default 128 threshold on every dim: tile");
  R.check(plan.tileSizes[0] == 32 && plan.tileSizes[1] == 32 && plan.tileSizes[2] == 32,
          "default policy's tile size (32) is used for all three dims");
}

CASE(LOOP_001_B, "LOOP-001-B", "TilingDecision: a tiny matmul is not tiled") {
  TypeContext types;
  TypedGraph graph;
  graph.name = "g";
  GraphNode node = makeMatMulNode(graph, types, 4, 4, 4);

  TilingDecision decision;
  TilingPlan plan = decision.decide(node, graph);
  R.check(!plan.tile, "4x4x4 is well below the default 128 threshold: skip tiling");
}

CASE(LOOP_001_C, "LOOP-001-C", "TilingDecision: dimensions divisible by the tile size") {
  TypeContext types;
  TypedGraph graph;
  graph.name = "g";
  GraphNode node = makeMatMulNode(graph, types, 128, 128, 128);

  TilingDecision decision;
  TilingPlan plan = decision.decide(node, graph);
  R.check(plan.tile && plan.tileSizes[0] == 32, "128 is exactly divisible by tile size 32 (4 tiles)");
}

CASE(LOOP_001_D, "LOOP-001-D", "TilingDecision: dimensions with a remainder still get a valid plan") {
  TypeContext types;
  TypedGraph graph;
  graph.name = "g";
  GraphNode node = makeMatMulNode(graph, types, 150, 150, 150);

  TilingDecision decision;
  TilingPlan plan = decision.decide(node, graph);
  R.check(plan.tile, "150 exceeds the threshold");
  R.check(plan.tileSizes[0] == 32, "tile size is still 32; MLIR's tiling handles the remainder (150 = 4*32 + 22)");
}

CASE(LOOP_001_E, "LOOP-001-E", "TilingDecision: symbolic dimensions are conservatively not tiled") {
  TypeContext types;
  GraphBuilder b("g");
  const TensorType* lhsT = types.tensor(
      {Axis{"m", Dimension::makeSymbol(types.internSymbol("M"))}, Axis{"k", Dimension::makeConst(256)}});
  const TensorType* rhsT = types.tensor({Axis{"k", Dimension::makeConst(256)}, Axis{"n", Dimension::makeConst(256)}});
  const TensorType* outT = types.tensor(
      {Axis{"m", Dimension::makeSymbol(types.internSymbol("M"))}, Axis{"n", Dimension::makeConst(256)}});
  ValueId lhs = b.addInput("a", lhsT, {});
  ValueId rhs = b.addInput("b", rhsT, {});
  ValueId out = b.emit(OpKind::MatMul, {lhs, rhs}, {}, outT, {}, "c");
  b.setOutput(out);
  TypedGraph graph = b.finish();

  TilingDecision decision;
  TilingPlan plan = decision.decide(graph.node(0), graph);
  R.check(!plan.tile, "a symbolic M dimension has no static loop bound to plan a tile count around: skip");
}

CASE(LOOP_001_F, "LOOP-001-F", "TilingDecision: memory pressure selects a smaller, more conservative tile size") {
  TypeContext types;
  TypedGraph graph;
  graph.name = "g";
  GraphNode node = makeMatMulNode(graph, types, 256, 256, 256);

  // A LivenessResult whose peak already exceeds the policy's memory
  // budget (a real consumer of the peak-memory estimate, LIVE-004).
  LivenessResult liveness;
  liveness.peakLiveBytes = TilingPolicy{}.memoryBudgetBytes + 1;

  TilingDecision decision;
  TilingPlan plan = decision.decide(node, graph, &liveness);
  R.check(plan.tile, "still large enough to tile");
  R.check(plan.tileSizes[0] == 16, "under memory pressure, the conservative tile size (16) is used instead of 32");
}

#undef CASE

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
