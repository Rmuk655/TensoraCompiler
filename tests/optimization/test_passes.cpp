// Level-1 graph optimization tests (RM 3.21), matching
// docs/testing/TEST_MATRIX.md OPT-*. These use hand-built GraphBuilder
// fixtures (ARCH-008), not the full source pipeline. This is fast and
// precise, and it is what 08_TEST_PLAN.md recommends for optimization
// tests.
#include <string>
#include <vector>

#include "tensora/optimization/Liveness.h"
#include "tensora/optimization/GraphPass.h"
#include "support/TestHarness.h"

using namespace tensora;
using tensora::testing::Reporter;
#define CASE TEST_CASE

namespace {

const TensorType* t2(TypeContext& types, const char* n0, int64_t s0, const char* n1, int64_t s1) {
  return types.tensor({Axis{n0, Dimension::makeConst(s0)}, Axis{n1, Dimension::makeConst(s1)}});
}

bool hasFact(const TypedGraph& g, NodeId id, const std::string& fact) { return g.node(id).facts.has(fact); }

}  // namespace

// --- OPT-SF-001: ShapeFolding eliminates a no-op reshape. ---
CASE(OPT_SF_001, "OPT-SF-001", "ShapeFolding: a reshape to the same shape is eliminated") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrs;
  attrs.reshapeTarget = tt->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrs, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createShapeFoldingPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "the no-op reshape is folded away");
  R.check(g.nodes().size() == 1 && g.nodes()[0].op == OpKind::Relu, "only relu remains");
  R.check(g.nodes()[0].inputs[0] == x, "relu now reads x directly");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_SF_002, "OPT-SF-002", "ShapeFolding near-miss: a reshape that actually changes shape is kept") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* tt2 = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrs;
  attrs.reshapeTarget = tt2->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrs, tt2, {}, "v1");
  b.setOutput(v1);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createShapeFoldingPass()->run(g, ctx);
  R.check(!r.changed, "a real shape change is not folded");
  R.check(g.nodes().size() == 1, "the reshape node remains");
}

// --- OPT-DCE-001/002: DeadOpElimination drops unreached values, keeps reached ones. ---
CASE(OPT_DCE_001, "OPT-DCE-001", "DeadOpElimination drops a value not reaching return") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId dead = b.emit(OpKind::Relu, {x}, {}, tt, {}, "dead");
  ValueId live = b.emit(OpKind::Relu, {x}, {}, tt, {}, "live");
  (void)dead;
  b.setOutput(live);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createDeadOpEliminationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "one dead node removed");
  R.check(g.nodes().size() == 1, "only the live node remains");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_DCE_002, "OPT-DCE-002", "DeadOpElimination near-miss: a value reaching return is kept") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createDeadOpEliminationPass()->run(g, ctx);
  R.check(!r.changed, "nothing is dead when every node reaches return");
  R.check(g.nodes().size() == 2, "both nodes remain");
}

// --- OPT-DCE-003: a whole dead chain (not just one dead node) is removed,
// while a same-shaped live chain right next to it is kept -- e.g.
// `h = linear(x, W1, b1); a = relu(h); y = linear(x, W2, b2); return a`:
// `y` is dead on its own, but so is any chain built only on top of it.
CASE(OPT_DCE_003, "OPT-DCE-003", "DeadOpElimination removes a whole dead chain, not just one dead node") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  // Live chain: x -> live1 -> live2, and live2 is returned.
  ValueId live1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "live1");
  ValueId live2 = b.emit(OpKind::Relu, {live1}, {}, tt, {}, "live2");
  // Dead chain: x -> dead1 -> dead2. Nothing ever reads dead2, and
  // dead2 is not returned, so neither dead1 nor dead2 reaches return.
  ValueId dead1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "dead1");
  ValueId dead2 = b.emit(OpKind::Relu, {dead1}, {}, tt, {}, "dead2");
  (void)dead2;
  b.setOutput(live2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createDeadOpEliminationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 2, "both dead nodes are removed in one pass, not just the first");
  R.check(g.nodes().size() == 2, "only the two live nodes remain");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

// --- OPT-RC-001: ReshapeCancellation merges a chain into one reshape. ---
CASE(OPT_RC_001, "OPT-RC-001", "ReshapeCancellation merges reshape(reshape(x,A),B) into reshape(x,B)") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* ttA = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  const TensorType* ttB = types.tensor({Axis{"d", Dimension::makeConst(2)}, Axis{"e", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrsA;
  attrsA.reshapeTarget = ttA->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrsA, ttA, {}, "v1");
  NodeAttrs attrsB;
  attrsB.reshapeTarget = ttB->axes();
  ValueId v2 = b.emit(OpKind::Reshape, {v1}, attrsB, ttB, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createReshapeCancellationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "the inner reshape is fused away");
  R.check(g.nodes().size() == 1, "one reshape node remains");
  if (g.nodes().size() == 1) {
    R.check(g.nodes()[0].inputs[0] == x, "the single reshape now reads x directly");
    R.check(g.nodes()[0].attrs.reshapeTarget.size() == 2, "its target is B (the outer target)");
  }
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_RC_002, "OPT-RC-002", "ReshapeCancellation near-miss: inner reshape used elsewhere is kept") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* ttA = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  const TensorType* ttB = types.tensor({Axis{"d", Dimension::makeConst(2)}, Axis{"e", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrsA;
  attrsA.reshapeTarget = ttA->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrsA, ttA, {}, "v1");
  NodeAttrs attrsB;
  attrsB.reshapeTarget = ttB->axes();
  ValueId v2 = b.emit(OpKind::Reshape, {v1}, attrsB, ttB, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v1}, {}, ttA, {}, "v3");  // second consumer of v1
  ValueId v4 = b.emit(OpKind::Add, {v2}, NodeAttrs{}, ttB, {}, "v4");
  (void)v3;
  b.setOutput(v4);
  // v2 = Add(v2_reshape) needs a tensor operand for Add; simplify: just
  // verify the pass leaves v1 alone because it has two consumers (v2, v3).
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createReshapeCancellationPass()->run(g, ctx);
  R.check(!r.changed, "inner reshape has two consumers, so it cannot be dropped");
  R.check(g.nodes().size() == 4, "all four nodes remain");
}

// --- OPT-RC-003: interaction with ShapeFolding (reshape immediately undone). ---
CASE(OPT_RC_003, "OPT-RC-003",
     "interaction: ReshapeCancellation + ShapeFolding together eliminate reshape-then-undo") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* ttFlat = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrsFlat;
  attrsFlat.reshapeTarget = ttFlat->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrsFlat, ttFlat, {}, "v1");
  NodeAttrs attrsBack;
  attrsBack.reshapeTarget = tt->axes();  // reshape right back to x's original shape
  ValueId v2 = b.emit(OpKind::Reshape, {v1}, attrsBack, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassManager pm;
  pm.add(createReshapeCancellationPass());
  pm.add(createShapeFoldingPass());
  PassResult r = pm.runAll(g, ctx);
  R.check(r.changed, "the pipeline changes the graph");
  R.check(g.nodes().empty(), "both reshapes fully cancel: merge to reshape(x,tt), then fold away as a no-op");
  R.check(g.output == x, "the graph's output is x directly");
}

// --- OPT-TC-001: TransposeCancellation fully cancels inverse perms. ---
CASE(OPT_TC_001, "OPT-TC-001", "TransposeCancellation: inverse perms cancel completely") {
  TypeContext types;
  const TensorType* tt = types.tensor(
      {Axis{"a", Dimension::makeConst(1)}, Axis{"b", Dimension::makeConst(2)}, Axis{"c", Dimension::makeConst(3)}});
  const TensorType* ttSwapped = types.tensor(
      {Axis{"c", Dimension::makeConst(3)}, Axis{"b", Dimension::makeConst(2)}, Axis{"a", Dimension::makeConst(1)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs p1;
  p1.perm = {2, 1, 0};
  ValueId v1 = b.emit(OpKind::Transpose, {x}, p1, ttSwapped, {}, "v1");
  NodeAttrs p2;
  p2.perm = {2, 1, 0};  // its own inverse
  ValueId v2 = b.emit(OpKind::Transpose, {v1}, p2, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createTransposeCancellationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 2, "both transpose nodes are removed");
  R.check(g.nodes().empty(), "no nodes remain");
  R.check(g.output == x, "output is x directly");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_TC_002, "OPT-TC-002", "TransposeCancellation: non-inverse perms compose into one transpose") {
  TypeContext types;
  const TensorType* tt = types.tensor(
      {Axis{"a", Dimension::makeConst(1)}, Axis{"b", Dimension::makeConst(2)}, Axis{"c", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs p1;
  p1.perm = {1, 2, 0};
  ValueId v1 = b.emit(OpKind::Transpose, {x}, p1, tt, {}, "v1");
  NodeAttrs p2;
  p2.perm = {1, 2, 0};  // same perm twice != identity when composed (a 3-cycle squared is another 3-cycle)
  ValueId v2 = b.emit(OpKind::Transpose, {v1}, p2, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createTransposeCancellationPass()->run(g, ctx);
  R.check(r.changed, "the pair still merges into one transpose (a real optimization, fewer ops)");
  R.check(g.nodes().size() == 1, "one transpose node remains");
  if (g.nodes().size() == 1) {
    R.check(g.nodes()[0].inputs[0] == x, "reads x directly");
    R.check(g.nodes()[0].attrs.perm == std::vector<uint32_t>{2, 0, 1}, "perm is the correct composition");
  }
}

CASE(OPT_TC_003, "OPT-TC-003", "TransposeCancellation near-miss: inner transpose used elsewhere is kept") {
  TypeContext types;
  const TensorType* tt = types.tensor(
      {Axis{"a", Dimension::makeConst(1)}, Axis{"b", Dimension::makeConst(2)}, Axis{"c", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs p1;
  p1.perm = {2, 1, 0};
  ValueId v1 = b.emit(OpKind::Transpose, {x}, p1, tt, {}, "v1");
  NodeAttrs p2;
  p2.perm = {2, 1, 0};
  ValueId v2 = b.emit(OpKind::Transpose, {v1}, p2, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v3");  // second use of v1
  ValueId v4 = b.emit(OpKind::Add, {v2}, NodeAttrs{}, tt, {}, "v4");
  (void)v3;
  b.setOutput(v4);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createTransposeCancellationPass()->run(g, ctx);
  R.check(!r.changed, "v1 has two consumers, so the pair cannot be touched");
  R.check(g.nodes().size() == 4, "all four nodes remain");
}

// --- OPT-EF-001: ElementwiseFusion tags a chain. ---
CASE(OPT_EF_001, "OPT-EF-001", "ElementwiseFusion tags a chain of >= 2 elementwise ops") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs a1;
  a1.scalar = 2.0;
  ValueId v1 = b.emit(OpKind::Add, {x}, a1, tt, {}, "v1");
  NodeAttrs a2;
  a2.scalar = 3.0;
  ValueId v2 = b.emit(OpKind::Mul, {v1}, a2, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createElementwiseFusionPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "one fusion group recognized");
  R.check(g.nodes().size() == 2, "the pass only tags; it does not restructure the graph");
  R.check(hasFact(g, 0, "fusion_group_0") && hasFact(g, 1, "fusion_group_0"), "both nodes share one fact tag");
}

CASE(OPT_EF_002, "OPT-EF-002", "ElementwiseFusion near-miss: a lone elementwise op is not tagged") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs a1;
  a1.scalar = 2.0;
  ValueId v1 = b.emit(OpKind::Add, {x}, a1, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");  // breaks the elementwise chain
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createElementwiseFusionPass()->run(g, ctx);
  R.check(!r.changed, "a chain of length 1 (Add alone) is not a fusion group");
  R.check(!hasFact(g, 0, "fusion_group_0"), "no fact recorded");
}

// --- OPT-LBA-001/002: LinearBiasActivationFusion. ---
CASE(OPT_LBA_001, "OPT-LBA-001", "LinearBiasActivationFusion tags linear immediately followed by relu") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  ValueId relu = b.emit(OpKind::Relu, {lin}, {}, y, {}, "relu");
  b.setOutput(relu);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createLinearBiasActivationFusionPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "one fusion pair recognized");
  R.check(hasFact(g, 0, "fused_linear_relu_0") && hasFact(g, 1, "fused_linear_relu_0"),
          "both linear and relu share one fact tag");
}

CASE(OPT_LBA_002, "OPT-LBA-002", "LinearBiasActivationFusion near-miss: linear+softmax is not tagged") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  NodeAttrs smAttrs;
  smAttrs.axisName = "out";
  ValueId sm = b.emit(OpKind::Softmax, {lin}, smAttrs, y, {}, "sm");
  b.setOutput(sm);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createLinearBiasActivationFusionPass()->run(g, ctx);
  R.check(!r.changed, "linear followed by softmax (not relu) is not a recognized fusion pattern");
}

// --- OPT-FIX-001: one sweep of DeadOpElimination can expose a fusion
// opportunity a later sweep must pick up. Graph: linear "lin" feeds two
// consumers, "dead" (a relu that never reaches return) and "live" (a
// relu that does). Because "lin" has two consumers, LinearBiasActivation-
// Fusion cannot tag it in the first sweep -- RM 3.9's fused-region
// legality needs the Linear's result read exactly once, since fusing it
// with a second, different consumer's activation would be wrong.
// DeadOpElimination removes "dead", dropping "lin" to one consumer, so a
// second sweep's LinearBiasActivationFusion call now tags "lin"+"live".
// A PassManager that runs every pass exactly once, in order, cannot see
// this: DeadOpElimination runs after the fusion passes in one sweep, so
// the opportunity would appear one sweep too late for a single pass.
CASE(OPT_FIX_001, "OPT-FIX-001",
     "DeadOpElimination removing a linear's dead consumer lets a later sweep fuse the surviving relu") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  ValueId dead = b.emit(OpKind::Relu, {lin}, {}, y, {}, "dead");  // never reaches return
  ValueId live = b.emit(OpKind::Relu, {lin}, {}, y, {}, "live");
  (void)dead;
  b.setOutput(live);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassManager pm;
  pm.add(createElementwiseFusionPass());
  pm.add(createLinearBiasActivationFusionPass());
  pm.add(createDeadOpEliminationPass());
  PassResult r = pm.runAll(g, ctx);

  R.check(r.changed, "the pipeline changes the graph across sweeps");
  R.check(r.stats.iterations >= 2, "reaching the fixed point took more than one sweep over the pass list");
  R.check(g.nodes().size() == 2, "\"dead\" is gone; only \"lin\" and \"live\" remain");
  if (g.nodes().size() == 2) {
    R.check(hasFact(g, 0, "fused_linear_relu_0") && hasFact(g, 1, "fused_linear_relu_0"),
            "lin+live are tagged fused, even though that was not possible in the first sweep");
  }
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_FIX_002, "OPT-FIX-002",
     "near-miss: a single sweep (no re-run) leaves the same pair untagged") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  ValueId dead = b.emit(OpKind::Relu, {lin}, {}, y, {}, "dead");
  ValueId live = b.emit(OpKind::Relu, {lin}, {}, y, {}, "live");
  (void)dead;
  b.setOutput(live);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  createLinearBiasActivationFusionPass()->run(g, ctx);  // one sweep only, matching the old single-pass behavior
  createDeadOpEliminationPass()->run(g, ctx);
  R.check(!hasFact(g, 0, "fused_linear_relu_0"),
          "without a second sweep, \"lin\" still has two consumers when fusion runs, so it is never tagged");
}

// --- LIVE-001: live intervals on a %1->%2->%3 chain. ---
CASE(LIVE_001, "LIVE-001", "live intervals on a %1->%2->%3 chain") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v2}, {}, tt, {}, "v3");
  b.setOutput(v3);
  TypedGraph g = b.finish();

  LivenessResult live = LivenessAnalysis().run(g);
  R.check(live.intervals.at(x).firstDef == kNoProducer && live.intervals.at(x).lastUse == 0,
          "x: live from graph entry to node 0 (v1's only use of it)");
  R.check(live.intervals.at(v1).firstDef == 0 && live.intervals.at(v1).lastUse == 1,
          "v1: defined at node 0, used at node 1");
  R.check(live.intervals.at(v2).firstDef == 1 && live.intervals.at(v2).lastUse == 2,
          "v2: defined at node 1, used at node 2");
  R.check(live.intervals.at(v3).firstDef == 2 && live.intervals.at(v3).lastUse == 3,
          "v3: defined at node 2, used at the virtual `return` position (numNodes=3)");
}

// --- LIVE-002: byte size of Tensor[4,256] = 4*256*4. ---
CASE(LIVE_002, "LIVE-002", "byte size of Tensor[batch=4,feat=256] is 4*256*4 = 4096 bytes") {
  TypeContext types;
  const TensorType* tt = t2(types, "batch", 4, "feat", 256);
  R.check(tensorByteSize(tt) == 4 * 256 * 4, "4*256 elements * 4 bytes/f32 = 4096");
}

// --- LIVE-003: peak-memory = max simultaneously-live, matches a hand calculation. ---
CASE(LIVE_003, "LIVE-003", "peak-memory equals the hand-computed max of simultaneously-live sizes") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);  // 2*3*4 = 24 bytes each
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");   // node 0, consumes x
  ValueId v2 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v2");   // node 1, also consumes x
  ValueId v3 = b.emit(OpKind::Add, {v1, v2}, {}, tt, {}, "v3");  // node 2, consumes both v1 and v2
  b.setOutput(v3);
  TypedGraph g = b.finish();

  LivenessResult live = LivenessAnalysis().run(g);
  // At node 2 (computing v3), v1 and v2 must both still be live (its
  // operands) and v3 itself becomes live: 3 tensors * 24 bytes = 72.
  // x is already dead by then (last used at node 1).
  R.check(live.peakLiveBytes == 3 * 24, "peak = 3 * 24 = 72 bytes (v1, v2, v3 simultaneously live)");
}

// --- LIVE-004: a decision consumes peakLiveBytes. ---
CASE(LIVE_004, "LIVE-004", "a real decision (exceedsMemoryBudget) consumes peakLiveBytes") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Add, {v1, v2}, {}, tt, {}, "v3");
  b.setOutput(v3);
  TypedGraph g = b.finish();

  LivenessResult live = LivenessAnalysis().run(g);  // peak = 72 bytes (LIVE-003)
  R.check(exceedsMemoryBudget(live, 50), "a 50-byte budget is exceeded by a 72-byte peak");
  R.check(!exceedsMemoryBudget(live, 100), "a 100-byte budget is not exceeded by a 72-byte peak");
}

#undef CASE

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
