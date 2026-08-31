// Typed Computation Graph tests (INTERFACES.md §3a). These check GraphBuilder and
// GraphVerifier rules (GR-*), matching docs/testing/TEST_MATRIX.md.
// Each case has its own name; see tests/support/TestHarness.h.
#include <string>
#include <vector>

#include "tensora/parser/Parser.h"
#include "tensora/semantic/SemanticAnalyzer.h"
#include "support/FixtureLoader.h"
#include "support/TestHarness.h"

using namespace tensora;
using tensora::testing::Reporter;
#define CASE TEST_CASE

namespace {

Result<TypedProgram> analyze(const std::string& code, TypeContext& types, DiagnosticEngine& diags) {
  SourceBuffer buf("test.tns", code);
  Parser parser(buf, diags);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, diags);
  return analyzer.analyze(parsed.value());
}

const TensorType* t1(TypeContext& types, const char* axisName, int64_t size) {
  return types.tensor({Axis{axisName, Dimension::makeConst(size)}});
}

}  // namespace

// --- GR-001: MLP program analyzed end to end -> golden structural TCG. ---
CASE(GR_001, "GR-001", "MLP program's TypedGraph has the expected structural shape") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(tensora::testing::loadFixtureSource(FIXTURE_GRAPH_PATH, "GR-001"), types, d);
  R.check(r.ok(), "MLP analyzes with no diagnostics");
  const TypedGraph& g = r.value().graphs()[0];
  R.check(g.params.size() == 5, "5 params (x, W1, b1, W2, b2)");
  R.check(g.nodes().size() == 3, "3 nodes (linear, relu, linear)");
  if (g.nodes().size() == 3) {
    R.check(g.nodes()[0].op == OpKind::Linear, "node 0 is linear");
    R.check(g.nodes()[1].op == OpKind::Relu, "node 1 is relu");
    R.check(g.nodes()[2].op == OpKind::Linear, "node 2 is linear");
    R.check(g.output == g.nodes()[2].output, "graph output is the last linear's result");
  }
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "sema's own output is GraphVerifier-clean");
}

// --- GR-002: a hand-built fixture with a cycle is rejected. ---
CASE(GR_002, "GR-002", "GraphVerifier rejects a hand-built fixture with a cycle") {
  TypeContext types;
  GraphBuilder b("cyclic");
  const TensorType* tt = t1(types, "a", 4);
  ValueId x = b.addInput("x", tt, {});
  // node0's input is ValueId 2 (node1's future output), fabricated before
  // node1 exists -- a forward reference, i.e. a cycle in producer order.
  ValueId v0 = b.emit(OpKind::Relu, {2}, {}, tt, {}, "v0");
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  (void)v0;
  b.setOutput(v1);
  TypedGraph g = b.finish();

  GraphVerifier verifier;
  DiagnosticList diags = verifier.verify(g);
  R.check(!diags.empty(), "the cycle is detected");
  bool anyIr = false;
  for (const Diagnostic& diag : diags) anyIr |= diag.code == DiagCode::IrInvariantViolation;
  R.check(anyIr, "reports IrInvariantViolation");
}

// --- GR-003: a hand-built fixture with a dangling operand is rejected. ---
CASE(GR_003, "GR-003", "GraphVerifier rejects a hand-built fixture with a dangling operand") {
  TypeContext types;
  GraphBuilder b("dangling");
  const TensorType* tt = t1(types, "a", 4);
  ValueId v0 = b.emit(OpKind::Relu, {9999}, {}, tt, {}, "v0");  // 9999 does not exist
  b.setOutput(v0);
  TypedGraph g = b.finish();

  GraphVerifier verifier;
  DiagnosticList diags = verifier.verify(g);
  R.check(!diags.empty(), "the dangling operand is detected");
  bool anyIr = false;
  for (const Diagnostic& diag : diags) anyIr |= diag.code == DiagCode::IrInvariantViolation;
  R.check(anyIr, "reports IrInvariantViolation");
}

// --- GR-004: producer/consumer queries correct on a hand-built chain. ---
CASE(GR_004, "GR-004", "producer/consumer queries are correct on a %1->%2->%3 chain") {
  TypeContext types;
  GraphBuilder b("chain");
  const TensorType* tt = t1(types, "a", 4);
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v2}, {}, tt, {}, "v3");
  b.setOutput(v3);
  TypedGraph g = b.finish();

  R.check(g.producerOf(x) == kNoProducer, "x (a param) has no producer");
  R.check(g.producerOf(v1) == 0 && g.producerOf(v2) == 1 && g.producerOf(v3) == 2, "producer node ids are 0,1,2");
  R.check(g.consumersOf(x).size() == 1 && g.consumersOf(x)[0] == 0, "x is consumed by node 0");
  R.check(g.consumersOf(v1).size() == 1 && g.consumersOf(v1)[0] == 1, "v1 is consumed by node 1");
  R.check(g.consumersOf(v3).empty(), "v3 (the output) has no consumers");

  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "the chain is GraphVerifier-clean");
}

// --- GR-005: single reachable output; nodes stored in topological order. ---
CASE(GR_005, "GR-005", "a valid graph has a single valid output and nodes in topological order") {
  TypeContext types;
  GraphBuilder b("g");
  const TensorType* tt = t1(types, "a", 4);
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  b.setOutput(v1);
  TypedGraph g = b.finish();

  R.check(g.output < g.values().size(), "output is a valid ValueId");
  R.check(g.nodes()[0].id == 0, "the single node is at position 0 (topological order trivially holds)");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "verifier accepts");
}

// --- GR-006: a GraphVerifier-clean fixture built entirely by hand passes (ARCH-008). ---
CASE(GR_006, "GR-006", "a hand-built fixture (no sema) is GraphVerifier-clean, demonstrating ARCH-008") {
  TypeContext types;
  GraphBuilder b("handbuilt_mlp");
  const TensorType* x = types.tensor({Axis{"batch", Dimension::makeSymbol(types.internSymbol("B"))},
                                      Axis{"feat", Dimension::makeConst(784)}});
  const TensorType* w1 = types.tensor({Axis{"in", Dimension::makeConst(784)}, Axis{"out", Dimension::makeConst(256)}});
  const TensorType* b1 = types.tensor({Axis{"out", Dimension::makeConst(256)}});
  const TensorType* h = types.tensor({Axis{"batch", Dimension::makeSymbol(types.internSymbol("B"))},
                                      Axis{"out", Dimension::makeConst(256)}});

  ValueId xId = b.addInput("x", x, {});
  ValueId w1Id = b.addWeight("W1", w1, {});
  ValueId b1Id = b.addWeight("b1", b1, {});
  ValueId hId = b.emit(OpKind::Linear, {xId, w1Id, b1Id}, {}, h, {}, "h");
  ValueId reluId = b.emit(OpKind::Relu, {hId}, {}, h, {}, "h_relu");
  b.setOutput(reluId);
  TypedGraph g = b.finish();

  GraphVerifier verifier;
  DiagnosticList diags = verifier.verify(g);
  R.check(diags.empty(), "a fixture built without sema obeys the same invariants as a sema-produced graph");
  R.check(g.params.size() == 3 && g.nodes().size() == 2, "3 params, 2 nodes as built");
}

// --- GR-007: a graph referencing the same global input twice adds it as exactly one param. ---
CASE(GR_007, "GR-007", "referencing the same global input twice adds exactly one param (lazy dedup)") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(tensora::testing::loadFixtureSource(FIXTURE_GRAPH_PATH, "GR-007"), types, d);
  R.check(r.ok(), "program analyzes with no diagnostics");
  const TypedGraph& g = r.value().graphs()[0];
  R.check(g.params.size() == 1, "x is added as a param exactly once");
  if (g.params.size() == 1) {
    R.check(g.consumersOf(g.params[0]).size() == 2, "x has two consumers (both relu nodes)");
  }
}

#undef CASE

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
