// Phase 3 semantic tests (RM 3.4-3.19; ARCHITECTURE.md §6). These cover
// every required course check and every Tensora-specific operator
// contract. Each case has its own name (SEM-*, ATT-*, BND-*), matching
// docs/testing/TEST_MATRIX.md. See tests/support/TestHarness.h for the
// harness (shared with test_lexer.cpp/test_parser.cpp).
//
// Cases that only check ok/reject-plus-DiagCode (no AST/type structure
// inspection) live in tests/fixtures/semantic.tns instead, one real
// .tns program per case -- tools/gen_fixture_cases.py expands that file
// into a generated TEST_CASE(...) per entry (see cmake/Tests.cmake).
// Cases that need a real C++ assertion on the resulting type/symbol
// table stay here, hand-written.
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

const TensorType* asTensor(const Type* t) {
  return t->kind() == Type::Kind::Tensor ? static_cast<const TensorType*>(t) : nullptr;
}
bool isConst(const Axis& a, const std::string& name, int64_t v) {
  return a.name == name && a.size.kind == Dimension::Kind::Const && a.size.constant == v;
}
bool isSymbol(const Axis& a, const std::string& name) {
  return a.name == name && a.size.kind == Dimension::Kind::Symbol;
}
bool firstDiagIs(const DiagnosticEngine& d, DiagCode code) {
  return !d.all().empty() && d.all()[0].code == code;
}

// ============================== SEM-SCOPE ==============================

CASE(SEM_SCOPE_001, "SEM-SCOPE-001", "rebinding h = relu(h) shadows the prior binding (accept)") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("input x: Tensor[a=1]\ngraph g {\n  h = relu(x)\n  h = relu(h)\n  return h\n}\n", types, d);
  R.check(r.ok(), "rebinding within a graph is accepted");
  R.check(d.all().empty(), "no diagnostics");
}

// ============================== SEM-POISON ==============================

CASE(SEM_POISON_001, "SEM-POISON-001", "one bad statement does not cascade into spurious follow-on errors") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[a=1,b=2]\n"
      "graph g {\n"
      "  bad = undefined_name\n"
      "  ok1 = relu(x)\n"
      "  ok2 = relu(ok1)\n"
      "  return ok2\n"
      "}\n",
      types, d);
  R.check(!r.ok(), "the program is rejected overall");
  R.check(d.all().size() == 1, "exactly one diagnostic: the poisoned value never causes a second error");
}

CASE(SEM_POISON_002, "SEM-POISON-002", "two independent errors in different graphs are both reported") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("graph a { return undefined1 }\ngraph b { return undefined2 }\n", types, d);
  R.check(!r.ok(), "the program is rejected overall");
  R.check(d.all().size() == 2, "both independent undefined-name errors are reported");
}

// ============================== SEM-MM / SEM-LIN ==============================

CASE(SEM_MM_001, "SEM-MM-001", "matmul: [batch=B,feat=784] @ [in=784,out=256] -> [batch=B,out=256]") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,feat=784]\nweight w: Tensor[in=784,out=256]\n"
      "graph g {\n  y = x @ w\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "valid matmul accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && yType->rank() == 2 && isSymbol(yType->axes()[0], "batch") && isConst(yType->axes()[1], "out", 256),
          "result is Tensor[batch=B, out=256]");
}

CASE(SEM_LIN_001, "SEM-LIN-001", "linear with a matching bias is accepted") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,in=784]\nweight w: Tensor[in=784,out=256]\nweight b: Tensor[out=256]\n"
      "graph g {\n  y = linear(x, w, b)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "valid linear accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && yType->rank() == 2 && isConst(yType->axes()[1], "out", 256), "result trailing axis is out=256");
}

CASE(SEM_EW_002, "SEM-EW-002", "elementwise: size-1 broadcast is accepted") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input a: Tensor[batch=B,feat=4]\ninput b: Tensor[batch=B,feat=1]\ngraph g {\n  y = a + b\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "size-1 broadcast axis accepted regardless of name");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[1], "feat", 4), "result takes the non-1 side's axis");
}

CASE(SEM_EW_005, "SEM-EW-005", "elementwise: tensor-scalar x * 0.125 imposes no constraint") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("input x: Tensor[a=1,b=2]\ngraph g {\n  y = x * 0.125\n  return y\n}\n", types, d);
  R.check(r.ok(), "tensor-scalar multiply accepted");
  const TypedGraph& g = r.value().graphs()[0];
  R.check(typesEqual(g.value(g.output).type, g.value(g.params[0]).type), "result type equals x's type exactly");
}

// ============================== SEM-RS / SEM-FL ==============================

CASE(SEM_RS_001, "SEM-RS-001", "reshape: 768 -> [head=12,dim=64] is accepted") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("input x: Tensor[feat=768]\ngraph g {\n  y = reshape(x, [head=12,dim=64])\n  return y\n}\n",
                    types, d);
  R.check(r.ok(), "12*64 == 768 accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && yType->rank() == 2 && isConst(yType->axes()[0], "head", 12) &&
               isConst(yType->axes()[1], "dim", 64),
           "result is exactly the target axis list");
}

CASE(SEM_FL_001, "SEM-FL-001", "flatten: concrete tail flattens to flat=200704") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,channel=64,height=56,width=56]\ngraph g {\n  y = flatten(x)\n  return y\n}\n", types,
      d);
  R.check(r.ok(), "concrete tail flattens");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && yType->rank() == 2 && isSymbol(yType->axes()[0], "batch") &&
               isConst(yType->axes()[1], "flat", 64 * 56 * 56),
           "result is Tensor[batch=B, flat=200704]");
}

// ============================== SEM-TR / SEM-RD / SEM-SM ==============================

CASE(SEM_TR_001, "SEM-TR-001", "transpose: a valid perm on rank-4 is accepted") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[a=1,b=2,c=3,d=4]\ngraph g {\n  y = transpose(x, perm=[0,1,3,2])\n  return y\n}\n", types, d);
  R.check(r.ok(), "valid perm accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[2], "d", 4) && isConst(yType->axes()[3], "c", 3),
          "axes reordered per perm");
}

CASE(SEM_RD_001, "SEM-RD-001", "sum(x, axis=feat): an existing axis is removed") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("input x: Tensor[batch=B,feat=4]\ngraph g {\n  y = sum(x, axis=feat)\n  return y\n}\n", types, d);
  R.check(r.ok(), "existing axis accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && yType->rank() == 1 && isSymbol(yType->axes()[0], "batch"), "feat axis removed");
}

CASE(SEM_SM_001, "SEM-SM-001", "softmax(scores, axis=kseq) is accepted, shape unchanged") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input scores: Tensor[batch=B,qseq=S,kseq=S]\ngraph g {\n  y = softmax(scores, axis=kseq)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "existing axis accepted");
  const TypedGraph& g = r.value().graphs()[0];
  R.check(typesEqual(g.value(g.output).type, g.value(g.params[0]).type), "softmax's result type equals input type");
}

// ============================== SEM-CV / SEM-PL ==============================

CASE(SEM_CV_001, "SEM-CV-001", "conv2d stride=1,padding=1 preserves HxW") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,channel=3,height=224,width=224]\n"
      "weight w: Tensor[out_channel=32,in_channel=3,kh=3,kw=3]\nweight b: Tensor[out_channel=32]\n"
      "graph g {\n  y = conv2d(x, w, b, stride=1, padding=1)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "valid conv2d accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[1], "channel", 32) && isConst(yType->axes()[2], "height", 224) &&
               isConst(yType->axes()[3], "width", 224),
           "output is [batch=B, channel=32, height=224, width=224]");
}

CASE(SEM_CV_005, "SEM-CV-005", "conv2d: stride=2 at 224x224 exactly halves H and W when the division is exact") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,channel=3,height=224,width=224]\n"
      "weight w: Tensor[out_channel=64,in_channel=3,kh=4,kw=4]\nweight b: Tensor[out_channel=64]\n"
      "graph g {\n  y = conv2d(x, w, b, stride=2, padding=1)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "(224+2-4)/2 = 222/2 = 111, +1 = 112: exact, accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[1], "channel", 64) && isConst(yType->axes()[2], "height", 112) &&
               isConst(yType->axes()[3], "width", 112),
           "output is [batch=B, channel=64, height=112, width=112]");
}

CASE(SEM_PL_001, "SEM-PL-001", "maxpool size=2 halves HxW") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,channel=32,height=112,width=112]\ngraph g {\n  y = maxpool(x, size=2)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "valid maxpool accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[2], "height", 56) && isConst(yType->axes()[3], "width", 56),
          "112 halved to 56");
}

// ============================== ATT (whitepaper 10.4) ==============================

CASE(ATT_001, "ATT-001", "the full attention program (whitepaper 10.4) type-checks end to end") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input Q: Tensor[batch=B, head=12, qseq=S, dim=64]\n"
      "input K: Tensor[batch=B, head=12, kseq=S, dim=64]\n"
      "input V: Tensor[batch=B, head=12, kseq=S, dim=64]\n"
      "\n"
      "graph attention {\n"
      "    kt      = transpose(K, perm=[0, 1, 3, 2])\n"
      "    scores  = Q @ kt\n"
      "    scores  = scores * 0.125\n"
      "    weights = softmax(scores, axis=kseq)\n"
      "    out     = weights @ V\n"
      "    return out\n"
      "}\n",
      types, d);
  R.check(r.ok(), "whitepaper's attention program type-checks with no diagnostics");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* outType = asTensor(g.value(g.output).type);
  R.check(outType && outType->rank() == 4 && isConst(outType->axes()[1], "head", 12) &&
               isConst(outType->axes()[3], "dim", 64),
           "out : Tensor[batch=B, head=12, qseq=S, dim=64]");
}

CASE(BND_MM_002, "BND-MM-002", "matmul boundary: M=1 or N=1") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input a: Tensor[m=1,k=4]\ninput b: Tensor[k=4,n=4]\ngraph g {\n  y = a @ b\n  return y\n}\n", types, d);
  R.check(r.ok(), "M=1 is valid");
  DiagnosticEngine d2;
  TypeContext types2;
  auto r2 = analyze(
      "input a: Tensor[m=4,k=4]\ninput b: Tensor[k=4,n=1]\ngraph g {\n  y = a @ b\n  return y\n}\n", types2, d2);
  R.check(r2.ok(), "N=1 is valid");
}

CASE(BND_BC_001, "BND-BC-001", "broadcast boundary: size-1 on each side (of different axes)") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input a: Tensor[batch=1,feat=4]\ninput b: Tensor[batch=8,feat=1]\ngraph g {\n  y = a + b\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "a broadcasts on batch, b broadcasts on feat: both accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[0], "batch", 8) && isConst(yType->axes()[1], "feat", 4),
          "result takes each side's non-1 axis");
}

CASE(BND_CV_001, "BND-CV-001", "conv2d boundary: output exact-divide vs remainder") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,channel=1,height=4,width=4]\n"
      "weight w: Tensor[out_channel=1,in_channel=1,kh=2,kw=2]\nweight b: Tensor[out_channel=1]\n"
      "graph g {\n  y = conv2d(x, w, b, stride=2, padding=0)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "(4-2)/2+1 = 2 exactly: accepted");

  TypeContext types2;
  DiagnosticEngine d2;
  auto r2 = analyze(
      "input x: Tensor[batch=B,channel=1,height=5,width=5]\n"
      "weight w: Tensor[out_channel=1,in_channel=1,kh=2,kw=2]\nweight b: Tensor[out_channel=1]\n"
      "graph g {\n  y = conv2d(x, w, b, stride=2, padding=0)\n  return y\n}\n",
      types2, d2);
  R.check(!r2.ok(), "(5-2)/2 = 1.5: remainder is rejected");
  R.check(firstDiagIs(d2, DiagCode::SemaConvArithmetic), "reports SemaConvArithmetic");
}

CASE(BND_CV_002, "BND-CV-002", "conv2d boundary: kernel size == input size (output spatial size 1)") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,channel=1,height=5,width=5]\n"
      "weight w: Tensor[out_channel=1,in_channel=1,kh=5,kw=5]\nweight b: Tensor[out_channel=1]\n"
      "graph g {\n  y = conv2d(x, w, b, stride=1, padding=0)\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "kernel == input size accepted");
  const TypedGraph& g = r.value().graphs()[0];
  const TensorType* yType = asTensor(g.value(g.output).type);
  R.check(yType && isConst(yType->axes()[2], "height", 1) && isConst(yType->axes()[3], "width", 1),
          "output spatial size is 1");
}

CASE(BND_RD_001, "BND-RD-001", "reduce boundary: axis 0, last axis, and the only axis") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("input x: Tensor[a=1,b=2,c=3]\ngraph g {\n  y = sum(x, axis=a)\n  return y\n}\n", types, d);
  R.check(r.ok(), "reducing axis 0 accepted");

  TypeContext types2;
  DiagnosticEngine d2;
  auto r2 = analyze("input x: Tensor[a=1,b=2,c=3]\ngraph g {\n  y = sum(x, axis=c)\n  return y\n}\n", types2, d2);
  R.check(r2.ok(), "reducing the last axis accepted");

  TypeContext types3;
  DiagnosticEngine d3;
  auto r3 = analyze("input x: Tensor[a=1]\ngraph g {\n  y = sum(x, axis=a)\n  return y\n}\n", types3, d3);
  R.check(!r3.ok(), "reducing the only axis leaves rank 0, which TensorType (RM 3.3 non-empty axes) cannot express");
  R.check(firstDiagIs(d3, DiagCode::SemaRankMismatch), "reports SemaRankMismatch");
}

CASE(BND_TR_001, "BND-TR-001", "transpose boundary: identity perm and full-reverse perm") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze("input x: Tensor[a=1,b=2,c=3]\ngraph g {\n  y = transpose(x, perm=[0,1,2])\n  return y\n}\n",
                    types, d);
  R.check(r.ok(), "identity perm accepted");
  const TypedGraph& g1 = r.value().graphs()[0];
  R.check(typesEqual(g1.value(g1.output).type, g1.value(g1.params[0]).type), "identity perm leaves the type unchanged");

  TypeContext types2;
  DiagnosticEngine d2;
  auto r2 = analyze("input x: Tensor[a=1,b=2,c=3]\ngraph g {\n  y = transpose(x, perm=[2,1,0])\n  return y\n}\n",
                     types2, d2);
  R.check(r2.ok(), "full-reverse perm accepted");
  const TypedGraph& g2 = r2.value().graphs()[0];
  const TensorType* yType = asTensor(g2.value(g2.output).type);
  R.check(yType && isConst(yType->axes()[0], "c", 3) && isConst(yType->axes()[2], "a", 1),
          "axes fully reversed");
}

CASE(BND_DIM_001, "BND-DIM-001", "bounded solver boundary: just inside vs just outside the affine subset") {
  TypeContext types;
  DiagnosticEngine d;
  auto r = analyze(
      "input x: Tensor[batch=B,feat=768]\ngraph g {\n  y = reshape(x, [batch=B,head=12,dim=64])\n  return y\n}\n",
      types, d);
  R.check(r.ok(), "just inside: the same symbol B cancels on both sides");

  TypeContext types2;
  DiagnosticEngine d2;
  auto r2 = analyze(
      "input x: Tensor[batch=B,feat=768]\ngraph g {\n  y = reshape(x, [batch=C,head=12,dim=64])\n  return y\n}\n",
      types2, d2);
  R.check(!r2.ok(), "just outside: B vs an unrelated C cannot be decided");
  R.check(firstDiagIs(d2, DiagCode::SemaUnresolvableShape), "reports SemaUnresolvableShape");
}

// ============================== SYM (symbol-table report) ==============================
// These exercise SemanticAnalyzer::symbolTable() (--emit-symbols'
// source of truth) directly, so they construct SemanticAnalyzer
// themselves instead of using the `analyze()` helper above, which
// discards the analyzer once it returns.

CASE(SYM_001, "SYM-001", "globals appear in declaration order with correct kind and type") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns", "input x: Tensor[batch=4, feat=8]\nweight W: Tensor[in=8, out=2]\n"
                               "graph g { y = linear(x, W, W)\n return y }\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  analyzer.analyze(parsed.value());

  const SymbolTableReport& report = analyzer.symbolTable();
  R.check(report.globals.size() == 2, "two globals recorded");
  if (report.globals.size() == 2) {
    R.check(report.globals[0].name == "x" && report.globals[0].kind == Symbol::Kind::Input,
             "x recorded first, kind Input (declaration order)");
    R.check(report.globals[1].name == "W" && report.globals[1].kind == Symbol::Kind::Weight,
             "W recorded second, kind Weight");
    const TensorType* xt = asTensor(report.globals[0].type);
    R.check(xt && xt->rank() == 2 && isConst(xt->axes()[0], "batch", 4), "x's recorded type is its verified decl type");
  }
}

CASE(SYM_002, "SYM-002", "graph-local symbols are recorded with the resolved (inferred) type, per graph") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns", "input x: Tensor[batch=4, feat=8]\ngraph g {\n  h = relu(x)\n  return h\n}\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  analyzer.analyze(parsed.value());

  const SymbolTableReport& report = analyzer.symbolTable();
  R.check(report.graphs.size() == 1 && report.graphs[0].graphName == "g", "one graph, named g");
  if (!report.graphs.empty()) {
    const auto& locals = report.graphs[0].locals;
    R.check(locals.size() == 1 && locals[0].name == "h" && locals[0].kind == Symbol::Kind::Value,
             "h recorded as a local Value symbol");
    const TensorType* ht = locals.empty() ? nullptr : asTensor(locals[0].type);
    R.check(ht && isConst(ht->axes()[0], "batch", 4), "h's recorded type is relu's inferred (input-preserving) type");
  }
}

CASE(SYM_003, "SYM-003", "shadowing (h = relu(h)) collapses to one recorded local entry, not two") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns",
                    "input x: Tensor[batch=4, feat=8]\ngraph g {\n  h = relu(x)\n  h = relu(h)\n  return h\n}\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  Result<TypedProgram> r = analyzer.analyze(parsed.value());
  R.check(r.ok(), "shadowing is accepted");

  const auto& locals = analyzer.symbolTable().graphs.at(0).locals;
  R.check(locals.size() == 1, "one recorded entry for h, not one per binding");
}

CASE(SYM_004, "SYM-004", "the resolution log records global::/local::/undefined, deduped per name") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns",
                    "input x: Tensor[batch=4, feat=8]\ngraph g {\n  h = relu(x)\n  y = relu(h)\n  return y\n}\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  analyzer.analyze(parsed.value());

  const auto& res = analyzer.symbolTable().graphs.at(0).resolutions;
  R.check(res.size() == 3, "one entry per distinct referenced name (x, h, y), not one per resolveOperand call");
  if (res.size() == 3) {
    R.check(res[0].first == "x" && res[0].second == "global::x", "x resolves to the global scope");
    R.check(res[1].first == "h" && res[1].second == "local::h", "h resolves to the local scope");
    R.check(res[2].first == "y" && res[2].second == "local::y", "y (the return name) resolves to the local scope");
  }
}

CASE(SYM_005, "SYM-005", "an undefined reference logs 'undefined' and is not silently absent") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns", "graph g {\n  y = relu(does_not_exist)\n  return y\n}\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  Result<TypedProgram> r = analyzer.analyze(parsed.value());
  R.check(!r.ok(), "undefined name is rejected (no crash)");

  const auto& res = analyzer.symbolTable().graphs.at(0).resolutions;
  bool found = false;
  for (const auto& [name, resolvedAs] : res) {
    if (name == "does_not_exist") found = (resolvedAs == "undefined");
  }
  R.check(found, "does_not_exist is logged as undefined, not silently dropped");
}

CASE(SYM_006, "SYM-006", "a rejected redeclaration never enters the recorded global scope") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns", "input x: Tensor[a=1]\nweight x: Tensor[a=1]\ngraph g { return x }\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  Result<TypedProgram> r = analyzer.analyze(parsed.value());
  R.check(!r.ok(), "the redeclaration is rejected");

  const auto& globals = analyzer.symbolTable().globals;
  R.check(globals.size() == 1, "only the first declaration of x is recorded, not a duplicate entry");
  if (globals.size() == 1) R.check(globals[0].kind == Symbol::Kind::Input, "the surviving entry is the original Input");
}

CASE(SYM_007, "SYM-007", "a rejected local rebind of a global name never enters the recorded local scope") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns", "input x: Tensor[a=1]\ngraph g {\n  x = relu(x)\n  return x\n}\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  Result<TypedProgram> r = analyzer.analyze(parsed.value());
  R.check(!r.ok(), "shadowing a global locally is rejected");

  const auto& locals = analyzer.symbolTable().graphs.at(0).locals;
  R.check(locals.empty(), "the rejected local 'x' binding is not recorded (KindConflict never reaches bindLocal's insert)");
}

CASE(SYM_008, "SYM-008", "multiple graphs each get their own recorded scope, in program order") {
  TypeContext types;
  DiagnosticEngine d;
  SourceBuffer buf("test.tns",
                    "input x: Tensor[a=1]\ngraph first { p = relu(x)\n return p }\ngraph second { q = relu(x)\n "
                    "return q }\n");
  Parser parser(buf, d);
  Result<ast::AstProgram> parsed = parser.parseProgram();
  SemanticAnalyzer analyzer(types, d);
  Result<TypedProgram> r = analyzer.analyze(parsed.value());
  R.check(r.ok(), "both graphs are valid");

  const auto& graphs = analyzer.symbolTable().graphs;
  R.check(graphs.size() == 2, "two recorded graph scopes");
  if (graphs.size() == 2) {
    R.check(graphs[0].graphName == "first" && graphs[0].locals.size() == 1 && graphs[0].locals[0].name == "p",
             "first graph's scope has only its own local p");
    R.check(graphs[1].graphName == "second" && graphs[1].locals.size() == 1 && graphs[1].locals[0].name == "q",
             "second graph's scope has only its own local q");
  }
}

#undef CASE

}  // namespace

int main(int argc, char** argv) {
  tensora::testing::registerFixtureCases(FIXTURE_SEMANTIC_PATH, [](const std::string& src) {
    TypeContext types;
    DiagnosticEngine d;
    auto r = analyze(src, types, d);
    std::string diag = d.all().empty() ? std::string() : tensora::testing::diagCodeName(d.all()[0].code);
    return std::make_pair(r.ok(), diag);
  });
  return tensora::testing::runRegisteredCases(argc, argv);
}
