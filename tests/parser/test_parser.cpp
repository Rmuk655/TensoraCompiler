// Phase 2 parser tests (RM 3.2, 3.3, 3.24). These cover every grammar
// production and alternative, error and recovery paths, and correct
// SourceSpan/diagnostic values. Each case has its own name (PAR-*,
// BND-PAR-*), matching the stable IDs in docs/testing/TEST_MATRIX.md.
// See tests/support/TestHarness.h for the harness (shared with
// test_lexer.cpp).
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "tensora/parser/Parser.h"
#include "support/TestHarness.h"

using namespace tensora;
using tensora::testing::Reporter;
#define CASE TEST_CASE

namespace {

// Parser lexes and parses source text directly (no intermediate project
// Token vector -- see Parser.h); the SourceBuffer need not outlive the
// returned AST (AST nodes own std::string copies, not string_views), so a
// local per-call buffer is fine.
Result<ast::AstProgram> parse(const std::string& code, DiagnosticEngine& diags) {
  SourceBuffer buf("test.tns", code);
  Parser parser(buf, diags);
  return parser.parseProgram();
}

const ast::NameExpr* asName(const ast::Expr* e) { return dynamic_cast<const ast::NameExpr*>(e); }
const ast::ScalarExpr* asScalar(const ast::Expr* e) { return dynamic_cast<const ast::ScalarExpr*>(e); }
const ast::MatMulExpr* asMatMul(const ast::Expr* e) { return dynamic_cast<const ast::MatMulExpr*>(e); }
const ast::BinaryExpr* asBinary(const ast::Expr* e) { return dynamic_cast<const ast::BinaryExpr*>(e); }
const ast::CallExpr* asCall(const ast::Expr* e) { return dynamic_cast<const ast::CallExpr*>(e); }

bool isConstAxis(const ast::AxisSyntax& a, const std::string& name, int64_t v) {
  return a.name == name && a.size.k == ast::DimSyntax::K::Const && a.size.constant == v;
}
bool isSymbolAxis(const ast::AxisSyntax& a, const std::string& name, const std::string& sym) {
  return a.name == name && a.size.k == ast::DimSyntax::K::Symbol && a.size.symbol == sym;
}

// --- PAR-001: MLP program (whitepaper 10.1) -> AST golden. ---
CASE(PAR_001, "PAR-001", "MLP program (whitepaper 10.1) parses to the expected AST") {
  const std::string src =
      "input  x:  Tensor[batch=B, feat=784]\n"
      "weight W1: Tensor[in=784, out=256]\n"
      "weight b1: Tensor[out=256]\n"
      "weight W2: Tensor[in=256, out=10]\n"
      "weight b2: Tensor[out=10]\n"
      "\n"
      "graph mlp {\n"
      "    h = linear(x, W1, b1)\n"
      "    h = relu(h)\n"
      "    y = linear(h, W2, b2)\n"
      "    return y\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "MLP program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 5, "MLP has 5 decls (1 input + 4 weights)");
  if (p.decls.size() == 5) {
    R.check(p.decls[0].kind == ast::Decl::Kind::Input && p.decls[0].name == "x", "decl 0 is input x");
    R.check(p.decls[0].type.axes.size() == 2 && isSymbolAxis(p.decls[0].type.axes[0], "batch", "B") &&
                 isConstAxis(p.decls[0].type.axes[1], "feat", 784),
             "x : Tensor[batch=B, feat=784]");
    R.check(p.decls[1].kind == ast::Decl::Kind::Weight && p.decls[1].name == "W1", "decl 1 is weight W1");
  }
  R.check(p.graphs.size() == 1 && p.graphs[0].name == "mlp", "one graph named mlp");
  if (p.graphs.size() == 1) {
    const ast::GraphDecl& g = p.graphs[0];
    R.check(g.stmts.size() == 3, "mlp has 3 statements");
    R.check(g.returnName == "y", "mlp returns y");
    if (g.stmts.size() == 3) {
      const auto* c0 = asCall(g.stmts[0].rhs.get());
      R.check(g.stmts[0].name == "h" && c0 && c0->callee == "linear" && c0->args.size() == 3,
              "h = linear(x, W1, b1)");
      const auto* c1 = asCall(g.stmts[1].rhs.get());
      R.check(g.stmts[1].name == "h" && c1 && c1->callee == "relu" && c1->args.size() == 1,
              "h = relu(h)");
      const auto* c2 = asCall(g.stmts[2].rhs.get());
      R.check(g.stmts[2].name == "y" && c2 && c2->callee == "linear" && c2->args.size() == 3,
              "y = linear(h, W2, b2)");
    }
  }
}

// --- PAR-002: CNN program (whitepaper 10.2) -> AST golden. ---
CASE(PAR_002, "PAR-002", "CNN program (whitepaper 10.2) parses to the expected AST") {
  const std::string src =
      "input  image: Tensor[batch=B, channel=3, height=224, width=224]\n"
      "weight Wc1:   Tensor[out_channel=32, in_channel=3, kh=3, kw=3]\n"
      "weight bc1:   Tensor[out_channel=32]\n"
      "weight Wc2:   Tensor[out_channel=64, in_channel=32, kh=3, kw=3]\n"
      "weight bc2:   Tensor[out_channel=64]\n"
      "weight Wfc:   Tensor[in=200704, out=10]\n"
      "weight bfc:   Tensor[out=10]\n"
      "\n"
      "graph cnn {\n"
      "    x = conv2d(image, Wc1, bc1, stride=1, padding=1)\n"
      "    x = relu(x)\n"
      "    x = maxpool(x, size=2)\n"
      "    x = conv2d(x, Wc2, bc2, stride=1, padding=1)\n"
      "    x = relu(x)\n"
      "    x = maxpool(x, size=2)\n"
      "    f = flatten(x)\n"
      "    y = linear(f, Wfc, bfc)\n"
      "    return y\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "CNN program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 7, "CNN has 7 decls (1 input + 6 weights)");
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 8 && p.graphs[0].returnName == "y",
          "cnn has 8 statements and returns y");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 8) {
    const auto& stmts = p.graphs[0].stmts;
    const auto* conv0 = asCall(stmts[0].rhs.get());
    R.check(conv0 && conv0->callee == "conv2d" && conv0->args.size() == 5,
            "first conv2d call has 5 args (3 positional + stride= + padding=)");
    if (conv0 && conv0->args.size() == 5) {
      R.check(!conv0->args[0].keyword.has_value() && conv0->args[0].kind == ast::Arg::Kind::Name &&
                   conv0->args[0].name == "image",
               "conv2d arg 0 is positional image");
      R.check(conv0->args[3].keyword.has_value() && *conv0->args[3].keyword == "stride" &&
                   conv0->args[3].kind == ast::Arg::Kind::Int && conv0->args[3].intVal == 1,
               "conv2d arg 3 is stride=1");
      R.check(conv0->args[4].keyword.has_value() && *conv0->args[4].keyword == "padding" &&
                   conv0->args[4].intVal == 1,
               "conv2d arg 4 is padding=1");
    }
    const auto* pool0 = asCall(stmts[2].rhs.get());
    R.check(pool0 && pool0->callee == "maxpool" && pool0->args.size() == 2 &&
                 pool0->args[1].keyword.has_value() && *pool0->args[1].keyword == "size" &&
                 pool0->args[1].intVal == 2,
             "maxpool(x, size=2)");
    const auto* flat = asCall(stmts[6].rhs.get());
    R.check(flat && flat->callee == "flatten" && flat->args.size() == 1, "flatten(x) has 1 arg");
  }
}

// --- PAR-003: Transformer FFN program (whitepaper 10.3) -> AST golden. ---
CASE(PAR_003, "PAR-003", "Transformer FFN program (whitepaper 10.3) parses to the expected AST") {
  const std::string src =
      "input  x:  Tensor[batch=B, seq=S, model=768]\n"
      "weight W1: Tensor[in=768, out=3072]\n"
      "weight b1: Tensor[out=3072]\n"
      "weight W2: Tensor[in=3072, out=768]\n"
      "weight b2: Tensor[out=768]\n"
      "\n"
      "graph transformer_ffn {\n"
      "    h = linear(x, W1, b1)\n"
      "    h = relu(h)\n"
      "    y = linear(h, W2, b2)\n"
      "    return y\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "FFN program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 5, "FFN has 5 decls");
  R.check(p.decls[0].type.axes.size() == 3 && isSymbolAxis(p.decls[0].type.axes[0], "batch", "B") &&
               isSymbolAxis(p.decls[0].type.axes[1], "seq", "S") &&
               isConstAxis(p.decls[0].type.axes[2], "model", 768),
           "x : Tensor[batch=B, seq=S, model=768] (two symbolic axes, one concrete)");
  R.check(p.graphs.size() == 1 && p.graphs[0].name == "transformer_ffn" &&
               p.graphs[0].stmts.size() == 3 && p.graphs[0].returnName == "y",
           "transformer_ffn has 3 statements and returns y");
}

// --- PAR-004: multi-head attention program (whitepaper 10.4) -> AST golden. ---
CASE(PAR_004, "PAR-004", "Attention program (whitepaper 10.4) parses to the expected AST") {
  const std::string src =
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
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "attention program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 3, "attention has 3 input decls, no weights");
  R.check(p.decls[0].type.axes.size() == 4, "Q has a rank-4 tensorType");
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 5 && p.graphs[0].returnName == "out",
          "attention has 5 statements and returns out");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 5) {
    const auto& s = p.graphs[0].stmts;
    const auto* transp = asCall(s[0].rhs.get());
    R.check(transp && transp->callee == "transpose" && transp->args.size() == 2 &&
                 transp->args[1].keyword.has_value() && *transp->args[1].keyword == "perm" &&
                 transp->args[1].kind == ast::Arg::Kind::IntList &&
                 transp->args[1].intList == std::vector<int64_t>{0, 1, 3, 2},
             "kt = transpose(K, perm=[0,1,3,2])");
    const auto* mm0 = asMatMul(s[1].rhs.get());
    R.check(mm0 && asName(mm0->lhs.get()) && asName(mm0->lhs.get())->name == "Q" &&
                 asName(mm0->rhs.get()) && asName(mm0->rhs.get())->name == "kt",
             "scores = Q @ kt is a MatMulExpr");
    const auto* bin0 = asBinary(s[2].rhs.get());
    R.check(bin0 && bin0->op == ast::BinaryExpr::Op::Mul && asScalar(bin0->rhs.get()) &&
                 asScalar(bin0->rhs.get())->value == 0.125,
             "scores = scores * 0.125 is a Mul BinaryExpr with a scalar operand");
    const auto* sm = asCall(s[3].rhs.get());
    R.check(sm && sm->callee == "softmax" && sm->args.size() == 2 &&
                 sm->args[1].keyword.has_value() && *sm->args[1].keyword == "axis" &&
                 sm->args[1].kind == ast::Arg::Kind::Name && sm->args[1].name == "kseq",
             "weights = softmax(scores, axis=kseq)");
    const auto* mm1 = asMatMul(s[4].rhs.get());
    R.check(mm1 != nullptr, "out = weights @ V is a MatMulExpr");
  }
}

// --- PAR-005: tensorType/axis -- 1, 2, and 4 axes; concrete and symbolic dimSize. ---
CASE(PAR_005, "PAR-005", "tensorType/axis: 1/2/4 axes, concrete and symbolic dimSize") {
  const std::string src =
      "input a: Tensor[x=1]\n"
      "input b: Tensor[x=2, y=3]\n"
      "input c: Tensor[n=B, m=4, k=C, j=8]\n"
      "graph g { return a }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 3, "three decls");
  if (p.decls.size() == 3) {
    R.check(p.decls[0].type.axes.size() == 1 && isConstAxis(p.decls[0].type.axes[0], "x", 1),
            "a: Tensor[x=1] (1 axis, concrete)");
    R.check(p.decls[1].type.axes.size() == 2 && isConstAxis(p.decls[1].type.axes[0], "x", 2) &&
                 isConstAxis(p.decls[1].type.axes[1], "y", 3),
             "b: Tensor[x=2, y=3] (2 axes, concrete)");
    R.check(p.decls[2].type.axes.size() == 4 && isSymbolAxis(p.decls[2].type.axes[0], "n", "B") &&
                 isConstAxis(p.decls[2].type.axes[1], "m", 4) &&
                 isSymbolAxis(p.decls[2].type.axes[2], "k", "C") &&
                 isConstAxis(p.decls[2].type.axes[3], "j", 8),
             "c: Tensor[n=B, m=4, k=C, j=8] (4 axes, alternating symbolic/concrete)");
  }
}

// --- PAR-006: decl kinds -- inputDecl alone, weightDecl alone, both, multiple of each. ---
CASE(PAR_006, "PAR-006", "decl: inputDecl and weightDecl, alone and combined, multiple of each") {
  const std::string src =
      "input a: Tensor[x=1]\n"
      "weight w: Tensor[x=1]\n"
      "input a2: Tensor[x=1]\n"
      "weight w2: Tensor[x=1]\n"
      "graph g { return a }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 4, "four decls in order");
  if (p.decls.size() == 4) {
    R.check(p.decls[0].kind == ast::Decl::Kind::Input && p.decls[0].name == "a", "decl 0: input a");
    R.check(p.decls[1].kind == ast::Decl::Kind::Weight && p.decls[1].name == "w", "decl 1: weight w");
    R.check(p.decls[2].kind == ast::Decl::Kind::Input && p.decls[2].name == "a2", "decl 2: input a2");
    R.check(p.decls[3].kind == ast::Decl::Kind::Weight && p.decls[3].name == "w2", "decl 3: weight w2");
  }
}

// --- PAR-007: program -- multiple graphDecls (graphDecl+ repetition). ---
CASE(PAR_007, "PAR-007", "program: multiple graphDecls (graphDecl+ repetition)") {
  const std::string src =
      "graph g1 { return a }\n"
      "graph g2 { x = a\n return x }\n"
      "graph g3 { return a }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program with three graphs parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.empty(), "zero decls (parser does not require any decl to exist)");
  R.check(p.graphs.size() == 3, "three graphDecls");
  if (p.graphs.size() == 3) {
    R.check(p.graphs[0].name == "g1" && p.graphs[0].stmts.empty(), "g1: no statements");
    R.check(p.graphs[1].name == "g2" && p.graphs[1].stmts.size() == 1, "g2: one statement");
    R.check(p.graphs[2].name == "g3", "g3 present");
  }
}

// --- PAR-008: statement -- two statements no terminator; a bare-identifier statement. ---
CASE(PAR_008, "PAR-008", "statement: sequential statements with no terminator, incl. bare-identifier rhs") {
  const std::string src =
      "graph g {\n"
      "  a = b\n"
      "  c = d\n"
      "  return c\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 2,
          "two statements parsed with no terminator between them");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 2) {
    const auto* n0 = asName(p.graphs[0].stmts[0].rhs.get());
    R.check(p.graphs[0].stmts[0].name == "a" && n0 && n0->name == "b",
            "a = b is a bare-identifier statement (NameExpr rhs)");
    const auto* n1 = asName(p.graphs[0].stmts[1].rhs.get());
    R.check(p.graphs[0].stmts[1].name == "c" && n1 && n1->name == "d", "c = d likewise");
  }
}

// --- PAR-009: expression/binop -- each of @ + - * / with identifier and float operand. ---
CASE(PAR_009, "PAR-009", "expression/binop: each of @ + - * / with identifier and float operand") {
  const std::string src =
      "graph g {\n"
      "  a1 = x @ y\n"
      "  a2 = x + y\n"
      "  a3 = x - 0.5\n"
      "  a4 = x * y\n"
      "  a5 = x / 0.25\n"
      "  return a5\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 5, "five binop statements");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 5) {
    const auto& s = p.graphs[0].stmts;
    R.check(asMatMul(s[0].rhs.get()) != nullptr, "a1 = x @ y is MatMulExpr");
    const auto* b1 = asBinary(s[1].rhs.get());
    R.check(b1 && b1->op == ast::BinaryExpr::Op::Add && asName(b1->rhs.get()), "a2 = x + y is Add(identifier)");
    const auto* b2 = asBinary(s[2].rhs.get());
    R.check(b2 && b2->op == ast::BinaryExpr::Op::Sub && asScalar(b2->rhs.get()) &&
                 asScalar(b2->rhs.get())->value == 0.5,
             "a3 = x - 0.5 is Sub(float)");
    const auto* b3 = asBinary(s[3].rhs.get());
    R.check(b3 && b3->op == ast::BinaryExpr::Op::Mul && asName(b3->rhs.get()), "a4 = x * y is Mul(identifier)");
    const auto* b4 = asBinary(s[4].rhs.get());
    R.check(b4 && b4->op == ast::BinaryExpr::Op::Div && asScalar(b4->rhs.get()) &&
                 asScalar(b4->rhs.get())->value == 0.25,
             "a5 = x / 0.25 is Div(float)");
  }
}

// --- PAR-010: callExpr -- one positional arg, multiple positional args. ---
CASE(PAR_010, "PAR-010", "callExpr: one positional arg, multiple positional args") {
  const std::string src =
      "graph g {\n"
      "  a = relu(x)\n"
      "  b = linear(x, w, bias)\n"
      "  return b\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 2, "two call statements");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 2) {
    const auto* c0 = asCall(p.graphs[0].stmts[0].rhs.get());
    R.check(c0 && c0->callee == "relu" && c0->args.size() == 1, "relu(x): 1 positional arg");
    const auto* c1 = asCall(p.graphs[0].stmts[1].rhs.get());
    R.check(c1 && c1->callee == "linear" && c1->args.size() == 3, "linear(x, w, bias): 3 positional args");
  }
}

// --- PAR-011: arg kinds -- keyword(int), keyword(identifier), reshape axis-list, all mixed. ---
CASE(PAR_011, "PAR-011", "arg kinds: keyword(int), keyword(identifier), axis-list, all mixed in one call") {
  const std::string src =
      "graph g {\n"
      "  a = f(x, stride=1, axis=nm, [p=1, q=2])\n"
      "  return a\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1, "one statement");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1) {
    const auto* c = asCall(p.graphs[0].stmts[0].rhs.get());
    R.check(c && c->args.size() == 4, "f(...) has 4 args mixing all arg kinds");
    if (c && c->args.size() == 4) {
      R.check(!c->args[0].keyword.has_value() && c->args[0].kind == ast::Arg::Kind::Name &&
                   c->args[0].name == "x",
               "arg 0: positional identifier x");
      R.check(c->args[1].keyword.has_value() && *c->args[1].keyword == "stride" &&
                   c->args[1].kind == ast::Arg::Kind::Int && c->args[1].intVal == 1,
               "arg 1: keyword int stride=1");
      R.check(c->args[2].keyword.has_value() && *c->args[2].keyword == "axis" &&
                   c->args[2].kind == ast::Arg::Kind::Name && c->args[2].name == "nm",
               "arg 2: keyword identifier axis=nm");
      R.check(!c->args[3].keyword.has_value() && c->args[3].kind == ast::Arg::Kind::AxisList &&
                   c->args[3].axisList.size() == 2 && isConstAxis(c->args[3].axisList[0], "p", 1) &&
                   isConstAxis(c->args[3].axisList[1], "q", 2),
               "arg 3: reshape-style axis-list [p=1, q=2]");
    }
  }
}

// --- PAR-012: argValue kinds -- intLit, identifier, intList. ---
CASE(PAR_012, "PAR-012", "argValue kinds: intLit (stride=1), identifier (axis=nm), intList (perm=[...])") {
  const std::string src =
      "graph g {\n"
      "  a = f(x, stride=1)\n"
      "  b = f(x, axis=nm)\n"
      "  c = f(x, perm=[0, 1, 3, 2])\n"
      "  return c\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 3, "three calls");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 3) {
    const auto* ca = asCall(p.graphs[0].stmts[0].rhs.get());
    R.check(ca && ca->args[1].kind == ast::Arg::Kind::Int && ca->args[1].intVal == 1, "stride=1 is Int");
    const auto* cb = asCall(p.graphs[0].stmts[1].rhs.get());
    R.check(cb && cb->args[1].kind == ast::Arg::Kind::Name && cb->args[1].name == "nm", "axis=nm is Name");
    const auto* cc = asCall(p.graphs[0].stmts[2].rhs.get());
    R.check(cc && cc->args[1].kind == ast::Arg::Kind::IntList &&
                 cc->args[1].intList == std::vector<int64_t>{0, 1, 3, 2},
             "perm=[0,1,3,2] is IntList");
  }
}

// --- PAR-013: conv2d-shaped call -- multiple positional + multiple keyword args together. ---
CASE(PAR_013, "PAR-013", "conv2d-shaped call: multiple positional + multiple keyword args together") {
  const std::string src =
      "graph g {\n"
      "  a = conv2d(image, W, b, stride=1, padding=1)\n"
      "  return a\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1, "one statement");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1) {
    const auto* c = asCall(p.graphs[0].stmts[0].rhs.get());
    R.check(c && c->args.size() == 5, "conv2d(...) has 5 args (3 positional + 2 keyword)");
  }
}

// --- PAR-014: negative -- missing '=' in a statement. ---
CASE(PAR_014, "PAR-014", "negative: missing '=' in a statement") {
  const std::string src = "graph g {\n  a b\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "malformed statement is rejected");
  R.check(d.all().size() >= 1 && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing '='");
}

// --- PAR-015: negative -- missing ']' closing a tensorType. ---
CASE(PAR_015, "PAR-015", "negative: missing ']' closing a tensorType") {
  const std::string src = "input x: Tensor[a=1\ngraph g { return x }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "unterminated tensorType is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing ']'");
}

// --- PAR-016: negative -- missing '[' opening a tensorType. ---
CASE(PAR_016, "PAR-016", "negative: missing '[' opening a tensorType") {
  const std::string src = "input x: Tensor a=1]\ngraph g { return x }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "tensorType missing '[' is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing '['");
}

// --- PAR-017: negative -- empty tensorType (axis list requires >= 1). ---
CASE(PAR_017, "PAR-017", "negative: empty tensorType Tensor[] (axis list requires >= 1)") {
  const std::string src = "input x: Tensor[]\ngraph g { return x }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "Tensor[] is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing first axis");
}

// --- PAR-018: negative -- missing '=' in an axis. ---
CASE(PAR_018, "PAR-018", "negative: missing '=' in an axis") {
  const std::string src = "input x: Tensor[a 1]\ngraph g { return x }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "axis missing '=' is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing '='");
}

// --- PAR-019: negative -- graphDecl missing returnStmt before '}'. ---
CASE(PAR_019, "PAR-019", "negative: graphDecl missing returnStmt before '}'") {
  const std::string src = "graph g {\n  a = b\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "graph without return is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing return");
}

// --- PAR-020: negative -- graphDecl missing closing '}'. ---
CASE(PAR_020, "PAR-020", "negative: graphDecl missing closing '}'") {
  const std::string src = "graph g {\n  return a\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "graph missing '}' is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing '}'");
}

// --- PAR-021: negative -- returnStmt missing its identifier. ---
CASE(PAR_021, "PAR-021", "negative: returnStmt missing its identifier") {
  const std::string src = "graph g {\n  return\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "return without a name is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing return identifier");
}

// --- PAR-022: negative -- zero-arg call (arg list requires >= 1). ---
CASE(PAR_022, "PAR-022", "negative: zero-arg call relu() (arg list requires >= 1)") {
  const std::string src = "graph g {\n  a = relu()\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "relu() is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing first arg");
}

// --- PAR-023: negative -- missing closing ')' on a call. ---
CASE(PAR_023, "PAR-023", "negative: missing closing ')' on a call") {
  const std::string src = "graph g {\n  a = relu(x\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "unterminated call is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing ')'");
}

// --- PAR-024: negative -- trailing comma in an arg list. ---
CASE(PAR_024, "PAR-024", "negative: trailing comma in an arg list relu(h,)") {
  const std::string src = "graph g {\n  a = relu(x,)\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "trailing comma is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected after the trailing comma");
}

// --- PAR-025: negative -- keyword arg with no value. ---
CASE(PAR_025, "PAR-025", "negative: keyword arg with no value (stride=))") {
  const std::string src = "graph g {\n  a = f(stride=)\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "stride= with no value is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected for the missing arg value");
}

// --- PAR-026: negative -- a decl after a graphDecl (violates decl* graphDecl+ ordering). ---
CASE(PAR_026, "PAR-026", "negative: a decl after a graphDecl violates decl* graphDecl+ ordering") {
  const std::string src = "graph g { return a }\ninput x: Tensor[b=1]\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "a decl following a graph is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected (\"expecting end of file or KW_GRAPH\")");
}

// --- PAR-027: negative -- a program with zero graphDecls (graphDecl+ requires >= 1). ---
CASE(PAR_027, "PAR-027", "negative: a program with zero graphDecls (graphDecl+ requires >= 1)") {
  const std::string src = "input x: Tensor[a=1]\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "a program with no graph at all is rejected");
  R.check(!d.all().empty() && d.all()[0].code == DiagCode::ParseExpected,
          "reports ParseExpected (\"expecting KW_INPUT or KW_WEIGHT or KW_GRAPH\")");
}

// --- PAR-028: recovery -- two independent errors in different sections. ---
CASE(PAR_028, "PAR-028", "recovery: two independent syntax errors (one in decls, one in a graph body)") {
  const std::string src =
      "input good: Tensor[a=1]\n"
      "weight bad\n"
      "graph g {\n"
      "  m = good\n"
      "  n p\n"
      "  return m\n"
      "}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "the program overall is rejected");
  R.check(d.all().size() == 2, "exactly two diagnostics: one per independent error");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 1 && p.decls[0].name == "good",
          "recovery keeps the valid decl and drops the malformed one");
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1 &&
               p.graphs[0].stmts[0].name == "m" && p.graphs[0].returnName == "m",
           "recovery keeps the valid statement and drops the malformed one, and still finds return");
}

// --- PAR-029: recovery -- two independent malformed statements in the same graph body. ---
// The two errors are separated by two valid statements (6 clean token
// shifts) so each is past ParserErrorStrategy's built-in error-suppression
// window -- after recovering from one syntax error, it silently swallows
// another less than 3 successfully-shifted tokens later (to avoid
// cascades from one root cause, ParserErrorStrategy.h), so two malformed
// statements placed back-to-back would report only the first. That
// window is real recovery behavior, not a bug; this case is deliberately
// spaced to demonstrate genuine two-error recovery instead of exercising
// the suppression window.
CASE(PAR_029, "PAR-029", "recovery: two independent malformed statements in the same graph body") {
  const std::string src = "graph g {\n  a b\n  ok1 = m\n  ok2 = n\n  c d\n  return ok2\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "the program overall is rejected");
  R.check(d.all().size() == 2, "exactly two diagnostics: one per malformed statement");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 2 && p.graphs[0].returnName == "ok2",
          "both malformed statements are dropped but the two valid ones and the return survive");
}

// --- PAR-030: diagnostics -- a missing-token diagnostic's SourceSpan points at the offending token. ---
CASE(PAR_030, "PAR-030", "diagnostics: SourceSpan of a missing-token error points at the offending token") {
  const std::string src = "graph g {\nb c\nreturn b\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "malformed statement is rejected");
  R.check(!d.all().empty(), "a diagnostic is reported");
  if (!d.all().empty()) {
    const Diagnostic& diag = d.all()[0];
    R.check(diag.span.line == 2, "diagnostic points at line 2 (where 'b c' is)");
    R.check(diag.span.col == 3, "diagnostic points at column 3 (the unexpected 'c')");
  }
}

// --- PAR-031: regression -- an integer literal where a binop operand
// requires a float literal (RM 3.24: `operand ::= identifier | floatLit`)
// used to segfault AstBuilder::buildOperand, which assumed a non-matching
// IDENTIFIER meant FLOAT_LITERAL had matched instead of checking for a
// broken (exception) OperandContext. Must report a clean diagnostic, not
// crash. ---
CASE(PAR_031, "PAR-031", "regression: bare int literal as a binop operand reports a diagnostic, not a crash") {
  const std::string src = "graph g {\n  a = x * 2\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "a bare int literal operand (float literal required) is rejected");
  R.check(!d.all().empty(), "a diagnostic is reported");
}

// --- PAR-032: regression -- an axis dimSize that is neither an int
// literal nor an identifier (RM 3.24: `dimSize ::= intLit | identifier`)
// used to segfault AstBuilder::buildDimSize the same way PAR-031 did.
// Must report a clean diagnostic, not crash. ---
CASE(PAR_032, "PAR-032", "regression: an illegal axis dimSize token reports a diagnostic, not a crash") {
  const std::string src = "input x: Tensor[y=@]\ngraph g { return x }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(!result.ok(), "an illegal dimSize token is rejected");
  R.check(!d.all().empty(), "a diagnostic is reported");
}

// --- BND-PAR-001: graph body that is only return (zero statements) vs a single statement. ---
CASE(BND_PAR_001, "BND-PAR-001", "graph body that is only return (zero statements) vs a single statement") {
  const std::string src = "graph empty1 { return z }\ngraph one1 { w = z\n return w }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 2, "two graphs");
  if (p.graphs.size() == 2) {
    R.check(p.graphs[0].stmts.empty() && p.graphs[0].returnName == "z",
            "empty1: zero statements, body is only return");
    R.check(p.graphs[1].stmts.size() == 1 && p.graphs[1].returnName == "w",
            "one1: exactly one statement before return");
  }
}

// --- BND-PAR-002: minimal legal program -- zero decls, one graph, return-only body. ---
CASE(BND_PAR_002, "BND-PAR-002", "minimal legal program: zero decls, exactly one graph, return-only body") {
  const std::string src = "graph onlyreturn { return z }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "minimal program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.empty(), "zero decls");
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.empty() && p.graphs[0].returnName == "z",
          "exactly one graph, empty body, returns z");
}

// --- BND-PAR-003: tensorType axis-count extremes -- 1 axis (min) vs 4 axes (conv-style). ---
CASE(BND_PAR_003, "BND-PAR-003", "tensorType axis-count extremes: 1 axis (min) vs 4 axes (conv-style)") {
  const std::string src =
      "input a: Tensor[x=1]\n"
      "input b: Tensor[batch=B, channel=3, height=224, width=224]\n"
      "graph g { return a }\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.decls.size() == 2, "two decls");
  if (p.decls.size() == 2) {
    R.check(p.decls[0].type.axes.size() == 1, "a has the minimum 1 axis");
    R.check(p.decls[1].type.axes.size() == 4 && isSymbolAxis(p.decls[1].type.axes[0], "batch", "B") &&
                 isConstAxis(p.decls[1].type.axes[1], "channel", 3) &&
                 isConstAxis(p.decls[1].type.axes[2], "height", 224) &&
                 isConstAxis(p.decls[1].type.axes[3], "width", 224),
             "b has a conv-style 4-axis tensorType");
  }
}

// --- BND-PAR-004: call arg-count extreme -- conv2d's 5-arg call. ---
CASE(BND_PAR_004, "BND-PAR-004", "call arg-count extreme: conv2d's 5-arg call (3 positional + 2 keyword)") {
  const std::string src = "graph g {\n  a = conv2d(image, W, b, stride=1, padding=1)\n  return a\n}\n";
  DiagnosticEngine d;
  auto result = parse(src, d);
  R.check(result.ok(), "program parses with no diagnostics");
  const ast::AstProgram& p = result.value();
  R.check(p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1, "one statement");
  if (p.graphs.size() == 1 && p.graphs[0].stmts.size() == 1) {
    const auto* c = asCall(p.graphs[0].stmts[0].rhs.get());
    R.check(c && c->args.size() == 5, "conv2d(...) has 5 args");
  }
}

#undef CASE

}  // namespace

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
