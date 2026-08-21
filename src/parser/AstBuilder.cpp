// This file implements the parse-tree-to-AST conversion.
// It walks the ANTLR parse tree and creates the AST. Semantic analysis
// reads the AST next.
#include "AstBuilder.h"

#include <charconv>
#include <optional>
#include <string>

namespace tensora {

namespace {

std::optional<int64_t> tryParseInt64(std::string_view text) {
  int64_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

ast::ExprPtr makeName(std::string name, SourceSpan span) {
  auto e = std::make_unique<ast::NameExpr>();
  e->name = std::move(name);
  e->span = span;
  return e;
}

ast::ExprPtr makeScalar(double value, SourceSpan span) {
  auto e = std::make_unique<ast::ScalarExpr>();
  e->value = value;
  e->span = span;
  return e;
}

}  // namespace

SourceSpan AstBuilder::spanOf(antlr4::ParserRuleContext* ctx) const {
  antlr4::Token* start = ctx->getStart();
  antlr4::Token* stop = ctx->getStop() != nullptr ? ctx->getStop() : start;
  SourceSpan s;
  s.fileId = fileId_;
  s.beginOffset = static_cast<uint32_t>(start->getStartIndex());
  s.endOffset = static_cast<uint32_t>(stop->getStopIndex() + 1);
  s.line = static_cast<uint32_t>(start->getLine());
  s.col = static_cast<uint32_t>(start->getCharPositionInLine() + 1);
  return s;
}

SourceSpan AstBuilder::spanOf(antlr4::tree::TerminalNode* node) const {
  antlr4::Token* tok = node->getSymbol();
  SourceSpan s;
  s.fileId = fileId_;
  s.beginOffset = static_cast<uint32_t>(tok->getStartIndex());
  s.endOffset = static_cast<uint32_t>(tok->getStopIndex() + 1);
  s.line = static_cast<uint32_t>(tok->getLine());
  s.col = static_cast<uint32_t>(tok->getCharPositionInLine() + 1);
  return s;
}

// LANG-001: an INT_LITERAL must fit in a signed 64-bit integer, the
// same rule the lexer's own token-adaptation path uses
// (TokenAdapter.cpp). The parser reads INT_LITERAL text directly from
// the ANTLR token stream, not through that path (Parser.h). So it
// needs its own overflow check, to report the same LexIntegerOverflow
// diagnostic instead of silently misparsing the value.
int64_t AstBuilder::parseIntLiteral(antlr4::tree::TerminalNode* node) {
  std::string text = node->getText();
  if (auto value = tryParseInt64(text)) return *value;
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = DiagCode::LexIntegerOverflow;
  d.span = spanOf(node);
  d.message = "integer literal '" + text + "' does not fit in a signed 64-bit integer";
  diags_.report(d);
  return 0;
}

ast::TensorTypeSyntax AstBuilder::buildTensorType(generated::TensoraParser::TensorTypeContext* ctx) {
  ast::TensorTypeSyntax t;
  for (auto* axisCtx : ctx->axis()) {
    if (axisCtx->exception) continue;
    t.axes.push_back(buildAxis(axisCtx));
  }
  t.span = spanOf(ctx);
  return t;
}

ast::AxisSyntax AstBuilder::buildAxis(generated::TensoraParser::AxisContext* ctx) {
  return ast::AxisSyntax{ctx->IDENTIFIER()->getText(), buildDimSize(ctx->dimSize()), spanOf(ctx)};
}

ast::DimSyntax AstBuilder::buildDimSize(generated::TensoraParser::DimSizeContext* ctx) {
  if (auto* intTok = ctx->INT_LITERAL()) {
    return ast::DimSyntax{ast::DimSyntax::K::Const, parseIntLiteral(intTok), {}};
  }
  // A syntax error here (for example "y=@") can be caught and
  // recovered by this rule's own call, without the enclosing axis
  // context ever seeing an exception (each rule's try/catch block is
  // local; see buildArgValue's note above). So this checks explicitly,
  // instead of assuming IDENTIFIER() is non-null.
  if (auto* id = ctx->IDENTIFIER()) return ast::DimSyntax{ast::DimSyntax::K::Symbol, 0, id->getText()};
  return ast::DimSyntax{ast::DimSyntax::K::Const, 0, {}};
}

ast::ExprPtr AstBuilder::buildOperand(generated::TensoraParser::OperandContext* ctx) {
  if (auto* id = ctx->IDENTIFIER()) return makeName(id->getText(), spanOf(ctx));
  // A syntax error here (for example a bare integer literal, which this
  // rule does not accept, since RM 3.24 requires a float literal) can
  // be caught and recovered by this rule's own call, without the
  // enclosing BinaryExpression context ever seeing an exception (each
  // rule's try/catch block is local; see the matching note on
  // buildArgValue above). So this checks explicitly, instead of
  // assuming FLOAT_LITERAL() is non-null.
  if (auto* f = ctx->FLOAT_LITERAL()) return makeScalar(std::stod(f->getText()), spanOf(ctx));
  return makeScalar(0.0, spanOf(ctx));
}

ast::ExprPtr AstBuilder::buildCallExpr(generated::TensoraParser::CallExprContext* ctx) {
  auto call = std::make_unique<ast::CallExpr>();
  call->callee = ctx->IDENTIFIER()->getText();
  for (auto* argCtx : ctx->arg()) {
    if (argCtx->exception) continue;
    call->args.push_back(buildArg(argCtx));
  }
  call->span = spanOf(ctx);
  return call;
}

ast::ExprPtr AstBuilder::buildExpression(generated::TensoraParser::ExpressionContext* ctx) {
  using namespace generated;
  if (ctx == nullptr || ctx->exception) return nullptr;

  if (auto* c = dynamic_cast<TensoraParser::CallExpressionContext*>(ctx)) {
    return buildCallExpr(c->callExpr());
  }
  if (auto* b = dynamic_cast<TensoraParser::BinaryExpressionContext*>(ctx)) {
    ast::ExprPtr lhs = makeName(b->IDENTIFIER()->getText(), spanOf(b->IDENTIFIER()));
    ast::ExprPtr rhs = buildOperand(b->operand());
    SourceSpan span = spanOf(b);
    size_t tokenType = b->binop()->getStart()->getType();
    if (tokenType == TensoraParser::AT) {
      auto e = std::make_unique<ast::MatMulExpr>();
      e->lhs = std::move(lhs);
      e->rhs = std::move(rhs);
      e->span = span;
      return e;
    }
    ast::BinaryExpr::Op op = ast::BinaryExpr::Op::Add;
    switch (tokenType) {
      case TensoraParser::PLUS: op = ast::BinaryExpr::Op::Add; break;
      case TensoraParser::MINUS: op = ast::BinaryExpr::Op::Sub; break;
      case TensoraParser::STAR: op = ast::BinaryExpr::Op::Mul; break;
      case TensoraParser::SLASH: op = ast::BinaryExpr::Op::Div; break;
      default: break;
    }
    auto e = std::make_unique<ast::BinaryExpr>();
    e->op = op;
    e->lhs = std::move(lhs);
    e->rhs = std::move(rhs);
    e->span = span;
    return e;
  }
  if (auto* n = dynamic_cast<TensoraParser::NameExpressionContext*>(ctx)) {
    return makeName(n->IDENTIFIER()->getText(), spanOf(ctx));
  }
  return nullptr;
}

ast::Arg AstBuilder::buildArgValue(generated::TensoraParser::ArgValueContext* ctx) {
  using namespace generated;
  ast::Arg a;
  a.span = spanOf(ctx);
  // A syntax error inside argValue (for example "stride=)") can be
  // caught and recovered by this rule's own call, without the
  // enclosing `arg` context ever seeing an exception (each rule's
  // try/catch block is local). So `ctx` here may be the plain
  // ArgValueContext base, not one of its three labeled subclasses.
  // This checks explicitly, instead of assuming one of the
  // dynamic_casts below succeeds.
  if (ctx->exception) return a;
  if (auto* iv = dynamic_cast<TensoraParser::IntArgValueContext*>(ctx)) {
    a.kind = ast::Arg::Kind::Int;
    a.intVal = parseIntLiteral(iv->INT_LITERAL());
    return a;
  }
  if (auto* nv = dynamic_cast<TensoraParser::NameArgValueContext*>(ctx)) {
    a.kind = ast::Arg::Kind::Name;
    a.name = nv->IDENTIFIER()->getText();
    return a;
  }
  if (auto* lv = dynamic_cast<TensoraParser::IntListArgValueContext*>(ctx)) {
    a.kind = ast::Arg::Kind::IntList;
    for (auto* tok : lv->INT_LITERAL()) {
      a.intList.push_back(parseIntLiteral(tok));
    }
  }
  return a;
}

ast::Arg AstBuilder::buildArg(generated::TensoraParser::ArgContext* ctx) {
  using namespace generated;
  ast::Arg fallback;
  fallback.span = spanOf(ctx);
  if (ctx->exception) return fallback;  // same reasoning as buildArgValue above
  if (auto* k = dynamic_cast<TensoraParser::KeywordArgContext*>(ctx)) {
    ast::Arg a = buildArgValue(k->argValue());
    a.keyword = k->IDENTIFIER()->getText();
    a.span = spanOf(ctx);
    return a;
  }
  if (auto* p = dynamic_cast<TensoraParser::PositionalArgContext*>(ctx)) {
    ast::Arg a;
    a.kind = ast::Arg::Kind::Name;
    a.name = p->IDENTIFIER()->getText();
    a.span = spanOf(ctx);
    return a;
  }
  if (auto* axisListCtx = dynamic_cast<TensoraParser::AxisListArgContext*>(ctx)) {
    ast::Arg a;
    a.kind = ast::Arg::Kind::AxisList;
    for (auto* axisCtx : axisListCtx->axis()) {
      if (axisCtx->exception) continue;
      a.axisList.push_back(buildAxis(axisCtx));
    }
    a.span = spanOf(ctx);
    return a;
  }
  return fallback;
}

ast::AstProgram AstBuilder::build(generated::TensoraParser::ProgramContext* ctx) {
  ast::AstProgram program;

  for (auto* declCtx : ctx->decl()) {
    if (declCtx->exception) continue;
    if (auto* inputCtx = declCtx->inputDecl()) {
      if (inputCtx->exception) continue;
      ast::Decl d;
      d.kind = ast::Decl::Kind::Input;
      d.name = inputCtx->IDENTIFIER()->getText();
      d.type = buildTensorType(inputCtx->tensorType());
      d.span = spanOf(declCtx);
      program.decls.push_back(std::move(d));
    } else if (auto* weightCtx = declCtx->weightDecl()) {
      if (weightCtx->exception) continue;
      ast::Decl d;
      d.kind = ast::Decl::Kind::Weight;
      d.name = weightCtx->IDENTIFIER()->getText();
      d.type = buildTensorType(weightCtx->tensorType());
      d.span = spanOf(declCtx);
      program.decls.push_back(std::move(d));
    }
  }

  for (auto* graphCtx : ctx->graphDecl()) {
    if (graphCtx->exception) continue;
    ast::GraphDecl g;
    g.name = graphCtx->IDENTIFIER()->getText();
    for (auto* stmtCtx : graphCtx->statement()) {
      if (stmtCtx->exception) continue;
      ast::Stmt s;
      s.name = stmtCtx->IDENTIFIER()->getText();
      s.rhs = buildExpression(stmtCtx->expression());
      s.span = spanOf(stmtCtx);
      g.stmts.push_back(std::move(s));
    }
    auto* retCtx = graphCtx->returnStmt();
    if (retCtx != nullptr && retCtx->exception == nullptr) {
      g.returnName = retCtx->IDENTIFIER()->getText();
    }
    g.span = spanOf(graphCtx);
    program.graphs.push_back(std::move(g));
  }

  return program;
}

}  // namespace tensora
