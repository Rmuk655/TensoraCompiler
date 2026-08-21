#pragma once
#include "TensoraParser.h"
#include "antlr4-runtime.h"
#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"

// Walks the ANTLR parse tree (grammar/Tensora.g4) and builds the
// project AST (tensora/shared/Ast.h) (INTERFACES.md Section 2). This
// is plain recursive descent
// over context pointers. It uses dynamic_cast to the labeled-
// alternative subclasses ANTLR generates for `expression`, `arg`, and
// `argValue`, instead of the generated Visitor base: the AST is built
// from move-only std::unique_ptr nodes, and the Visitor's std::any-
// based dispatch cannot carry those. This approach keeps the
// ANTLR/AST boundary as one small, ordinary C++ class, keeping AST
// construction in project-owned code.
//
// A context whose `exception` field is set means ANTLR's
// DefaultErrorStrategy caught a syntax error while parsing it, and
// recovered. This code silently skips that subtree; it does not build
// it, even partially.
namespace tensora {

class AstBuilder {
 public:
  AstBuilder(uint32_t fileId, DiagnosticEngine& diags) : fileId_(fileId), diags_(diags) {}

  ast::AstProgram build(generated::TensoraParser::ProgramContext* ctx);

 private:
  SourceSpan spanOf(antlr4::ParserRuleContext* ctx) const;
  SourceSpan spanOf(antlr4::tree::TerminalNode* node) const;
  int64_t parseIntLiteral(antlr4::tree::TerminalNode* node);

  ast::TensorTypeSyntax buildTensorType(generated::TensoraParser::TensorTypeContext* ctx);
  ast::AxisSyntax buildAxis(generated::TensoraParser::AxisContext* ctx);
  ast::DimSyntax buildDimSize(generated::TensoraParser::DimSizeContext* ctx);
  ast::ExprPtr buildExpression(generated::TensoraParser::ExpressionContext* ctx);
  ast::ExprPtr buildOperand(generated::TensoraParser::OperandContext* ctx);
  ast::ExprPtr buildCallExpr(generated::TensoraParser::CallExprContext* ctx);
  ast::Arg buildArg(generated::TensoraParser::ArgContext* ctx);
  ast::Arg buildArgValue(generated::TensoraParser::ArgValueContext* ctx);

  uint32_t fileId_;
  DiagnosticEngine& diags_;
};

}  // namespace tensora
