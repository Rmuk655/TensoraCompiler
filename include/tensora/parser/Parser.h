#pragma once
#include <string>

#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Result.h"
#include "tensora/shared/SourceBuffer.h"

namespace tensora {

// Runs the ANTLR-generated TensoraLexer and TensoraParser (grammar/
// Tensora.g4) directly over source text, then walks the parse tree to
// build the AST below (see src/parser/AstBuilder.h). No project Token
// vector sits between the lexer and the parser here (Lexer/Token exist
// only for the lexer's own standalone demo and tests; see the RM
// migration doc, Section 5). This class builds a complete, best-effort
// AST and reports syntax errors into the DiagnosticEngine. On an error,
// it recovers at the next statement or top-level declaration/graph, so
// one file can report several syntax errors in one pass. It never
// inspects shapes or types.
class Parser {
 public:
  Parser(const SourceBuffer& src, DiagnosticEngine& diags);

  Result<ast::AstProgram> parseProgram();

 private:
  const SourceBuffer& src_;
  DiagnosticEngine& diags_;
};

// For debugging and demos only (--emit-parse-tree): a parenthesized
// (LISP-style) dump of the raw ANTLR parse tree, before AstBuilder
// turns it into the AST above. No other compiler phase reads this
// dump. It exists so a reader can see what ANTLR itself produced,
// separate from the project's own AST.
std::string dumpParseTree(const SourceBuffer& src);

}  // namespace tensora
