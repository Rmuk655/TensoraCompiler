// This file implements the parser.
// The parser reads source text, runs the ANTLR lexer and parser, and
// creates the AST. Semantic analysis reads the AST next.
#include "tensora/parser/Parser.h"

#include "AstBuilder.h"
#include "ParserErrorListener.h"
#include "ParserErrorStrategy.h"
#include "TensoraLexer.h"
#include "TensoraParser.h"
#include "antlr4-runtime.h"
#include "tensora/lexer/TokenAdapter.h"

namespace tensora {

Parser::Parser(const SourceBuffer& src, DiagnosticEngine& diags) : src_(src), diags_(diags) {}

Result<ast::AstProgram> Parser::parseProgram() {
  antlr4::ANTLRInputStream input(src_.text());
  generated::TensoraLexer lexer(&input);
  lexer.removeErrorListeners();  // illegal characters are diagnosed below, once, for the whole file

  antlr4::CommonTokenStream tokens(&lexer);
  // The grammar routes illegal characters to a hidden ERRORS channel
  // (see grammar/Tensora.g4). So the parser below never sees them as
  // spurious "unexpected token" syntax errors. This code reports them
  // here instead, exactly once per character, matching
  // Lexer::tokenize()'s behavior for the standalone lexer path.
  // TokenAdapter.cpp is the one place both agree on what "illegal"
  // means.
  tokens.fill();
  for (antlr4::Token* tok : tokens.getTokens()) {
    if (static_cast<size_t>(tok->getType()) == generated::TensoraParser::ERROR) {
      adaptToken(*tok, src_, diags_);
    }
  }

  generated::TensoraParser parser(&tokens);
  parser.removeErrorListeners();
  ParserErrorListener errorListener(src_.fileId(), diags_);
  parser.addErrorListener(&errorListener);
  parser.setErrorHandler(std::make_shared<ParserErrorStrategy>());

  generated::TensoraParser::ProgramContext* tree = parser.program();

  AstBuilder builder(src_.fileId(), diags_);
  ast::AstProgram program = builder.build(tree);

  bool ok = !diags_.hasErrors();
  return Result<ast::AstProgram>::make(std::move(program), ok, diags_.all());
}

std::string dumpParseTree(const SourceBuffer& src) {
  antlr4::ANTLRInputStream input(src.text());
  generated::TensoraLexer lexer(&input);
  lexer.removeErrorListeners();
  antlr4::CommonTokenStream tokens(&lexer);
  generated::TensoraParser parser(&tokens);
  parser.removeErrorListeners();  // this dump is best-effort; diagnostics come from parseProgram()
  generated::TensoraParser::ProgramContext* tree = parser.program();
  return tree->toStringTree(&parser);
}

}  // namespace tensora
