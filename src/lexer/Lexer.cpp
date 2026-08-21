// This file implements the lexer.
// The lexer reads source text and creates tokens.
// The parser reads these tokens (Parser.cpp lexes its own copy directly;
// this path exists for standalone lexer use: --emit-tokens and the
// lexer tests).
#include "tensora/lexer/Lexer.h"

#include <memory>

#include "TensoraLexer.h"
#include "antlr4-runtime.h"
#include "tensora/lexer/TokenAdapter.h"

namespace tensora {

Lexer::Lexer(const SourceBuffer& src, DiagnosticEngine& diags) : src_(src), diags_(diags) {}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> tokens;

  antlr4::ANTLRInputStream input(src_.text());
  generated::TensoraLexer lexer(&input);
  lexer.removeErrorListeners();  // the grammar's own ERROR rule reports every illegal char; no console noise

  for (;;) {
    std::unique_ptr<antlr4::Token> tok(lexer.nextToken());
    Token t = adaptToken(*tok, src_, diags_);
    tokens.push_back(t);
    if (t.kind == TokenKind::EndOfFile) break;
  }

  return tokens;
}

}  // namespace tensora
