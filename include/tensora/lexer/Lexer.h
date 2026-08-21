#pragma once
#include <vector>

#include "tensora/shared/Diagnostic.h"
#include "tensora/lexer/Token.h"
#include "tensora/shared/SourceBuffer.h"

namespace tensora {

// Turns source text into a list of tokens that always ends with EndOfFile.
// Whitespace and '#' comments are dropped. An illegal character produces an
// Error token plus a LexIllegalChar diagnostic, and scanning continues, so
// one run can report several problems.
class Lexer {
public:
  Lexer(const SourceBuffer& src, DiagnosticEngine& diags);

  std::vector<Token> tokenize();

private:
  const SourceBuffer& src_;
  DiagnosticEngine& diags_;
};

}  // namespace tensora
