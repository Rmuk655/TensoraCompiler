#pragma once
#include "antlr4-runtime.h"
#include "tensora/shared/Diagnostic.h"

// Converts ANTLR's syntax-error callback into a project Diagnostic.
//
// ANTLR reports two kinds of syntax error here. A mismatched-token
// error (it found token X, but expected token Y) names one specific
// expected token, so this becomes DiagCode::ParseExpected. A "no
// viable alternative" error (antlr4::NoViableAltException) happens
// when the current token cannot start any rule the parser could try
// next. There is no single expected token to name, so this becomes
// DiagCode::ParseUnexpectedToken instead.
//
// Recovery itself uses ParserErrorStrategy (panic-mode resync to the
// enclosing rule's follow set).
namespace tensora {

class ParserErrorListener : public antlr4::BaseErrorListener {
 public:
  ParserErrorListener(uint32_t fileId, DiagnosticEngine& diags) : fileId_(fileId), diags_(diags) {}

  void syntaxError(antlr4::Recognizer* recognizer, antlr4::Token* offendingSymbol, size_t line,
                    size_t charPositionInLine, const std::string& msg,
                    std::exception_ptr e) override;

 private:
  uint32_t fileId_;
  DiagnosticEngine& diags_;
};

}  // namespace tensora
