#include "ParserErrorListener.h"

namespace tensora {

namespace {

// See the class comment above: a "no viable alternative" error has no
// single expected token to name, so it gets its own DiagCode.
DiagCode classify(std::exception_ptr e) {
  if (!e) return DiagCode::ParseExpected;
  try {
    std::rethrow_exception(e);
  } catch (const antlr4::NoViableAltException&) {
    return DiagCode::ParseUnexpectedToken;
  } catch (...) {
    return DiagCode::ParseExpected;
  }
}

}  // namespace

void ParserErrorListener::syntaxError(antlr4::Recognizer*, antlr4::Token* offendingSymbol, size_t line,
                                       size_t charPositionInLine, const std::string& msg,
                                       std::exception_ptr e) {
  SourceSpan span;
  span.fileId = fileId_;
  span.line = static_cast<uint32_t>(line);
  span.col = static_cast<uint32_t>(charPositionInLine + 1);
  if (offendingSymbol != nullptr) {
    span.beginOffset = static_cast<uint32_t>(offendingSymbol->getStartIndex());
    span.endOffset = static_cast<uint32_t>(offendingSymbol->getStopIndex() + 1);
  } else {
    span.beginOffset = span.endOffset = 0;
  }

  Diagnostic d;
  d.severity = Severity::Error;
  d.code = classify(e);
  d.span = span;
  d.message = msg;
  diags_.report(d);
}

}  // namespace tensora
