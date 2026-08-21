#include "ParserErrorStrategy.h"

namespace tensora {

// Always refuse the single-token repair. The base class's recoverInline()
// then throws instead of silently inserting a token, so the enclosing
// rule's own error handling drops the bad construct (panic-mode recovery).
bool ParserErrorStrategy::singleTokenInsertion(antlr4::Parser*) { return false; }

// Always refuse the single-token repair. Returning nullptr tells the base
// class no deletion is possible, so it falls back to panic-mode recovery
// the same way singleTokenInsertion above does.
antlr4::Token* ParserErrorStrategy::singleTokenDeletion(antlr4::Parser*) { return nullptr; }

// Report the error only after enough clean token matches since the last
// one. A cascade error (the same root cause, reported again right after
// recovery) has too few clean matches in between and is dropped here
// instead of reaching the DiagnosticEngine.
void ParserErrorStrategy::reportError(antlr4::Parser* recognizer, const antlr4::RecognitionException& e) {
  if (cleanShiftsSinceError_ < kCleanShiftsRequired) return;
  cleanShiftsSinceError_ = 0;
  antlr4::DefaultErrorStrategy::reportError(recognizer, e);
}

void ParserErrorStrategy::reportMatch(antlr4::Parser* recognizer) {
  if (cleanShiftsSinceError_ < kCleanShiftsRequired) ++cleanShiftsSinceError_;
  antlr4::DefaultErrorStrategy::reportMatch(recognizer);
}

}  // namespace tensora
