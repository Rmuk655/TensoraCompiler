#pragma once
#include "antlr4-runtime.h"

// This file defines the parser's error-recovery strategy.
// The parser uses this strategy when it finds a syntax error.
//
// ANTLR's default strategy repairs some errors by inserting or deleting
// one token (for example, it can silently add a missing "="). That repair
// hides the error instead of dropping the bad statement. This strategy
// turns that repair off, so every syntax error falls back to panic-mode
// recovery: the parser drops the bad construct and resyncs at the next
// safe point (the next statement or declaration).
//
// Turning off single-token repair has one side effect: after the parser
// drops a bad construct, it can resync at a point that itself looks like
// the start of a new (still broken) construct, causing a second, spurious
// diagnostic from the same root cause. The base class only requires one
// clean token match to stop suppressing new errors; that is too little
// here. This class instead requires a few consecutive clean matches
// (`kCleanShiftsRequired`) before it reports a new error.
namespace tensora {

class ParserErrorStrategy : public antlr4::DefaultErrorStrategy {
 public:
  bool singleTokenInsertion(antlr4::Parser* recognizer) override;
  antlr4::Token* singleTokenDeletion(antlr4::Parser* recognizer) override;
  void reportError(antlr4::Parser* recognizer, const antlr4::RecognitionException& e) override;
  void reportMatch(antlr4::Parser* recognizer) override;

 private:
  static constexpr int kCleanShiftsRequired = 3;
  int cleanShiftsSinceError_ = kCleanShiftsRequired;
};

}  // namespace tensora
