#pragma once
#include <string>
#include <vector>

#include "tensora/shared/SourceBuffer.h"
#include "tensora/shared/SourceSpan.h"

namespace tensora {

enum class Severity { Error, Warning, Note };

// These codes stay stable, so tests can check the error category
// instead of the exact wording. The parser and semantic phases share
// this one set of codes.
enum class DiagCode {
  LexIllegalChar,
  LexIntegerOverflow,
  ParseUnexpectedToken,
  ParseExpected,
  SemaUndefinedName,
  SemaRedeclaration,
  SemaArgCount,
  SemaArgKind,
  SemaTypeMismatch,
  SemaRankMismatch,
  SemaShapeMismatch,
  SemaAxisNameMismatch,
  SemaBroadcast,
  SemaReshapeCount,
  SemaTransposePerm,
  SemaAxisNotFound,
  SemaConvChannel,
  SemaConvArithmetic,
  SemaUnresolvableShape,
  // RM 3.3: an axis size must be a concrete positive integer, or a
  // symbolic dimension. The compiler never checks a symbolic dimension
  // for a positive value statically.
  SemaNonPositiveDimension,
  // RM 3.3: axis names must be unique within one tensor type (Stage 9
  // decision, 2026-08-19 -- see docs/architecture/DECISIONS.md ARCH-013).
  SemaDuplicateAxisName,
  // INTERFACES.md §3a.6: GraphVerifier re-checks TCG invariants (SSA, acyclic and
  // closed, single output), no matter how the graph was built. A
  // violation here is a compiler-internal defect (a buggy pass or a bad
  // hand-built fixture), not a user-program error. Every Sema* code
  // above is instead about the *source program*'s meaning.
  IrInvariantViolation
};

struct Diagnostic {
  Severity severity = Severity::Error;
  DiagCode code = DiagCode::LexIllegalChar;
  SourceSpan span;
  std::string message;
};

// INTERFACES.md §0's DiagnosticList alias.
using DiagnosticList = std::vector<Diagnostic>;

// Collects diagnostics from every phase. One engine per compilation run.
class DiagnosticEngine {
public:
  void report(const Diagnostic& d) { items_.push_back(d); }

  bool hasErrors() const {
    for (const Diagnostic& d : items_) {
      if (d.severity == Severity::Error) return true;
    }
    return false;
  }

  const DiagnosticList& all() const { return items_; }

  // One line per diagnostic: "file:line:col: error: message".
  std::string render(const SourceBuffer& src) const {
    std::string out;
    for (const Diagnostic& d : items_) {
      out += src.name();
      out += ":" + std::to_string(d.span.line);
      out += ":" + std::to_string(d.span.col);
      out += (d.severity == Severity::Error) ? ": error: " : ": note: ";
      out += d.message;
      out += "\n";
    }
    return out;
  }

private:
  DiagnosticList items_;
};

}  // namespace tensora
