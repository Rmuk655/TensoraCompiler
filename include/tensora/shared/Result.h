#pragma once
#include <optional>
#include <utility>

#include "tensora/shared/Diagnostic.h"

namespace tensora {

// A phase's outcome: a best-effort value (built even after recovery
// from an error), plus every diagnostic reported while building it.
// ok() is false when any Error-severity diagnostic was reported, but
// value() still holds the built value, so a caller or a test can look
// at the partial result.
template <class T>
class Result {
 public:
  static Result make(T value, bool ok, DiagnosticList errors) {
    Result r;
    r.value_.emplace(std::move(value));  // construction, not assignment: T need not be move-assignable
    r.ok_ = ok;
    r.errors_ = std::move(errors);
    return r;
  }

  bool ok() const { return ok_; }
  T& value() { return *value_; }
  const T& value() const { return *value_; }
  const DiagnosticList& errors() const { return errors_; }

 private:
  std::optional<T> value_;
  bool ok_ = false;
  DiagnosticList errors_;
};

}  // namespace tensora
