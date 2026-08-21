#pragma once
#include <unordered_map>
#include <vector>

#include "tensora/shared/Type.h"

// The bounded dimension solver (RM 3.4; ARCH-006). It uses union-find
// over SymbolId, plus integer arithmetic. It handles exactly three
// cases: a concrete integer, a symbolic variable, or unifying the two.
// It has no SMT solver and no general nonlinear solving. Each graph
// gets its own DimSolver (RM 3.4 scopes dimension reasoning to "within
// one graph"), even though TypeContext interns SymbolId names across
// the whole program.
namespace tensora {

// INTERFACES.md sketches sameElementCount as returning bool. Here it
// returns an enum instead, because semantic analysis must tell two
// cases apart: "the counts are decidable and differ" (RM 3.12's
// SemaReshapeCount) versus "this needs reasoning outside the bounded
// subset" (RM 3.4's SemaUnresolvableShape). These are different
// diagnostics for different problems, not the same failure.
struct ElementCountResult {
  enum class Status {
    Equal,          // The symbolic factors match, or there are none, and the constant products are equal.
    Mismatch,       // The symbolic factors match, or there are none, and the constant products differ.
    Unresolvable    // The symbolic factors differ in identity or in multiplicity.
  } status;
  int64_t lhsConstProduct = 1;
  int64_t rhsConstProduct = 1;
};

class DimSolver {
 public:
  // Const == Const: checked, not assumed. Symbol ~ Symbol: always
  // unifies; the two become synonyms. Symbol := Const: binds the
  // symbol. Returns false only when both sides already resolve to
  // different constants.
  bool unify(Dimension a, Dimension b);

  // Follows the union-find chain, with path compression, to the
  // current best-known representative: an unbound Symbol, or the Const
  // it was bound to, directly or through a chain.
  Dimension resolve(Dimension d) const;

  // RM 3.4 / RM 3.12: checks product-of-sizes equality within the
  // bounded subset. It cancels a symbolic factor only when that factor
  // appears with the same multiplicity on both sides. Otherwise the
  // comparison falls outside the subset (Unresolvable).
  ElementCountResult sameElementCount(const std::vector<Dimension>& lhs,
                                      const std::vector<Dimension>& rhs) const;

 private:
  mutable std::unordered_map<SymbolId, Dimension> rep_;  // Path-compressed each time resolve() runs.
};

}  // namespace tensora
