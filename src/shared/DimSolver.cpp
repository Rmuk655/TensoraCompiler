// This file implements the bounded symbolic-dimension solver.
// Semantic analysis uses it to unify axis sizes and check element-count
// equality (RM 3.4), without a general SMT solver.
#include "tensora/shared/DimSolver.h"

#include <algorithm>

namespace tensora {

Dimension DimSolver::resolve(Dimension d) const {
  while (d.kind == Dimension::Kind::Symbol) {
    auto it = rep_.find(d.symbol);
    if (it == rep_.end()) break;  // unbound: this symbol is its own representative
    if (dimensionsEqual(it->second, d)) break;  // already a fixed point
    d = it->second;
  }
  return d;
}

bool DimSolver::unify(Dimension a, Dimension b) {
  Dimension ra = resolve(a);
  Dimension rb = resolve(b);

  if (ra.kind == Dimension::Kind::Const && rb.kind == Dimension::Kind::Const) {
    return ra.constant == rb.constant;
  }
  if (ra.kind == Dimension::Kind::Symbol && rb.kind == Dimension::Kind::Symbol) {
    if (ra.symbol == rb.symbol) return true;
    rep_[ra.symbol] = rb;  // ra becomes a synonym of rb
    return true;
  }
  // Exactly one side is a free Symbol; bind it to the other's resolved value.
  if (ra.kind == Dimension::Kind::Symbol) {
    rep_[ra.symbol] = rb;
  } else {
    rep_[rb.symbol] = ra;
  }
  return true;
}

ElementCountResult DimSolver::sameElementCount(const std::vector<Dimension>& lhs,
                                                const std::vector<Dimension>& rhs) const {
  auto split = [this](const std::vector<Dimension>& dims, int64_t& constProduct,
                       std::vector<SymbolId>& symbols) {
    constProduct = 1;
    for (const Dimension& raw : dims) {
      Dimension d = resolve(raw);
      if (d.kind == Dimension::Kind::Const) {
        constProduct *= d.constant;
      } else {
        symbols.push_back(d.symbol);
      }
    }
    std::sort(symbols.begin(), symbols.end());
  };

  ElementCountResult result;
  std::vector<SymbolId> lhsSymbols, rhsSymbols;
  split(lhs, result.lhsConstProduct, lhsSymbols);
  split(rhs, result.rhsConstProduct, rhsSymbols);

  if (lhsSymbols != rhsSymbols) {
    result.status = ElementCountResult::Status::Unresolvable;
    return result;
  }
  result.status = (result.lhsConstProduct == result.rhsConstProduct)
                       ? ElementCountResult::Status::Equal
                       : ElementCountResult::Status::Mismatch;
  return result;
}

}  // namespace tensora
