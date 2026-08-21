// This file implements the tensor type representation.
// Semantic analysis, optimization, and codegen all read and create
// these types, so this file lives in shared/.
#include "tensora/shared/Type.h"

#include <sstream>

namespace tensora {

bool dimensionsEqual(const Dimension& a, const Dimension& b) {
  if (a.kind != b.kind) return false;
  if (a.kind == Dimension::Kind::Const) return a.constant == b.constant;
  return a.symbol == b.symbol;
}

bool typesEqual(const Type* a, const Type* b) {
  if (a->kind() != b->kind()) return false;
  if (a->kind() == Type::Kind::Tensor) {
    const auto* ta = static_cast<const TensorType*>(a);
    const auto* tb = static_cast<const TensorType*>(b);
    if (ta->rank() != tb->rank()) return false;
    for (size_t i = 0; i < ta->rank(); ++i) {
      if (ta->axes()[i].name != tb->axes()[i].name) return false;
      if (!dimensionsEqual(ta->axes()[i].size, tb->axes()[i].size)) return false;
    }
    return true;
  }
  return true;  // Scalar == Scalar, Invalid == Invalid (kind already matched)
}

std::string describeDimension(const Dimension& d, const TypeContext& ctx) {
  if (d.kind == Dimension::Kind::Const) return std::to_string(d.constant);
  return std::string(ctx.symbolName(d.symbol));
}

std::string describeType(const Type* t, const TypeContext& ctx) {
  switch (t->kind()) {
    case Type::Kind::Scalar:
      return "scalar";
    case Type::Kind::Invalid:
      return "<invalid>";
    case Type::Kind::Tensor: {
      const auto* tt = static_cast<const TensorType*>(t);
      std::ostringstream out;
      out << "Tensor[";
      for (size_t i = 0; i < tt->rank(); ++i) {
        if (i) out << ", ";
        out << tt->axes()[i].name << "=" << describeDimension(tt->axes()[i].size, ctx);
      }
      out << "]";
      return out.str();
    }
  }
  return "?";
}

}  // namespace tensora
