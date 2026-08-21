#pragma once
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The Tensora type system (RM 3.3-3.5; IR.md §2; ARCH-005/006).
// TypeContext interns and owns every type. Code refers to a type by
// raw const pointer (TypePtr = const Type*), for the life of one
// TypedProgram. The element type is always f32 in v1; there is no
// dtype field (RM 3.23).
namespace tensora {

// An interned symbolic-dimension name. The name -> id table is shared
// across the whole program (one TypeContext), only so the same
// spelling always gets the same id. This does NOT unify the symbol
// across graphs. Each graph gets its own fresh DimSolver
// (SemanticAnalyzer::analyzeGraph). So a symbol named "B" in one graph
// and "B" in another graph are independent, matching RM 3.4 ("within
// one graph").
using SymbolId = uint32_t;

struct Dimension {
  enum class Kind { Const, Symbol };
  Kind kind = Kind::Const;
  int64_t constant = 0;  // valid iff kind == Const (> 0, RM 3.3 "concrete positive integer")
  SymbolId symbol = 0;   // valid iff kind == Symbol

  static Dimension makeConst(int64_t v) {
    Dimension d;
    d.kind = Kind::Const;
    d.constant = v;
    return d;
  }
  static Dimension makeSymbol(SymbolId s) {
    Dimension d;
    d.kind = Kind::Symbol;
    d.symbol = s;
    return d;
  }
};

struct Axis {
  std::string name;  // e.g. "batch", "out", "kseq" (RM 3.5: axis names, not a reserved namespace)
  Dimension size;
};

class Type {
 public:
  enum class Kind { Tensor, Scalar, Invalid };
  explicit Type(Kind k) : kind_(k) {}
  virtual ~Type() = default;
  Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

// kind() == Tensor. The axes are ordered and never empty (RM 3.3); the rank equals axes().size().
class TensorType : public Type {
 public:
  explicit TensorType(std::vector<Axis> axes) : Type(Kind::Tensor), axes_(std::move(axes)) {}
  const std::vector<Axis>& axes() const { return axes_; }
  size_t rank() const { return axes_.size(); }

 private:
  std::vector<Axis> axes_;
};

// kind() == Scalar: an f32 scalar (a bare floatLiteral operand, RM 3.8).
class ScalarType : public Type {
 public:
  ScalarType() : Type(Kind::Scalar) {}
};

// kind() == Invalid: only used as semantic analysis's error-recovery
// poison value (ARCH-007). This never appears on a Value in a graph
// that GraphVerifier accepts as clean.
class InvalidType : public Type {
 public:
  InvalidType() : Type(Kind::Invalid) {}
};

// Owns every Type object, and the SymbolId <-> name table, for one
// compilation unit. Symbol interning applies across the whole program.
// (RM 3.4's "within one graph" scoping applies to DimSolver's
// unification state, which starts fresh for each graph -- see
// DimSolver.h -- not to naming.)
class TypeContext {
 public:
  const TensorType* tensor(std::vector<Axis> axes) {
    tensors_.emplace_back(std::move(axes));
    return &tensors_.back();
  }
  const ScalarType* scalar() { return &scalar_; }
  const InvalidType* invalid() { return &invalid_; }

  SymbolId internSymbol(std::string_view name) {
    auto it = symbolIds_.find(std::string(name));
    if (it != symbolIds_.end()) return it->second;
    SymbolId id = static_cast<SymbolId>(symbolNames_.size());
    symbolNames_.emplace_back(name);
    symbolIds_.emplace(std::string(name), id);
    return id;
  }
  std::string_view symbolName(SymbolId id) const { return symbolNames_.at(id); }

 private:
  std::deque<TensorType> tensors_;  // Keeps stable addresses, so a TypePtr stays valid as long as this context does.
  ScalarType scalar_;
  InvalidType invalid_;
  std::vector<std::string> symbolNames_;
  std::unordered_map<std::string, SymbolId> symbolIds_;
};

// Checks type equality by value: both names AND sizes must match
// exactly. Used, for example, by relu's rule that the result type must
// equal the input type exactly (RM 3.10).
bool dimensionsEqual(const Dimension& a, const Dimension& b);
bool typesEqual(const Type* a, const Type* b);

// Renders a type or dimension for a diagnostic message, for example
// "Tensor[batch=B, feat=784]", "784", or "B".
std::string describeDimension(const Dimension& d, const TypeContext& ctx);
std::string describeType(const Type* t, const TypeContext& ctx);

}  // namespace tensora
