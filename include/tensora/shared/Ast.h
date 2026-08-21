#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensora/shared/SourceSpan.h"

// AST nodes (RM 3.2-3.3, 3.24; INTERFACES.md §2). These hold syntax
// only; no types or shapes. The parser accepts argument lists
// permissively. Checking each operator's argument count and kind is a
// semantic-analysis task (RM 3.7-3.17), not a parser task.
namespace tensora::ast {

struct DimSyntax {
  enum class K { Const, Symbol } k = K::Const;
  int64_t constant = 0;
  std::string symbol;
};

struct AxisSyntax {
  std::string name;
  DimSyntax size;
  SourceSpan span;
};

struct TensorTypeSyntax {
  std::vector<AxisSyntax> axes;
  SourceSpan span;
};

struct Decl {
  enum class Kind { Input, Weight } kind = Kind::Input;
  std::string name;
  TensorTypeSyntax type;
  SourceSpan span;
};

struct Expr {
  SourceSpan span;
  virtual ~Expr() = default;
};
using ExprPtr = std::unique_ptr<Expr>;

struct NameExpr : Expr {
  std::string name;
};

struct ScalarExpr : Expr {
  double value = 0.0;
};

struct MatMulExpr : Expr {
  ExprPtr lhs, rhs;
};

struct BinaryExpr : Expr {
  enum class Op { Add, Sub, Mul, Div } op = Op::Add;
  ExprPtr lhs, rhs;
};

struct Arg {
  std::optional<std::string> keyword;  // e.g. "stride", "axis", "perm", or none
  enum class Kind { Name, Int, Float, IntList, AxisList } kind = Kind::Name;
  std::string name;                      // Name / axis reference
  int64_t intVal = 0;
  double floatVal = 0.0;
  std::vector<int64_t> intList;          // perm=[...]
  std::vector<AxisSyntax> axisList;      // reshape target [batch=B, head=12, dim=64]
  SourceSpan span;
};

struct CallExpr : Expr {
  std::string callee;
  std::vector<Arg> args;
};

struct Stmt {
  std::string name;
  ExprPtr rhs;
  SourceSpan span;
};

struct GraphDecl {
  std::string name;
  std::vector<Stmt> stmts;
  std::string returnName;
  SourceSpan span;
};

struct AstProgram {
  std::vector<Decl> decls;
  std::vector<GraphDecl> graphs;
};

}  // namespace tensora::ast
