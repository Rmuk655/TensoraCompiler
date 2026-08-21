#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Ir.h"
#include "tensora/shared/DimSolver.h"
#include "tensora/shared/Type.h"

// One contract checker per operator (RM 3.7-3.17; ARCHITECTURE.md §6.2).
// Each is an attribute-grammar rule: it builds the result TensorType
// from the operand types, or it reports a diagnostic and returns
// ctx.types.invalid() (ARCH-007 poisoning). SemanticAnalyzer's dispatch
// layer checks argument *count* and *kind* (positional vs. keyword,
// which keyword names are valid) before it calls a contract. So each
// contract always gets arguments with the right count and kind, and it
// only needs to check shapes and types.
namespace tensora {

struct ContractCtx {
  TypeContext& types;
  DimSolver& solver;
  DiagnosticEngine& diags;
  SourceSpan span;
};

const Type* checkMatMul(const Type* a, const Type* b, ContractCtx& ctx);
const Type* checkLinear(const Type* x, const Type* w, const Type* b, ContractCtx& ctx);
const Type* checkElementwise(ast::BinaryExpr::Op op, const Type* a, const Type* bOrScalar, ContractCtx& ctx);
const Type* checkRelu(const Type* x, ContractCtx& ctx);
const Type* checkSoftmax(const Type* x, std::string_view axis, ContractCtx& ctx);
const Type* checkReshape(const Type* x, const std::vector<Axis>& target, ContractCtx& ctx);
const Type* checkFlatten(const Type* x, ContractCtx& ctx);
const Type* checkTranspose(const Type* x, const std::vector<uint32_t>& perm, ContractCtx& ctx);
// sumOrMean must be OpKind::Sum or OpKind::Mean.
const Type* checkReduce(OpKind sumOrMean, const Type* x, std::string_view axis, ContractCtx& ctx);
const Type* checkConv2D(const Type* x, const Type* w, const Type* b, int64_t stride, int64_t padding,
                         ContractCtx& ctx);
// The caller already resolves `stride` before this call (RM 3.17: it
// defaults to `size` when the call omits it), so this contract always
// gets a concrete value.
const Type* checkMaxPool(const Type* x, int64_t size, int64_t stride, ContractCtx& ctx);

}  // namespace tensora
