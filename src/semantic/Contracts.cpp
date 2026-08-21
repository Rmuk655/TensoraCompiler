// This file implements the operator contract checkers.
// Each function checks one Tensora operator (matmul, linear, conv2d,
// and so on) and returns its verified result type. SemanticAnalyzer
// calls these functions while it checks a graph.
#include "tensora/semantic/Contracts.h"

#include <algorithm>
#include <optional>
#include <sstream>

namespace tensora {

namespace {

const Type* fail(ContractCtx& ctx, DiagCode code, std::string message) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = code;
  d.span = ctx.span;
  d.message = std::move(message);
  ctx.diags.report(d);
  return ctx.types.invalid();
}

bool isInvalid(const Type* t) { return t->kind() == Type::Kind::Invalid; }

// Returns the operand as a TensorType. If it is not one, this reports
// SemaTypeMismatch and returns nullptr. It never reports on an already-
// Invalid operand, since that error was already reported once
// (poisoning).
const TensorType* requireTensor(const Type* t, std::string_view what, ContractCtx& ctx) {
  if (isInvalid(t)) return nullptr;
  if (t->kind() != Type::Kind::Tensor) {
    fail(ctx, DiagCode::SemaTypeMismatch,
         std::string(what) + " requires a tensor operand; found " + describeType(t, ctx.types));
    return nullptr;
  }
  return static_cast<const TensorType*>(t);
}

std::string axesToString(const std::vector<Axis>& axes, ContractCtx& ctx) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i) out << ", ";
    out << axes[i].name << "=" << describeDimension(axes[i].size, ctx.types);
  }
  out << "]";
  return out.str();
}

// RM 3.8's trailing-align broadcast rule. checkElementwise uses this
// on the full axis lists, and checkMatMul uses it on just the
// leading (batch) axes. On an incompatibility, this reports a
// diagnostic and returns nullopt.
std::optional<std::vector<Axis>> broadcastAxes(const std::vector<Axis>& a, const std::vector<Axis>& b,
                                                ContractCtx& ctx) {
  size_t ra = a.size(), rb = b.size();
  size_t n = std::max(ra, rb);
  std::vector<Axis> result(n);

  for (size_t i = 0; i < n; ++i) {  // i counts positions from the right: 0 = last axis
    bool hasA = i < ra, hasB = i < rb;
    const Axis* axA = hasA ? &a[ra - 1 - i] : nullptr;
    const Axis* axB = hasB ? &b[rb - 1 - i] : nullptr;

    if (axA && !axB) {
      result[n - 1 - i] = *axA;  // shorter side is implicitly padded (size 1, name from the longer side)
      continue;
    }
    if (axB && !axA) {
      result[n - 1 - i] = *axB;
      continue;
    }

    Dimension da = ctx.solver.resolve(axA->size);
    Dimension db = ctx.solver.resolve(axB->size);
    bool aIsOne = da.kind == Dimension::Kind::Const && da.constant == 1;
    bool bIsOne = db.kind == Dimension::Kind::Const && db.constant == 1;

    if (!aIsOne && !bIsOne && axA->name != axB->name) {
      fail(ctx, DiagCode::SemaAxisNameMismatch,
           "axis " + std::to_string(n - 1 - i) + " is '" + axA->name + "' on one operand but '" + axB->name +
               "' on the other, and neither side is a size-1 broadcast axis");
      return std::nullopt;
    }

    if (aIsOne) {
      result[n - 1 - i] = *axB;
    } else if (bIsOne) {
      result[n - 1 - i] = *axA;
    } else if (ctx.solver.unify(axA->size, axB->size)) {
      result[n - 1 - i] = *axA;
    } else {
      fail(ctx, DiagCode::SemaBroadcast,
           "incompatible sizes for axis '" + axA->name + "': " + describeDimension(ctx.solver.resolve(axA->size), ctx.types) +
               " != " + describeDimension(ctx.solver.resolve(axB->size), ctx.types));
      return std::nullopt;
    }
  }
  return result;
}

}  // namespace

// --- RM 3.7: matrix multiplication (@) ---
const Type* checkMatMul(const Type* aType, const Type* bType, ContractCtx& ctx) {
  if (isInvalid(aType) || isInvalid(bType)) return ctx.types.invalid();
  const TensorType* a = requireTensor(aType, "matmul (@)", ctx);
  const TensorType* b = requireTensor(bType, "matmul (@)", ctx);
  if (!a || !b) return ctx.types.invalid();

  if (a->rank() < 2 || b->rank() < 2) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "matmul (@) requires rank >= 2 on both operands; found rank " + std::to_string(a->rank()) +
                    " and rank " + std::to_string(b->rank()));
  }

  const Axis& leftM = a->axes()[a->rank() - 2];
  const Axis& leftK = a->axes()[a->rank() - 1];
  const Axis& rightK = b->axes()[b->rank() - 2];
  const Axis& rightN = b->axes()[b->rank() - 1];

  if (!ctx.solver.unify(leftK.size, rightK.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "cannot multiply " + describeType(a, ctx.types) + " by " + describeType(b, ctx.types) +
                    ": contraction dims " + describeDimension(ctx.solver.resolve(leftK.size), ctx.types) +
                    " != " + describeDimension(ctx.solver.resolve(rightK.size), ctx.types));
  }

  std::vector<Axis> aLead(a->axes().begin(), a->axes().end() - 2);
  std::vector<Axis> bLead(b->axes().begin(), b->axes().end() - 2);
  auto lead = broadcastAxes(aLead, bLead, ctx);
  if (!lead) return ctx.types.invalid();

  std::vector<Axis> result = std::move(*lead);
  result.push_back(leftM);
  result.push_back(rightN);
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.9: linear ---
const Type* checkLinear(const Type* xType, const Type* wType, const Type* bType, ContractCtx& ctx) {
  if (isInvalid(xType) || isInvalid(wType) || isInvalid(bType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "linear", ctx);
  const TensorType* w = requireTensor(wType, "linear", ctx);
  const TensorType* bias = requireTensor(bType, "linear", ctx);
  if (!x || !w || !bias) return ctx.types.invalid();

  if (x->rank() < 1) {
    return fail(ctx, DiagCode::SemaRankMismatch, "linear's input must have rank >= 1");
  }
  if (w->rank() != 2) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "linear's weight must be rank 2 (Tensor[in, out]); found " + describeType(w, ctx.types));
  }
  if (bias->rank() != 1) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "linear's bias must be rank 1 (Tensor[out]); found " + describeType(bias, ctx.types));
  }

  const Axis& xK = x->axes().back();
  const Axis& wIn = w->axes()[0];
  const Axis& wOut = w->axes()[1];
  const Axis& biasAxis = bias->axes()[0];

  if (!ctx.solver.unify(xK.size, wIn.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "linear: input's trailing dim " + describeDimension(ctx.solver.resolve(xK.size), ctx.types) +
                    " != weight's in dim " + describeDimension(ctx.solver.resolve(wIn.size), ctx.types));
  }
  if (biasAxis.name != wOut.name) {
    return fail(ctx, DiagCode::SemaAxisNameMismatch,
                "linear: bias axis '" + biasAxis.name + "' does not match weight's out axis '" + wOut.name + "'");
  }
  if (!ctx.solver.unify(biasAxis.size, wOut.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "linear: bias size " + describeDimension(ctx.solver.resolve(biasAxis.size), ctx.types) +
                    " != weight's out size " + describeDimension(ctx.solver.resolve(wOut.size), ctx.types));
  }

  std::vector<Axis> result(x->axes().begin(), x->axes().end() - 1);
  result.push_back(Axis{wOut.name, wOut.size});
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.8: elementwise arithmetic and broadcasting ---
const Type* checkElementwise(ast::BinaryExpr::Op /*op*/, const Type* aType, const Type* bOrScalarType,
                              ContractCtx& ctx) {
  if (isInvalid(aType) || isInvalid(bOrScalarType)) return ctx.types.invalid();
  const TensorType* a = requireTensor(aType, "elementwise operator", ctx);
  if (!a) return ctx.types.invalid();

  if (bOrScalarType->kind() == Type::Kind::Scalar) {
    return a;  // tensor-scalar: applies elementwise, no name/size constraint (RM 3.8)
  }
  const TensorType* b = requireTensor(bOrScalarType, "elementwise operator", ctx);
  if (!b) return ctx.types.invalid();

  auto result = broadcastAxes(a->axes(), b->axes(), ctx);
  if (!result) return ctx.types.invalid();
  return ctx.types.tensor(std::move(*result));
}

// --- RM 3.10: relu ---
const Type* checkRelu(const Type* xType, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "relu", ctx);
  if (!x) return ctx.types.invalid();
  return x;  // result type equals input type exactly
}

// --- RM 3.11: softmax ---
const Type* checkSoftmax(const Type* xType, std::string_view axis, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "softmax", ctx);
  if (!x) return ctx.types.invalid();

  bool found = std::any_of(x->axes().begin(), x->axes().end(),
                            [&](const Axis& ax) { return ax.name == axis; });
  if (!found) {
    return fail(ctx, DiagCode::SemaAxisNotFound,
                "softmax: axis '" + std::string(axis) + "' not found in " + describeType(x, ctx.types));
  }
  return x;  // result type equals input type exactly (RM 3.11)
}

// --- RM 3.12: reshape ---
const Type* checkReshape(const Type* xType, const std::vector<Axis>& target, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "reshape", ctx);
  if (!x) return ctx.types.invalid();

  std::vector<Dimension> srcDims, dstDims;
  for (const Axis& ax : x->axes()) srcDims.push_back(ax.size);
  for (const Axis& ax : target) dstDims.push_back(ax.size);

  ElementCountResult r = ctx.solver.sameElementCount(srcDims, dstDims);
  if (r.status == ElementCountResult::Status::Unresolvable) {
    return fail(ctx, DiagCode::SemaUnresolvableShape,
                "reshape: cannot statically resolve whether " + describeType(x, ctx.types) + " and " +
                    axesToString(target, ctx) + " have the same element count (symbolic factors do not match)");
  }
  if (r.status == ElementCountResult::Status::Mismatch) {
    return fail(ctx, DiagCode::SemaReshapeCount,
                "reshape: element count " + std::to_string(r.rhsConstProduct) + " != " +
                    std::to_string(r.lhsConstProduct) + " (source " + describeType(x, ctx.types) + " -> target " +
                    axesToString(target, ctx) + ")");
  }
  return ctx.types.tensor(target);
}

// --- RM 3.13: flatten ---
const Type* checkFlatten(const Type* xType, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "flatten", ctx);
  if (!x) return ctx.types.invalid();

  if (x->rank() < 2) {
    return fail(ctx, DiagCode::SemaRankMismatch, "flatten requires rank >= 2; found " + describeType(x, ctx.types));
  }

  int64_t product = 1;
  for (size_t i = 1; i < x->rank(); ++i) {
    Dimension d = ctx.solver.resolve(x->axes()[i].size);
    if (d.kind != Dimension::Kind::Const) {
      return fail(ctx, DiagCode::SemaUnresolvableShape,
                  "flatten: trailing axis '" + x->axes()[i].name + "' must have a concrete size; found symbolic '" +
                      describeDimension(d, ctx.types) + "'");
    }
    product *= d.constant;
  }

  std::vector<Axis> result;
  result.push_back(x->axes()[0]);
  result.push_back(Axis{"flat", Dimension::makeConst(product)});
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.14: transpose ---
const Type* checkTranspose(const Type* xType, const std::vector<uint32_t>& perm, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "transpose", ctx);
  if (!x) return ctx.types.invalid();

  size_t rank = x->rank();
  if (perm.size() != rank) {
    return fail(ctx, DiagCode::SemaTransposePerm,
                "transpose: perm has " + std::to_string(perm.size()) + " entries but the input has rank " +
                    std::to_string(rank));
  }
  std::vector<bool> seen(rank, false);
  for (uint32_t p : perm) {
    if (p >= rank) {
      return fail(ctx, DiagCode::SemaTransposePerm,
                  "transpose: perm index " + std::to_string(p) + " is out of range for rank " +
                      std::to_string(rank));
    }
    if (seen[p]) {
      return fail(ctx, DiagCode::SemaTransposePerm, "transpose: perm index " + std::to_string(p) + " repeated");
    }
    seen[p] = true;
  }

  std::vector<Axis> result;
  result.reserve(rank);
  for (uint32_t p : perm) result.push_back(x->axes()[p]);
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.15: sum / mean ---
const Type* checkReduce(OpKind sumOrMean, const Type* xType, std::string_view axis, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(
      xType, sumOrMean == OpKind::Sum ? "sum" : "mean", ctx);
  if (!x) return ctx.types.invalid();

  auto it = std::find_if(x->axes().begin(), x->axes().end(), [&](const Axis& ax) { return ax.name == axis; });
  if (it == x->axes().end()) {
    return fail(ctx, DiagCode::SemaAxisNotFound,
                std::string(sumOrMean == OpKind::Sum ? "sum" : "mean") + ": axis '" + std::string(axis) +
                    "' not found in " + describeType(x, ctx.types));
  }
  if (x->rank() == 1) {
    // RM 3.3 says a TensorType's axis list is never empty. Reducing
    // the only axis of a rank-1 tensor would need a rank-0 result. A
    // TensorType cannot represent that at all, not even as an empty
    // result.
    return fail(ctx, DiagCode::SemaRankMismatch,
                std::string(sumOrMean == OpKind::Sum ? "sum" : "mean") +
                    ": cannot reduce the only axis of a rank-1 tensor (result would have rank 0)");
  }

  std::vector<Axis> result;
  result.reserve(x->rank() - 1);
  for (auto axIt = x->axes().begin(); axIt != x->axes().end(); ++axIt) {
    if (axIt != it) result.push_back(*axIt);
  }
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.16: conv2d ---
const Type* checkConv2D(const Type* xType, const Type* wType, const Type* bType, int64_t stride, int64_t padding,
                         ContractCtx& ctx) {
  if (isInvalid(xType) || isInvalid(wType) || isInvalid(bType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "conv2d", ctx);
  const TensorType* w = requireTensor(wType, "conv2d", ctx);
  const TensorType* bias = requireTensor(bType, "conv2d", ctx);
  if (!x || !w || !bias) return ctx.types.invalid();

  if (x->rank() != 4) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "conv2d's input must be rank 4 (Tensor[batch,channel,height,width]); found " +
                    describeType(x, ctx.types));
  }
  if (w->rank() != 4) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "conv2d's weight must be rank 4 (Tensor[out_channel,in_channel,kh,kw]); found " +
                    describeType(w, ctx.types));
  }
  if (bias->rank() != 1) {
    return fail(ctx, DiagCode::SemaRankMismatch, "conv2d's bias must be rank 1; found " + describeType(bias, ctx.types));
  }

  const Axis& xBatch = x->axes()[0];
  const Axis& xChan = x->axes()[1];
  const Axis& xH = x->axes()[2];
  const Axis& xW = x->axes()[3];
  const Axis& wOutC = w->axes()[0];
  const Axis& wInC = w->axes()[1];
  const Axis& wKh = w->axes()[2];
  const Axis& wKw = w->axes()[3];

  if (!ctx.solver.unify(xChan.size, wInC.size)) {
    return fail(ctx, DiagCode::SemaConvChannel,
                "conv2d: input channel " + describeDimension(ctx.solver.resolve(xChan.size), ctx.types) +
                    " != weight in_channel " + describeDimension(ctx.solver.resolve(wInC.size), ctx.types));
  }
  if (!ctx.solver.unify(bias->axes()[0].size, wOutC.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "conv2d: bias size " + describeDimension(ctx.solver.resolve(bias->axes()[0].size), ctx.types) +
                    " != weight out_channel " + describeDimension(ctx.solver.resolve(wOutC.size), ctx.types));
  }

  Dimension h = ctx.solver.resolve(xH.size);
  Dimension wd = ctx.solver.resolve(xW.size);
  Dimension kh = ctx.solver.resolve(wKh.size);
  Dimension kw = ctx.solver.resolve(wKw.size);
  if (h.kind != Dimension::Kind::Const || wd.kind != Dimension::Kind::Const ||
      kh.kind != Dimension::Kind::Const || kw.kind != Dimension::Kind::Const) {
    return fail(ctx, DiagCode::SemaUnresolvableShape,
                "conv2d: spatial arithmetic requires concrete height/width/kernel sizes; found a symbolic dimension");
  }
  if (stride <= 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic, "conv2d: stride must be a positive integer; found " +
                                                        std::to_string(stride));
  }

  int64_t hNum = h.constant + 2 * padding - kh.constant;
  int64_t wNum = wd.constant + 2 * padding - kw.constant;
  if (hNum < 0 || hNum % stride != 0 || wNum < 0 || wNum % stride != 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic,
                "conv2d: non-dividing spatial arithmetic for height=" + std::to_string(h.constant) +
                    ", width=" + std::to_string(wd.constant) + ", kernel=(" + std::to_string(kh.constant) + "," +
                    std::to_string(kw.constant) + "), stride=" + std::to_string(stride) +
                    ", padding=" + std::to_string(padding));
  }
  int64_t hout = hNum / stride + 1;
  int64_t wout = wNum / stride + 1;

  // RM 3.16's schema reuses the *input's* axis name at each position
  // (batch=B on both sides, channel=Cin -> channel=Cout, and so on).
  // The weight only supplies the new Cout *size*, not a new axis
  // name.
  std::vector<Axis> result = {
      Axis{xBatch.name, xBatch.size},
      Axis{xChan.name, wOutC.size},
      Axis{xH.name, Dimension::makeConst(hout)},
      Axis{xW.name, Dimension::makeConst(wout)},
  };
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.17: maxpool ---
const Type* checkMaxPool(const Type* xType, int64_t size, int64_t stride, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "maxpool", ctx);
  if (!x) return ctx.types.invalid();

  if (x->rank() != 4) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "maxpool's input must be rank 4 (Tensor[batch,channel,height,width]); found " +
                    describeType(x, ctx.types));
  }

  const Axis& xBatch = x->axes()[0];
  const Axis& xChan = x->axes()[1];
  const Axis& xH = x->axes()[2];
  const Axis& xW = x->axes()[3];

  Dimension h = ctx.solver.resolve(xH.size);
  Dimension wd = ctx.solver.resolve(xW.size);
  if (h.kind != Dimension::Kind::Const || wd.kind != Dimension::Kind::Const) {
    return fail(ctx, DiagCode::SemaUnresolvableShape,
                "maxpool: spatial arithmetic requires concrete height/width; found a symbolic dimension");
  }
  if (stride <= 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic, "maxpool: stride must be a positive integer; found " +
                                                        std::to_string(stride));
  }

  int64_t hNum = h.constant - size;   // maxpool == conv2d's formula with padding=0
  int64_t wNum = wd.constant - size;
  if (hNum < 0 || hNum % stride != 0 || wNum < 0 || wNum % stride != 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic,
                "maxpool: non-dividing spatial arithmetic for height=" + std::to_string(h.constant) +
                    ", width=" + std::to_string(wd.constant) + ", size=" + std::to_string(size) +
                    ", stride=" + std::to_string(stride));
  }
  int64_t hout = hNum / stride + 1;
  int64_t wout = wNum / stride + 1;

  std::vector<Axis> result = {
      Axis{xBatch.name, xBatch.size},
      Axis{xChan.name, xChan.size},
      Axis{xH.name, Dimension::makeConst(hout)},
      Axis{xW.name, Dimension::makeConst(wout)},
  };
  return ctx.types.tensor(std::move(result));
}

}  // namespace tensora
