// This file implements the codegen phase's lowering step.
// It converts a verified, optimized Typed Computation Graph to MLIR
// (func/linalg/tensor/arith/scf). Standard MLIR/LLVM passes take the
// MLIR this file produces the rest of the way to native code.
#include "tensora/codegen/MlirLowerer.h"

#include <unordered_map>
#include <unordered_set>

#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace tensora {

using namespace mlir;
// tensora::Value and tensora::Type (shared/Ir.h, shared/Type.h) share
// their names with mlir::Value and mlir::Type. The enclosing
// `namespace tensora` always wins over `using namespace mlir` for a
// bare name. So every MLIR value or type below must use these aliases
// instead of the bare names.
using MlirValue = mlir::Value;
using MlirType = mlir::Type;

namespace {

// ARCH-005: concrete dim -> static; symbolic dim -> dynamic (?).
RankedTensorType lowerTensorType(const TensorType* t, OpBuilder& b) {
  SmallVector<int64_t> shape;
  shape.reserve(t->rank());
  for (const Axis& ax : t->axes()) {
    shape.push_back(ax.size.kind == Dimension::Kind::Const ? ax.size.constant : ShapedType::kDynamic);
  }
  return RankedTensorType::get(shape, b.getF32Type());
}

MlirValue zeroFilledAccumulator(OpBuilder& b, Location loc, RankedTensorType type, ValueRange dynSizes) {
  MlirValue empty = tensor::EmptyOp::create(b, loc, type, dynSizes);
  MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  return linalg::FillOp::create(b, loc, TypeRange{type}, ValueRange{zero}, ValueRange{empty}).getResult(0);
}

// Right-aligns `operandType` to `outRank` (RM 3.8 trailing-axis
// alignment). Operand axis i maps to output axis (i + rankDiff). If
// semantic analysis marked an operand axis as the size-1 broadcast
// side, that axis maps to a constant-0 affine expression instead of a
// real dimension. This matches exactly which side semantic analysis
// verified as broadcastable. This function does not re-check
// broadcast legality; it only replays what semantic analysis already
// proved.
AffineMap broadcastOperandMap(MLIRContext* ctx, const TensorType* operandType, size_t outRank) {
  size_t inRank = operandType->rank();
  size_t rankDiff = outRank - inRank;
  SmallVector<AffineExpr> exprs(inRank);
  for (size_t i = 0; i < inRank; ++i) {
    const Dimension& d = operandType->axes()[i].size;
    bool isBroadcastOne = (d.kind == Dimension::Kind::Const && d.constant == 1);
    exprs[i] = isBroadcastOne ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i + rankDiff, ctx);
  }
  return AffineMap::get(outRank, 0, exprs, ctx);
}

SmallVector<utils::IteratorType> parallelIterators(size_t rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

MlirValue elementwiseBinaryOp(OpBuilder& b, Location loc, OpKind op, MlirValue lhs, MlirValue rhs) {
  switch (op) {
    case OpKind::Add: return arith::AddFOp::create(b, loc, lhs, rhs);
    case OpKind::Sub: return arith::SubFOp::create(b, loc, lhs, rhs);
    case OpKind::Mul: return arith::MulFOp::create(b, loc, lhs, rhs);
    case OpKind::Div: return arith::DivFOp::create(b, loc, lhs, rhs);
    default: return lhs;
  }
}

// The same shape as `t`, except axis `axisIdx` has size 1. This is the
// "keep-dims" shape of a reduction along that one axis (RM 3.10/3.15:
// softmax, sum, and mean each reduce exactly one named axis).
RankedTensorType withAxisSizeOne(RankedTensorType t, size_t axisIdx) {
  SmallVector<int64_t> shape(t.getShape().begin(), t.getShape().end());
  shape[axisIdx] = 1;
  return RankedTensorType::get(shape, t.getElementType());
}

// An identity map on every axis except `axisIdx`. That one axis reads
// and writes its one (size-1) slot, instead of a per-iteration index.
// This lets a keep-dims reduction result broadcast back against the
// full-rank tensor it was reduced from, using the same generic-op
// broadcast mechanism as `broadcastOperandMap` above. This is that
// mechanism's reduction-axis case.
AffineMap keepDimMap(MLIRContext* ctx, size_t rank, size_t axisIdx) {
  SmallVector<AffineExpr> exprs(rank);
  for (size_t i = 0; i < rank; ++i) {
    exprs[i] = (i == axisIdx) ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i, ctx);
  }
  return AffineMap::get(rank, 0, exprs, ctx);
}

// Reduces `x` (rank R, shape `xType`) along `axisIdx` to a keep-dims
// (size-1-at-axisIdx) result, by taking the max or the sum. These are
// the two reductions softmax needs (RM 3.11's numerically-stable
// method: subtract the max, then exponentiate, then normalize by the
// sum).
MlirValue reduceAlongAxis(OpBuilder& b, Location loc, MlirValue x, RankedTensorType xType, size_t axisIdx,
                          bool isMax) {
  size_t rank = xType.getRank();
  RankedTensorType redType = withAxisSizeOne(xType, axisIdx);
  MlirValue init;
  if (isMax) {
    MlirValue negInf =
        arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    MlirValue empty = tensor::EmptyOp::create(b, loc, redType, ValueRange{});
    init = linalg::FillOp::create(b, loc, TypeRange{redType}, ValueRange{negInf}, ValueRange{empty}).getResult(0);
  } else {
    init = zeroFilledAccumulator(b, loc, redType, {});
  }
  AffineMap inMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  AffineMap outMap = keepDimMap(b.getContext(), rank, axisIdx);
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  iters[axisIdx] = utils::IteratorType::reduction;
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{redType}, ValueRange{x}, ValueRange{init}, ArrayRef<AffineMap>{inMap, outMap}, iters,
      [&](OpBuilder& body, Location l, ValueRange args) {
        MlirValue result = isMax ? MlirValue(arith::MaximumFOp::create(body, l, args[0], args[1]))
                                  : MlirValue(arith::AddFOp::create(body, l, args[0], args[1]));
        linalg::YieldOp::create(body, l, ValueRange{result});
      });
  return generic.getResult(0);
}

// Applies a binary scalar op between full-rank `x` and a keep-dims
// (size-1-at-axisIdx) `reduced`, and broadcasts `reduced` back out
// along that axis. This is softmax's "subtract the max" and "divide by
// the sum" step.
MlirValue broadcastAxisBinary(OpBuilder& b, Location loc, MlirValue x, MlirValue reduced, RankedTensorType xType,
                              size_t axisIdx, OpKind op) {
  size_t rank = xType.getRank();
  MlirValue init = tensor::EmptyOp::create(b, loc, xType, ValueRange{});
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  AffineMap redMap = keepDimMap(b.getContext(), rank, axisIdx);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{xType}, ValueRange{x, reduced}, ValueRange{init}, ArrayRef<AffineMap>{idMap, redMap, idMap},
      parallelIterators(rank), [&](OpBuilder& body, Location l, ValueRange args) {
        MlirValue result = elementwiseBinaryOp(body, l, op, args[0], args[1]);
        linalg::YieldOp::create(body, l, ValueRange{result});
      });
  return generic.getResult(0);
}

MlirValue lowerExp(OpBuilder& b, Location loc, MlirValue x, RankedTensorType xType) {
  size_t rank = xType.getRank();
  MlirValue init = tensor::EmptyOp::create(b, loc, xType, ValueRange{});
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{xType}, ValueRange{x}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
      parallelIterators(rank), [&](OpBuilder& body, Location l, ValueRange args) {
        MlirValue e = math::ExpOp::create(body, l, args[0]);
        linalg::YieldOp::create(body, l, ValueRange{e});
      });
  return generic.getResult(0);
}

size_t findAxisIndex(const TensorType* t, const std::string& name) {
  for (size_t i = 0; i < t->rank(); ++i) {
    if (t->axes()[i].name == name) return i;
  }
  llvm_unreachable("MlirLowerer: axis name not found (sema's checkSoftmax/checkReduce already verified it exists)");
}

// Softmax's keep-dims reduction keeps the same rank, with the axis
// size set to 1. RM 3.15's sum/mean instead *removes* the reduced axis
// entirely (rank - 1). The output affine map's domain still spans the
// full input rank, since linalg.generic iterates over the input's
// index space, but its range lists only the surviving dims, in order.
AffineMap dropAxisMap(MLIRContext* ctx, size_t inRank, size_t axisIdx) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(inRank - 1);
  for (size_t i = 0; i < inRank; ++i) {
    if (i != axisIdx) exprs.push_back(getAffineDimExpr(i, ctx));
  }
  return AffineMap::get(inRank, 0, exprs, ctx);
}

class LowererImpl {
 public:
  explicit LowererImpl(MLIRContext& ctx, const TilingDecision& tilingDecision)
      : ctx_(ctx), tilingDecision_(tilingDecision) {}

  ModuleOp lower(const TypedProgram& program) {
    OpBuilder builder(&ctx_);
    Location loc = builder.getUnknownLoc();
    ModuleOp module = ModuleOp::create(builder, loc);
    builder.setInsertionPointToEnd(module.getBody());
    for (const TypedGraph& graph : program.graphs()) lowerGraph(graph, builder, loc);
    return module;
  }

 private:
  void lowerGraph(const TypedGraph& graph, OpBuilder& moduleBuilder, Location loc) {
    // Computed once per graph and read by tilingDecision_.decide() below --
    // this is what lets the Level-2 tiling decision actually see peak
    // live-memory pressure (ARCH-009), instead of always taking the
    // liveness=nullptr default.
    currentLiveness_ = LivenessAnalysis().run(graph);

    SmallVector<MlirType> argTypes;
    for (ValueId p : graph.params) {
      argTypes.push_back(lowerTensorType(static_cast<const TensorType*>(graph.value(p).type), moduleBuilder));
    }
    MlirType resultType =
        lowerTensorType(static_cast<const TensorType*>(graph.value(graph.output).type), moduleBuilder);
    auto funcType = moduleBuilder.getFunctionType(argTypes, {resultType});
    auto func = func::FuncOp::create(moduleBuilder, loc, graph.name, funcType);
    func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), moduleBuilder.getUnitAttr());

    Block* entry = func.addEntryBlock();
    OpBuilder b(entry, entry->begin());

    std::unordered_map<ValueId, MlirValue> values;
    for (size_t i = 0; i < graph.params.size(); ++i) values[graph.params[i]] = entry->getArgument(i);

    std::unordered_set<NodeId> alreadyLowered;  // e.g. a Relu consumed by fusion above
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
      auto nid = static_cast<NodeId>(i);
      if (alreadyLowered.count(nid)) continue;
      const GraphNode& node = graph.node(nid);
      values[node.output] = lowerNode(graph, node, values, b, loc, alreadyLowered);
    }

    func::ReturnOp::create(b, loc, ValueRange{values.at(graph.output)});
  }

  MlirValue lowerNode(const TypedGraph& graph, const GraphNode& node, std::unordered_map<ValueId, MlirValue>& values,
                      OpBuilder& b, Location loc, std::unordered_set<NodeId>& alreadyLowered) {
    const auto* outType = static_cast<const TensorType*>(graph.value(node.output).type);
    RankedTensorType mlirOutType = lowerTensorType(outType, b);

    switch (node.op) {
      case OpKind::Add:
      case OpKind::Sub:
      case OpKind::Mul:
      case OpKind::Div:
        return lowerElementwise(graph, node, values, b, loc, mlirOutType);
      case OpKind::Relu:
        return lowerRelu(values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::MatMul: {
        const auto* aType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
        const auto* bType = static_cast<const TensorType*>(graph.value(node.inputs[1]).type);
        MlirValue a = values.at(node.inputs[0]);
        MlirValue rhsVal = values.at(node.inputs[1]);
        if (aType->rank() == 2 && bType->rank() == 2) {
          TilingPlan plan = tilingDecision_.decide(node, graph, &currentLiveness_);
          return lowerMatMul(a, rhsVal, b, loc, mlirOutType, plan);
        }
        return lowerBatchedMatMul(a, rhsVal, aType, bType, b, loc, mlirOutType);
      }
      case OpKind::Linear:
        return lowerLinear(graph, node, values, b, loc, mlirOutType, alreadyLowered);
      case OpKind::Softmax:
        return lowerSoftmax(graph, node, values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::Flatten:
        return lowerFlatten(values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::Transpose:
        return lowerTranspose(values.at(node.inputs[0]), node.attrs.perm, b, loc, mlirOutType);
      case OpKind::Conv2D:
        return lowerConv2D(node, values.at(node.inputs[0]), values.at(node.inputs[1]), values.at(node.inputs[2]), b,
                            loc, mlirOutType);
      case OpKind::Sum:
      case OpKind::Mean:
        return lowerReduce(graph, node, values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::MaxPool:
        return lowerMaxPool(node, values.at(node.inputs[0]), b, loc, mlirOutType);
      default:
        // This op is not lowered yet. See docs/testing/TEST_MATRIX.md
        // LOW-* and the final codegen report's Known Limitations.
        llvm_unreachable("MlirLowerer: unsupported OpKind for this milestone");
    }
  }

  // RM 3.8: a tensor-scalar op has no shape constraint, so this lowers
  // it as one uniform map over the tensor. A tensor-tensor op uses the
  // broadcast affine maps above, which replay the alignment semantic
  // analysis already verified.
  MlirValue lowerElementwise(const TypedGraph& graph, const GraphNode& node,
                             std::unordered_map<ValueId, MlirValue>& values, OpBuilder& b, Location loc,
                             RankedTensorType mlirOutType) {
    MlirValue lhs = values.at(node.inputs[0]);
    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});

    if (node.attrs.scalar.has_value()) {
      MlirValue scalar = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(*node.attrs.scalar));
      AffineMap idMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
      auto generic = linalg::GenericOp::create(
          b, loc, TypeRange{mlirOutType}, ValueRange{lhs}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
          parallelIterators(outRank), [&](OpBuilder& body, Location l, ValueRange args) {
            MlirValue result = elementwiseBinaryOp(body, l, node.op, args[0], scalar);
            linalg::YieldOp::create(body, l, ValueRange{result});
          });
      return generic.getResult(0);
    }

    MlirValue rhs = values.at(node.inputs[1]);
    const auto* lhsType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
    const auto* rhsType = static_cast<const TensorType*>(graph.value(node.inputs[1]).type);
    AffineMap lhsMap = broadcastOperandMap(b.getContext(), lhsType, outRank);
    AffineMap rhsMap = broadcastOperandMap(b.getContext(), rhsType, outRank);
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{lhs, rhs}, ValueRange{init},
        ArrayRef<AffineMap>{lhsMap, rhsMap, outMap}, parallelIterators(outRank),
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue result = elementwiseBinaryOp(body, l, node.op, args[0], args[1]);
          linalg::YieldOp::create(body, l, ValueRange{result});
        });
    return generic.getResult(0);
  }

  MlirValue lowerRelu(MlirValue input, OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    size_t rank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
    MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{input}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
        parallelIterators(rank), [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue result = arith::MaximumFOp::create(body, l, args[0], zero);
          linalg::YieldOp::create(body, l, ValueRange{result});
        });
    return generic.getResult(0);
  }

  // The dedicated rank-2 path (RM 3.7). This is the one path
  // MatmulTilingPass tiles, so it stays a plain linalg.matmul, instead
  // of going through the general batched-contraction lowering below.
  // `plan` comes from tilingDecision_, computed at the TCG level
  // before any MLIR exists. This function stamps `plan` onto the op as
  // an attribute, which MatmulTilingPass reads directly, instead of
  // computing its own tile sizes from raw MLIR shapes.
  MlirValue lowerMatMul(MlirValue lhs, MlirValue rhs, OpBuilder& b, Location loc, RankedTensorType mlirOutType,
                        const TilingPlan& plan) {
    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    auto matmul = linalg::MatmulOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{lhs, rhs}, ValueRange{acc});
    if (plan.tile) {
      matmul->setAttr("tensora.tile_sizes",
                       b.getDenseI64ArrayAttr({plan.tileSizes[0], plan.tileSizes[1], plan.tileSizes[2]}));
    }
    return matmul.getResult(0);
  }

  // RM 3.7's general case: rank >= 2 with broadcastable leading (batch)
  // axes. This mirrors elementwise broadcasting (broadcastOperandMap
  // above), but with M and N as the two trailing parallel dims and K
  // as the reduction dim, instead of one shared identity map. This is
  // correct for any rank, but MatmulTilingPass does not tile it (only
  // the plain rank-2 path above gets tiled). Batched matmul has no
  // performance goal in v1, only a correctness goal.
  MlirValue lowerBatchedMatMul(MlirValue lhs, MlirValue rhs, const TensorType* lhsType, const TensorType* rhsType,
                               OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    MLIRContext* ctx = b.getContext();
    size_t outRank = mlirOutType.getRank();
    size_t batchRank = outRank - 2;
    size_t mDim = batchRank, nDim = batchRank + 1, kDim = batchRank + 2;
    size_t numLoopDims = batchRank + 3;

    auto batchExprs = [&](const TensorType* t) {
      size_t tBatchRank = t->rank() - 2;
      size_t rankDiff = batchRank - tBatchRank;
      SmallVector<AffineExpr> exprs;
      for (size_t i = 0; i < tBatchRank; ++i) {
        const Dimension& d = t->axes()[i].size;
        bool isBroadcastOne = (d.kind == Dimension::Kind::Const && d.constant == 1);
        exprs.push_back(isBroadcastOne ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i + rankDiff, ctx));
      }
      return exprs;
    };

    SmallVector<AffineExpr> lhsExprs = batchExprs(lhsType);
    lhsExprs.push_back(getAffineDimExpr(mDim, ctx));
    lhsExprs.push_back(getAffineDimExpr(kDim, ctx));
    AffineMap lhsMap = AffineMap::get(numLoopDims, 0, lhsExprs, ctx);

    SmallVector<AffineExpr> rhsExprs = batchExprs(rhsType);
    rhsExprs.push_back(getAffineDimExpr(kDim, ctx));
    rhsExprs.push_back(getAffineDimExpr(nDim, ctx));
    AffineMap rhsMap = AffineMap::get(numLoopDims, 0, rhsExprs, ctx);

    SmallVector<AffineExpr> outExprs;
    for (size_t i = 0; i < batchRank; ++i) outExprs.push_back(getAffineDimExpr(i, ctx));
    outExprs.push_back(getAffineDimExpr(mDim, ctx));
    outExprs.push_back(getAffineDimExpr(nDim, ctx));
    AffineMap outMap = AffineMap::get(numLoopDims, 0, outExprs, ctx);

    SmallVector<utils::IteratorType> iters(numLoopDims, utils::IteratorType::parallel);
    iters[kDim] = utils::IteratorType::reduction;

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{lhs, rhs}, ValueRange{acc},
        ArrayRef<AffineMap>{lhsMap, rhsMap, outMap}, iters, [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue prod = arith::MulFOp::create(body, l, args[0], args[1]);
          MlirValue sum = arith::AddFOp::create(body, l, args[2], prod);
          linalg::YieldOp::create(body, l, ValueRange{sum});
        });
    return generic.getResult(0);
  }

  // RM 3.9's linear, for rank(x) == 1 or rank(x) >= 3. Every leading
  // axis of x passes straight through to the output. Unlike `@`, this
  // has no separate M axis. So this is a batched vector-matrix product
  // against the one shared (K, N) weight, not a batched matmul. Rank 2
  // stays on the dedicated linalg.matmul path (lowerMatMul), since
  // that is the one path MatmulTilingPass tiles.
  MlirValue lowerLinearContraction(MlirValue x, MlirValue w, OpBuilder& b, Location loc,
                                    RankedTensorType mlirOutType) {
    MLIRContext* ctx = b.getContext();
    size_t outRank = mlirOutType.getRank();
    size_t batchRank = outRank - 1;  // == rank(x) - 1
    size_t nDim = batchRank, kDim = batchRank + 1;
    size_t numLoopDims = batchRank + 2;

    SmallVector<AffineExpr> xExprs;
    for (size_t i = 0; i < batchRank; ++i) xExprs.push_back(getAffineDimExpr(i, ctx));
    xExprs.push_back(getAffineDimExpr(kDim, ctx));
    AffineMap xMap = AffineMap::get(numLoopDims, 0, xExprs, ctx);

    AffineMap wMap = AffineMap::get(numLoopDims, 0, {getAffineDimExpr(kDim, ctx), getAffineDimExpr(nDim, ctx)}, ctx);

    SmallVector<AffineExpr> outExprs;
    for (size_t i = 0; i < batchRank; ++i) outExprs.push_back(getAffineDimExpr(i, ctx));
    outExprs.push_back(getAffineDimExpr(nDim, ctx));
    AffineMap outMap = AffineMap::get(numLoopDims, 0, outExprs, ctx);

    SmallVector<utils::IteratorType> iters(numLoopDims, utils::IteratorType::parallel);
    iters[kDim] = utils::IteratorType::reduction;

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{x, w}, ValueRange{acc},
        ArrayRef<AffineMap>{xMap, wMap, outMap}, iters, [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue prod = arith::MulFOp::create(body, l, args[0], args[1]);
          MlirValue sum = arith::AddFOp::create(body, l, args[2], prod);
          linalg::YieldOp::create(body, l, ValueRange{sum});
        });
    return generic.getResult(0);
  }

  // Lowers x @ W + b (RM 3.9). When LinearBiasActivationFusion tagged
  // this linear and the Relu right after it with the same
  // "fused_linear_relu_*" fact, this function fuses that Relu into the
  // same linalg.generic bias-add op. It reads the graph-level tag
  // instead of finding the fusion pattern again on its own.
  MlirValue lowerLinear(const TypedGraph& graph, const GraphNode& node,
                        std::unordered_map<ValueId, MlirValue>& values, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType, std::unordered_set<NodeId>& alreadyLowered) {
    MlirValue x = values.at(node.inputs[0]);
    MlirValue w = values.at(node.inputs[1]);
    MlirValue bias = values.at(node.inputs[2]);

    // decide() only applies to an OpKind::MatMul node (ARCH-010's
    // C = A @ B reference computation). For a Linear node, it always
    // returns a no-tile plan, so this call never tiles. This function
    // reuses lowerMatMul only for its linalg.matmul emission, not for
    // its tiling behavior.
    size_t xRank = cast<RankedTensorType>(x.getType()).getRank();
    MlirValue matmulResult = xRank == 2 ? lowerMatMul(x, w, b, loc, mlirOutType, tilingDecision_.decide(node, graph, &currentLiveness_))
                                        : lowerLinearContraction(x, w, b, loc, mlirOutType);

    bool fuseRelu = false;
    NodeId reluId = 0;
    for (const std::string& fact : node.facts.all()) {
      if (fact.rfind("fused_linear_relu_", 0) != 0) continue;
      NodeId consumer = graph.consumersOf(node.output).empty() ? kNoProducer : graph.consumersOf(node.output)[0];
      if (consumer != kNoProducer && graph.node(consumer).op == OpKind::Relu &&
          graph.node(consumer).facts.has(fact)) {
        fuseRelu = true;
        reluId = consumer;
      }
    }

    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    AffineMap biasMap = AffineMap::get(outRank, 0, {getAffineDimExpr(outRank - 1, b.getContext())}, b.getContext());
    MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));

    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{matmulResult, bias}, ValueRange{init},
        ArrayRef<AffineMap>{outMap, biasMap, outMap}, parallelIterators(outRank),
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue biased = arith::AddFOp::create(body, l, args[0], args[1]);
          MlirValue result = fuseRelu ? MlirValue(arith::MaximumFOp::create(body, l, biased, zero)) : biased;
          linalg::YieldOp::create(body, l, ValueRange{result});
        });

    if (fuseRelu) {
      values[graph.node(reluId).output] = generic.getResult(0);
      alreadyLowered.insert(reluId);
    }
    return generic.getResult(0);
  }

  // RM 3.11: numerically-stable softmax along a named axis (subtract
  // the max, exponentiate, sum, then normalize). The result type
  // equals the input type exactly, matching the contract. There is no
  // new shape to compute.
  MlirValue lowerSoftmax(const TypedGraph& graph, const GraphNode& node, MlirValue x, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType) {
    const auto* xType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
    size_t axisIdx = findAxisIndex(xType, *node.attrs.axisName);

    MlirValue m = reduceAlongAxis(b, loc, x, mlirOutType, axisIdx, /*isMax=*/true);
    MlirValue shifted = broadcastAxisBinary(b, loc, x, m, mlirOutType, axisIdx, OpKind::Sub);
    MlirValue expx = lowerExp(b, loc, shifted, mlirOutType);
    MlirValue s = reduceAlongAxis(b, loc, expx, mlirOutType, axisIdx, /*isMax=*/false);
    return broadcastAxisBinary(b, loc, expx, s, mlirOutType, axisIdx, OpKind::Div);
  }

  // RM 3.13: flatten keeps axis 0 (batch) and collapses the rest into
  // one axis. This is exactly `tensor.collapse_shape`'s reassociation
  // [[0],[1..rank-1]].
  MlirValue lowerFlatten(MlirValue x, OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    auto xType = cast<RankedTensorType>(x.getType());
    size_t rank = xType.getRank();
    ReassociationIndices restGroup;
    for (size_t i = 1; i < rank; ++i) restGroup.push_back(static_cast<int64_t>(i));
    SmallVector<ReassociationIndices> reassoc{ReassociationIndices{0}, restGroup};
    return tensor::CollapseShapeOp::create(b, loc, mlirOutType, x, reassoc);
  }

  // RM 3.14: transpose lowers to `linalg.transpose`. MLIR's own
  // TilingInterface and bufferization infrastructure already handle it
  // as a `LinalgOp`. Nothing project-specific is needed beyond the
  // permutation semantic analysis already verified.
  MlirValue lowerTranspose(MlirValue x, const std::vector<uint32_t>& perm, OpBuilder& b, Location loc,
                           RankedTensorType mlirOutType) {
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    SmallVector<int64_t> permI64(perm.begin(), perm.end());
    auto transpose = linalg::TransposeOp::create(b, loc, x, init, permI64);
    return transpose->getResult(0);
  }

  // RM 3.16: conv2d lowers to `linalg.conv_2d_nchw_fchw`. The input,
  // weight, and output layouts match the contract's NCHW/FCHW layout
  // exactly (RM 3.16). This function zero-fills the accumulator
  // (linalg's conv op accumulates into `outs`, the same as matmul), and
  // adds an explicit `tensor.pad` when padding > 0, since the linalg
  // conv op itself has no padding attribute. The bias-add generic op
  // broadcasts over the channel axis (index 1), not the last axis:
  // unlike Linear's bias, which applies per output position, conv2d's
  // bias applies per output channel.
  MlirValue lowerConv2D(const GraphNode& node, MlirValue x, MlirValue w, MlirValue bias, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType) {
    int64_t padding = node.attrs.padding;
    int64_t stride = node.attrs.stride;

    MlirValue input = x;
    if (padding > 0) {
      auto xType = cast<RankedTensorType>(x.getType());
      SmallVector<int64_t> paddedShape(xType.getShape().begin(), xType.getShape().end());
      paddedShape[2] += 2 * padding;
      paddedShape[3] += 2 * padding;
      RankedTensorType paddedType = RankedTensorType::get(paddedShape, xType.getElementType());
      MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
      auto lowVals = getAsIndexOpFoldResult(b.getContext(), SmallVector<int64_t>{0, 0, padding, padding});
      auto highVals = getAsIndexOpFoldResult(b.getContext(), SmallVector<int64_t>{0, 0, padding, padding});
      input = tensor::PadOp::create(b, loc, paddedType, x, lowVals, highVals, zero);
    }

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    DenseIntElementsAttr strides = b.getI64TensorAttr({stride, stride});
    DenseIntElementsAttr dilations = b.getI64TensorAttr({1, 1});
    auto conv = linalg::Conv2DNchwFchwOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{input, w}, ValueRange{acc},
                                                  strides, dilations);
    MlirValue convResult = conv.getResult(0);

    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    AffineMap biasMap = AffineMap::get(outRank, 0, {getAffineDimExpr(1, b.getContext())}, b.getContext());
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{convResult, bias}, ValueRange{init},
        ArrayRef<AffineMap>{outMap, biasMap, outMap}, parallelIterators(outRank),
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue result = arith::AddFOp::create(body, l, args[0], args[1]);
          linalg::YieldOp::create(body, l, ValueRange{result});
        });
    return generic.getResult(0);
  }

  // RM 3.15: sum and mean remove the named axis entirely (rank - 1),
  // unlike softmax's keep-dims reduction. Mean runs sum, then a
  // divide-by-axis-size elementwise pass. Sum's accumulator identity
  // is 0, the same zero-fill approach as matmul's and conv2d's
  // accumulators.
  MlirValue lowerReduce(const TypedGraph& graph, const GraphNode& node, MlirValue x, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType) {
    const auto* xType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
    size_t axisIdx = findAxisIndex(xType, *node.attrs.axisName);
    size_t inRank = xType->rank();

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    AffineMap inMap = AffineMap::getMultiDimIdentityMap(inRank, b.getContext());
    AffineMap outMap = dropAxisMap(b.getContext(), inRank, axisIdx);
    SmallVector<utils::IteratorType> iters(inRank, utils::IteratorType::parallel);
    iters[axisIdx] = utils::IteratorType::reduction;
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{x}, ValueRange{acc}, ArrayRef<AffineMap>{inMap, outMap}, iters,
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue sum = arith::AddFOp::create(body, l, args[0], args[1]);
          linalg::YieldOp::create(body, l, ValueRange{sum});
        });
    MlirValue result = generic.getResult(0);
    if (node.op != OpKind::Mean) return result;

    int64_t axisSize = xType->axes()[axisIdx].size.constant;
    MlirValue count =
        arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(static_cast<float>(axisSize)));
    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap idMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    auto divGeneric = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{result}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
        parallelIterators(outRank), [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue divided = arith::DivFOp::create(body, l, args[0], count);
          linalg::YieldOp::create(body, l, ValueRange{divided});
        });
    return divGeneric.getResult(0);
  }

  // RM 3.17: maxpool lowers to `linalg.pooling_nchw_max`. Its second
  // "input" is a shape-only operand: the op never reads its values,
  // only its spatial extent, which gives the pooling window shape. The
  // op accumulates a running max into `outs`, so this needs the same
  // -inf-fill-then-max pattern softmax's own reduce-max uses. A
  // zero-filled accumulator would be wrong here, unlike for sum,
  // conv2d, or matmul.
  MlirValue lowerMaxPool(const GraphNode& node, MlirValue x, OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    int64_t size = node.attrs.poolSize;
    int64_t stride = node.attrs.stride;

    MlirValue negInf =
        arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    MlirValue empty = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    MlirValue acc = linalg::FillOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{negInf}, ValueRange{empty})
                        .getResult(0);
    RankedTensorType windowType = RankedTensorType::get({size, size}, b.getF32Type());
    MlirValue window = tensor::EmptyOp::create(b, loc, windowType, ValueRange{});
    DenseIntElementsAttr strides = b.getI64TensorAttr({stride, stride});
    DenseIntElementsAttr dilations = b.getI64TensorAttr({1, 1});
    auto pool = linalg::PoolingNchwMaxOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{x, window},
                                                  ValueRange{acc}, strides, dilations);
    return pool.getResult(0);
  }

  MLIRContext& ctx_;
  TilingDecision tilingDecision_;
  LivenessResult currentLiveness_;  // for the graph currently being lowered; set at the top of lowerGraph
};

}  // namespace

ModuleOp MlirLowerer::lower(const TypedProgram& program) {
  return LowererImpl(ctx_, tilingDecision_).lower(program);
}

}  // namespace tensora
