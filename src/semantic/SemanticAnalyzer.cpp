// This file implements semantic analysis.
// It checks the AST for name, scope, type, and shape errors, and
// creates the verified Typed Computation Graph (TCG). Optimization
// reads the TCG next.
#include "tensora/semantic/SemanticAnalyzer.h"

#include <initializer_list>
#include <string_view>
#include <unordered_set>

#include "tensora/semantic/Contracts.h"

namespace tensora {

namespace {

void reportSimple(DiagnosticEngine& diags, DiagCode code, SourceSpan span, std::string msg) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = code;
  d.span = span;
  d.message = std::move(msg);
  diags.report(d);
}

// Checks the two tensor-type rules RM 3.3 requires of any axis list:
// every concrete axis size must be positive, and axis names must be
// unique within the list. This reports one diagnostic per violation,
// and returns false so the caller can poison the result. This never
// checks a symbolic axis size for a positive value here, since its
// concrete value, if it has one, is not known until DimSolver
// resolves it later.
bool validateAxes(const std::vector<Axis>& axes, DiagnosticEngine& diags, SourceSpan span) {
  bool ok = true;
  std::unordered_set<std::string> seen;
  for (const Axis& ax : axes) {
    if (ax.size.kind == Dimension::Kind::Const && ax.size.constant <= 0) {
      reportSimple(diags, DiagCode::SemaNonPositiveDimension, span,
                   "axis '" + ax.name + "' has size " + std::to_string(ax.size.constant) +
                       "; RM 3.3 requires a concrete positive integer");
      ok = false;
    }
    if (!seen.insert(ax.name).second) {
      reportSimple(diags, DiagCode::SemaDuplicateAxisName, span,
                   "duplicate axis name '" + ax.name + "' within one tensor type");
      ok = false;
    }
  }
  return ok;
}

const Type* convertTensorType(const ast::TensorTypeSyntax& syn, TypeContext& types, DiagnosticEngine& diags) {
  std::vector<Axis> axes;
  axes.reserve(syn.axes.size());
  for (const ast::AxisSyntax& a : syn.axes) {
    Dimension d = a.size.k == ast::DimSyntax::K::Const ? Dimension::makeConst(a.size.constant)
                                                        : Dimension::makeSymbol(types.internSymbol(a.size.symbol));
    axes.push_back(Axis{a.name, d});
  }
  if (!validateAxes(axes, diags, syn.span)) return types.invalid();
  return types.tensor(std::move(axes));
}

std::vector<Axis> convertAxisList(const std::vector<ast::AxisSyntax>& syn, TypeContext& types) {
  std::vector<Axis> axes;
  axes.reserve(syn.size());
  for (const ast::AxisSyntax& a : syn) {
    Dimension d = a.size.k == ast::DimSyntax::K::Const ? Dimension::makeConst(a.size.constant)
                                                        : Dimension::makeSymbol(types.internSymbol(a.size.symbol));
    axes.push_back(Axis{a.name, d});
  }
  return axes;
}

// Checks that `args`'s first `positionalCount` entries are all
// unkeyworded Kind::Name (plain identifiers), and that no more
// positional entries follow. Every call-style operator uses this,
// except reshape, whose second argument is a positional axis list,
// not a name.
bool takePositionalNames(const std::vector<ast::Arg>& args, size_t positionalCount, std::string_view opName,
                         DiagnosticEngine& diags, SourceSpan span, std::vector<const ast::Arg*>& positional) {
  size_t have = 0;
  while (have < args.size() && !args[have].keyword.has_value()) ++have;
  if (have != positionalCount) {
    reportSimple(diags, DiagCode::SemaArgCount, span,
                 std::string(opName) + " expects " + std::to_string(positionalCount) +
                     " positional argument(s); found " + std::to_string(have));
    return false;
  }
  for (size_t i = 0; i < have; ++i) {
    if (args[i].kind != ast::Arg::Kind::Name) {
      reportSimple(diags, DiagCode::SemaArgKind, span,
                   std::string(opName) + "'s positional argument " + std::to_string(i) +
                       " must be a plain identifier");
      return false;
    }
    positional.push_back(&args[i]);
  }
  return true;
}

// Finds a keyword argument by name; order does not matter. Reports
// SemaArgCount if `required` is true and the argument is absent, or
// SemaArgKind if it is present with the wrong Arg::Kind. If the
// caller supplies `found`, this function sets it either way, correct
// kind or not. This lets an optional keyword's caller tell "absent"
// apart from "present but invalid" (the invalid case is already
// reported).
const ast::Arg* findKeyword(const std::vector<ast::Arg>& args, std::string_view key, ast::Arg::Kind expectedKind,
                            bool required, std::string_view opName, DiagnosticEngine& diags, SourceSpan span,
                            bool* found = nullptr) {
  for (const ast::Arg& a : args) {
    if (a.keyword.has_value() && *a.keyword == key) {
      if (found) *found = true;
      if (a.kind != expectedKind) {
        reportSimple(diags, DiagCode::SemaArgKind, span,
                     std::string(opName) + "'s '" + std::string(key) + "' argument has the wrong kind");
        return nullptr;
      }
      return &a;
    }
  }
  if (found) *found = false;
  if (required) {
    reportSimple(diags, DiagCode::SemaArgCount, span,
                 std::string(opName) + " requires a '" + std::string(key) + "' keyword argument");
  }
  return nullptr;
}

bool rejectUnknownKeywords(const std::vector<ast::Arg>& args, std::initializer_list<std::string_view> allowed,
                           std::string_view opName, DiagnosticEngine& diags, SourceSpan span) {
  for (const ast::Arg& a : args) {
    if (!a.keyword.has_value()) continue;
    bool ok = false;
    for (std::string_view k : allowed) {
      if (*a.keyword == k) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      reportSimple(diags, DiagCode::SemaArgKind, span,
                   std::string(opName) + ": unknown keyword argument '" + *a.keyword + "'");
      return false;
    }
  }
  return true;
}

}  // namespace

std::pair<ValueId, const Type*> SemanticAnalyzer::resolveOperand(GraphCtx& gc, const std::string& name,
                                                                  SourceSpan span) {
  const Symbol* sym = symbols_.resolve(name);
  if (gc.loggedNames.insert(name).second) {
    std::string resolvedAs = !sym                                  ? "undefined"
                              : sym->kind == Symbol::Kind::Value ? "local::" + name
                                                                  : "global::" + name;
    gc.resolutionLog.emplace_back(name, std::move(resolvedAs));
  }
  if (!sym) {
    reportSimple(diags_, DiagCode::SemaUndefinedName, span, "use of undefined name '" + name + "'");
    return {0, types_.invalid()};
  }
  if (sym->kind == Symbol::Kind::Value) {
    auto it = gc.localValues.find(name);
    return {it != gc.localValues.end() ? it->second : 0, sym->type};
  }
  auto git = gc.globalValues.find(name);
  if (git != gc.globalValues.end()) return {git->second, sym->type};
  if (sym->type->kind() == Type::Kind::Invalid) {
    // The declaration itself failed RM 3.3 validation. That error was
    // already reported at the declaration site, so this use is
    // poisoned, not a new error.
    return {0, sym->type};
  }
  const auto* tt = static_cast<const TensorType*>(sym->type);  // globals are always Tensor once past the invalid check
  ValueId vid = sym->kind == Symbol::Kind::Input ? gc.builder.addInput(name, tt, sym->span)
                                                  : gc.builder.addWeight(name, tt, sym->span);
  gc.globalValues.emplace(name, vid);
  return {vid, sym->type};
}

std::pair<ValueId, const Type*> SemanticAnalyzer::resolveExprOperand(GraphCtx& gc, const ast::Expr& e) {
  if (const auto* name = dynamic_cast<const ast::NameExpr*>(&e)) {
    return resolveOperand(gc, name->name, name->span);
  }
  if (dynamic_cast<const ast::ScalarExpr*>(&e)) {
    return {0, types_.scalar()};
  }
  return {0, types_.invalid()};  // unreachable given RM 3.24's `operand ::= identifier | floatLit`
}

std::pair<ValueId, const Type*> SemanticAnalyzer::analyzeMatMul(GraphCtx& gc, const ast::MatMulExpr& mm) {
  const auto* lhsName = static_cast<const ast::NameExpr*>(mm.lhs.get());  // grammar: lhs is always a NameExpr
  auto [lhsId, lhsType] = resolveOperand(gc, lhsName->name, lhsName->span);
  auto [rhsId, rhsType] = resolveExprOperand(gc, *mm.rhs);

  ContractCtx ctx{types_, gc.solver, diags_, mm.span};
  const Type* result = checkMatMul(lhsType, rhsType, ctx);
  if (result->kind() == Type::Kind::Invalid) return {0, result};

  ValueId out =
      gc.builder.emit(OpKind::MatMul, {lhsId, rhsId}, {}, static_cast<const TensorType*>(result), mm.span, "");
  return {out, result};
}

std::pair<ValueId, const Type*> SemanticAnalyzer::analyzeBinary(GraphCtx& gc, const ast::BinaryExpr& bin) {
  const auto* lhsName = static_cast<const ast::NameExpr*>(bin.lhs.get());
  auto [lhsId, lhsType] = resolveOperand(gc, lhsName->name, lhsName->span);
  auto [rhsId, rhsType] = resolveExprOperand(gc, *bin.rhs);

  ContractCtx ctx{types_, gc.solver, diags_, bin.span};
  const Type* result = checkElementwise(bin.op, lhsType, rhsType, ctx);
  if (result->kind() == Type::Kind::Invalid) return {0, result};

  NodeAttrs attrs;
  std::vector<ValueId> inputs = {lhsId};
  if (rhsType->kind() == Type::Kind::Scalar) {
    attrs.scalar = static_cast<const ast::ScalarExpr*>(bin.rhs.get())->value;
  } else {
    inputs.push_back(rhsId);
  }
  OpKind op = bin.op == ast::BinaryExpr::Op::Add   ? OpKind::Add
              : bin.op == ast::BinaryExpr::Op::Sub ? OpKind::Sub
              : bin.op == ast::BinaryExpr::Op::Mul ? OpKind::Mul
                                                    : OpKind::Div;
  ValueId out = gc.builder.emit(op, std::move(inputs), attrs, static_cast<const TensorType*>(result), bin.span, "");
  return {out, result};
}

std::pair<ValueId, const Type*> SemanticAnalyzer::analyzeCall(GraphCtx& gc, const ast::CallExpr& call) {
  const std::string& op = call.callee;
  const SourceSpan span = call.span;
  const std::vector<ast::Arg>& args = call.args;
  auto poison = [&]() -> std::pair<ValueId, const Type*> { return {0, types_.invalid()}; };

  if (op == "linear") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 3, "linear", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {}, "linear", diags_, span)) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    auto [wId, wType] = resolveOperand(gc, pos[1]->name, pos[1]->span);
    auto [bId, bType] = resolveOperand(gc, pos[2]->name, pos[2]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkLinear(xType, wType, bType, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    ValueId out =
        gc.builder.emit(OpKind::Linear, {xId, wId, bId}, {}, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "relu") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "relu", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {}, "relu", diags_, span)) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkRelu(xType, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    ValueId out = gc.builder.emit(OpKind::Relu, {xId}, {}, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "softmax") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "softmax", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"axis"}, "softmax", diags_, span)) return poison();
    const ast::Arg* axisArg =
        findKeyword(args, "axis", ast::Arg::Kind::Name, true, "softmax", diags_, span);
    if (!axisArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkSoftmax(xType, axisArg->name, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.axisName = axisArg->name;
    ValueId out = gc.builder.emit(OpKind::Softmax, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "reshape") {
    if (args.size() != 2) {
      reportSimple(diags_, DiagCode::SemaArgCount, span,
                   "reshape expects exactly 2 arguments (identifier, axis-list); found " +
                       std::to_string(args.size()));
      return poison();
    }
    if (args[0].keyword.has_value() || args[0].kind != ast::Arg::Kind::Name) {
      reportSimple(diags_, DiagCode::SemaArgKind, span, "reshape's first argument must be a plain identifier");
      return poison();
    }
    if (args[1].keyword.has_value() || args[1].kind != ast::Arg::Kind::AxisList) {
      reportSimple(diags_, DiagCode::SemaArgKind, span,
                   "reshape's second argument must be an axis list [name=size, ...]");
      return poison();
    }
    auto [xId, xType] = resolveOperand(gc, args[0].name, args[0].span);
    std::vector<Axis> target = convertAxisList(args[1].axisList, types_);
    if (!validateAxes(target, diags_, span)) return poison();
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkReshape(xType, target, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.reshapeTarget = target;
    ValueId out = gc.builder.emit(OpKind::Reshape, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "flatten") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "flatten", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {}, "flatten", diags_, span)) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkFlatten(xType, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    ValueId out = gc.builder.emit(OpKind::Flatten, {xId}, {}, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "transpose") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "transpose", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"perm"}, "transpose", diags_, span)) return poison();
    const ast::Arg* permArg =
        findKeyword(args, "perm", ast::Arg::Kind::IntList, true, "transpose", diags_, span);
    if (!permArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    std::vector<uint32_t> perm;
    perm.reserve(permArg->intList.size());
    for (int64_t v : permArg->intList) perm.push_back(static_cast<uint32_t>(v));
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkTranspose(xType, perm, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.perm = perm;
    ValueId out = gc.builder.emit(OpKind::Transpose, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "sum" || op == "mean") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, op, diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"axis"}, op, diags_, span)) return poison();
    const ast::Arg* axisArg = findKeyword(args, "axis", ast::Arg::Kind::Name, true, op, diags_, span);
    if (!axisArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    OpKind kind = op == "sum" ? OpKind::Sum : OpKind::Mean;
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkReduce(kind, xType, axisArg->name, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.axisName = axisArg->name;
    ValueId out = gc.builder.emit(kind, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "conv2d") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 3, "conv2d", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"stride", "padding"}, "conv2d", diags_, span)) return poison();
    const ast::Arg* strideArg =
        findKeyword(args, "stride", ast::Arg::Kind::Int, true, "conv2d", diags_, span);
    const ast::Arg* paddingArg =
        findKeyword(args, "padding", ast::Arg::Kind::Int, true, "conv2d", diags_, span);
    if (!strideArg || !paddingArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    auto [wId, wType] = resolveOperand(gc, pos[1]->name, pos[1]->span);
    auto [bId, bType] = resolveOperand(gc, pos[2]->name, pos[2]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkConv2D(xType, wType, bType, strideArg->intVal, paddingArg->intVal, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.stride = strideArg->intVal;
    attrs.padding = paddingArg->intVal;
    ValueId out =
        gc.builder.emit(OpKind::Conv2D, {xId, wId, bId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "maxpool") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "maxpool", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"size", "stride"}, "maxpool", diags_, span)) return poison();
    const ast::Arg* sizeArg = findKeyword(args, "size", ast::Arg::Kind::Int, true, "maxpool", diags_, span);
    if (!sizeArg) return poison();
    bool strideFound = false;
    const ast::Arg* strideArg =
        findKeyword(args, "stride", ast::Arg::Kind::Int, false, "maxpool", diags_, span, &strideFound);
    if (strideFound && !strideArg) return poison();  // present but wrong kind (already reported)
    int64_t stride = strideArg ? strideArg->intVal : sizeArg->intVal;  // RM 3.17: defaults to size
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkMaxPool(xType, sizeArg->intVal, stride, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.poolSize = sizeArg->intVal;
    attrs.stride = stride;
    ValueId out = gc.builder.emit(OpKind::MaxPool, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }

  reportSimple(diags_, DiagCode::SemaUndefinedName, span, "undefined operator '" + op + "'");
  return poison();
}

TypedGraph SemanticAnalyzer::analyzeGraph(const ast::GraphDecl& g) {
  symbols_.enterGraph();
  GraphCtx gc{GraphBuilder(g.name), DimSolver(), {}, {}, {}, {}};
  const size_t diagsBeforeGraph = diags_.all().size();

  for (const ast::Stmt& stmt : g.stmts) {
    ValueId resultValue = 0;
    const Type* resultType = types_.invalid();

    if (const auto* call = dynamic_cast<const ast::CallExpr*>(stmt.rhs.get())) {
      auto r = analyzeCall(gc, *call);
      resultValue = r.first;
      resultType = r.second;
    } else if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(stmt.rhs.get())) {
      auto r = analyzeMatMul(gc, *mm);
      resultValue = r.first;
      resultType = r.second;
    } else if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(stmt.rhs.get())) {
      auto r = analyzeBinary(gc, *bin);
      resultValue = r.first;
      resultType = r.second;
    } else if (const auto* name = dynamic_cast<const ast::NameExpr*>(stmt.rhs.get())) {
      auto r = resolveOperand(gc, name->name, name->span);
      resultValue = r.first;
      resultType = r.second;
    }

    if (resultType->kind() != Type::Kind::Invalid) gc.builder.setDebugName(resultValue, stmt.name);

    Symbol sym;
    sym.name = stmt.name;
    sym.kind = Symbol::Kind::Value;
    sym.type = resultType;
    sym.span = stmt.span;
    SymbolTable::BindOutcome outcome = symbols_.bindLocal(stmt.name, sym);
    if (outcome == SymbolTable::BindOutcome::KindConflict) {
      reportSimple(diags_, DiagCode::SemaRedeclaration, stmt.span,
                   "'" + stmt.name + "' is already declared as a global input/weight; a graph cannot rebind it");
    } else {
      gc.localValues[stmt.name] = resultValue;
    }
  }

  auto [retId, retType] = resolveOperand(gc, g.returnName, g.span);
  gc.builder.setOutput(retType->kind() != Type::Kind::Invalid ? retId : 0);

  SymbolTableReport::GraphScope scope;
  scope.graphName = g.name;
  for (const Symbol* s : symbols_.orderedLocals()) scope.locals.push_back(*s);
  scope.resolutions = std::move(gc.resolutionLog);
  symbolReport_.graphs.push_back(std::move(scope));

  TypedGraph graph = gc.builder.finish();
  // GraphVerifier catches bugs from a broken pass or a bad hand-built
  // fixture (INTERFACES.md §3a.6), on a graph that is otherwise clean. It should
  // not add a second, redundant diagnostic to a graph semantic
  // analysis already knows is broken (for example, `return` of an
  // undefined name leaves an empty graph with no output, which is
  // *expected*, since the error was already reported).
  if (diags_.all().size() == diagsBeforeGraph) {
    GraphVerifier verifier;
    for (Diagnostic& d : verifier.verify(graph)) diags_.report(d);
  }
  return graph;
}

Result<TypedProgram> SemanticAnalyzer::analyze(const ast::AstProgram& program) {
  TypedProgram output(types_);

  for (const ast::Decl& decl : program.decls) {
    const Type* type = convertTensorType(decl.type, types_, diags_);
    Symbol sym;
    sym.name = decl.name;
    sym.kind = decl.kind == ast::Decl::Kind::Input ? Symbol::Kind::Input : Symbol::Kind::Weight;
    sym.type = type;
    sym.span = decl.span;
    if (!symbols_.declareGlobal(sym)) {
      reportSimple(diags_, DiagCode::SemaRedeclaration, decl.span,
                   "redeclaration of top-level name '" + decl.name + "'");
    }
  }

  for (const Symbol* s : symbols_.orderedGlobals()) symbolReport_.globals.push_back(*s);

  std::unordered_set<std::string> graphNames;
  for (const ast::GraphDecl& g : program.graphs) {
    if (!graphNames.insert(g.name).second) {
      reportSimple(diags_, DiagCode::SemaRedeclaration, g.span, "duplicate graph name '" + g.name + "'");
      continue;
    }
    output.addGraph(analyzeGraph(g));
  }

  bool ok = !diags_.hasErrors();
  return Result<TypedProgram>::make(std::move(output), ok, diags_.all());
}

}  // namespace tensora
