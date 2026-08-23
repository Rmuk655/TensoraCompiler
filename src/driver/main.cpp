// The `tensora` driver: one binary, one flag per compiler phase. This
// lets each phase run on its own: --emit-tokens for the lexer,
// --emit-ast for the parser, --check/--emit-graph for semantic
// analysis, --emit-optimized-graph for optimization, --emit-mlir for
// lowering, --emit-llvm for backend translation. Two more flags run
// every phase together as a real compiler: --run JIT-executes the
// program, and -o emits a native object file. See
// docs/language/02_LANGUAGE_TUTORIAL.md Section 2.12 for the full CLI
// surface.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tensora/lexer/Lexer.h"
#include "tensora/parser/Parser.h"
#include "tensora/optimization/GraphPass.h"
#include "tensora/semantic/SemanticAnalyzer.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MatmulTilingPass.h"
#include "tensora/codegen/MlirLowerer.h"

#include "Npy.h"

using namespace tensora;

namespace {

// MlirLowerer computes the Level-2 TilingDecision and stamps its plan
// onto each linalg.matmul. MatmulTilingPass only applies that stamped
// plan. Both share this one TilingPolicy object, so they provably use
// the same policy, instead of two separately built structs that only
// happen to default to the same values.
const TilingPolicy kTilingPolicy{};

std::string readFile(const std::string& path, bool& ok) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    ok = false;
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  ok = true;
  return ss.str();
}

std::string exprToString(const ast::Expr& e);

std::string tensorTypeToString(const ast::TensorTypeSyntax& t) {
  std::string s = "Tensor[";
  for (size_t i = 0; i < t.axes.size(); ++i) {
    const ast::AxisSyntax& ax = t.axes[i];
    s += (i ? ", " : "") + ax.name + "=" +
         (ax.size.k == ast::DimSyntax::K::Const ? std::to_string(ax.size.constant) : ax.size.symbol);
  }
  return s + "]";
}

std::string argToString(const ast::Arg& a) {
  std::string prefix = a.keyword ? (*a.keyword + "=") : "";
  switch (a.kind) {
    case ast::Arg::Kind::Name: return prefix + a.name;
    case ast::Arg::Kind::Int: return prefix + std::to_string(a.intVal);
    case ast::Arg::Kind::Float: return prefix + std::to_string(a.floatVal);
    case ast::Arg::Kind::IntList: {
      std::string s = prefix + "[";
      for (size_t i = 0; i < a.intList.size(); ++i) s += (i ? "," : "") + std::to_string(a.intList[i]);
      return s + "]";
    }
    case ast::Arg::Kind::AxisList: {
      std::string s = prefix + "[";
      for (size_t i = 0; i < a.axisList.size(); ++i) {
        const ast::AxisSyntax& ax = a.axisList[i];
        s += (i ? "," : "") + ax.name + "=" +
             (ax.size.k == ast::DimSyntax::K::Const ? std::to_string(ax.size.constant) : ax.size.symbol);
      }
      return s + "]";
    }
  }
  return prefix;
}

std::string exprToString(const ast::Expr& e) {
  if (const auto* n = dynamic_cast<const ast::NameExpr*>(&e)) return n->name;
  if (const auto* s = dynamic_cast<const ast::ScalarExpr*>(&e)) return std::to_string(s->value);
  if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(&e)) {
    return exprToString(*mm->lhs) + " @ " + exprToString(*mm->rhs);
  }
  if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(&e)) {
    static const char* ops[] = {"+", "-", "*", "/"};
    return exprToString(*bin->lhs) + " " + ops[static_cast<int>(bin->op)] + " " + exprToString(*bin->rhs);
  }
  if (const auto* call = dynamic_cast<const ast::CallExpr*>(&e)) {
    std::string s = call->callee + "(";
    for (size_t i = 0; i < call->args.size(); ++i) s += (i ? ", " : "") + argToString(call->args[i]);
    return s + ")";
  }
  return "<?>";
}

// --emit-ast: prints the real AST node tree, using box-drawing
// characters. This is not the source text printed again. It shows the
// tree structure the parser actually built, not just a reformatted
// copy of the input program.
void printTreeLine(const std::string& prefix, bool isLast, const std::string& label) {
  std::cout << prefix << (isLast ? "└── " : "├── ") << label << "\n";
}

std::string childPrefix(const std::string& prefix, bool isLast) { return prefix + (isLast ? "    " : "│   "); }

void printExprTree(const ast::Expr& e, const std::string& prefix, bool isLast) {
  if (const auto* n = dynamic_cast<const ast::NameExpr*>(&e)) {
    printTreeLine(prefix, isLast, "Identifier(" + n->name + ")");
    return;
  }
  if (const auto* s = dynamic_cast<const ast::ScalarExpr*>(&e)) {
    printTreeLine(prefix, isLast, "Scalar(" + std::to_string(s->value) + ")");
    return;
  }
  if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(&e)) {
    printTreeLine(prefix, isLast, "MatMul");
    std::string cp = childPrefix(prefix, isLast);
    printExprTree(*mm->lhs, cp, false);
    printExprTree(*mm->rhs, cp, true);
    return;
  }
  if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(&e)) {
    static const char* ops[] = {"Add", "Sub", "Mul", "Div"};
    printTreeLine(prefix, isLast, std::string("BinaryExpr(") + ops[static_cast<int>(bin->op)] + ")");
    std::string cp = childPrefix(prefix, isLast);
    printExprTree(*bin->lhs, cp, false);
    printExprTree(*bin->rhs, cp, true);
    return;
  }
  if (const auto* call = dynamic_cast<const ast::CallExpr*>(&e)) {
    printTreeLine(prefix, isLast, "CallExpr(" + call->callee + ")");
    std::string cp = childPrefix(prefix, isLast);
    for (size_t i = 0; i < call->args.size(); ++i) {
      printTreeLine(cp, i + 1 == call->args.size(), "Arg(" + argToString(call->args[i]) + ")");
    }
    return;
  }
  printTreeLine(prefix, isLast, "?");
}

void printAstTree(const ast::AstProgram& program) {
  std::cout << "Program\n";
  size_t total = program.decls.size() + program.graphs.size();
  size_t i = 0;
  for (const ast::Decl& d : program.decls) {
    bool isLast = (++i == total);
    const char* kind = d.kind == ast::Decl::Kind::Input ? "InputDecl" : "WeightDecl";
    printTreeLine("", isLast, std::string(kind) + "(" + d.name + " : " + tensorTypeToString(d.type) + ")");
  }
  for (const ast::GraphDecl& g : program.graphs) {
    bool isLast = (++i == total);
    printTreeLine("", isLast, "GraphDecl(" + g.name + ")");
    std::string cp = childPrefix("", isLast);
    size_t stmtTotal = g.stmts.size() + 1;  // + 1 for the return statement
    size_t j = 0;
    for (const ast::Stmt& s : g.stmts) {
      bool stmtIsLast = (++j == stmtTotal);
      printTreeLine(cp, stmtIsLast, "Assignment(" + s.name + ")");
      printExprTree(*s.rhs, childPrefix(cp, stmtIsLast), true);
    }
    printTreeLine(cp, true, "Return(" + g.returnName + ")");
  }
}

// --emit-graph / --explain: prints one line per node, with the
// operator, its operands (by debug name), and the *verified* result
// type. This is the same information `describeType` renders in
// diagnostics, so it reuses the exact type-printing the rest of the
// compiler already uses, instead of inventing a second format.
void printTypedGraph(const TypedGraph& g, TypeContext& types, bool explainMode) {
  std::cout << "graph " << g.name << " {\n";
  for (ValueId p : g.params) {
    const Value& v = g.value(p);
    std::cout << "  param " << v.debugName << " : " << describeType(v.type, types) << "\n";
  }
  for (const GraphNode& node : g.nodes()) {
    const Value& out = g.value(node.output);
    std::cout << "  " << out.debugName << " = " << opKindName(node.op) << "(";
    for (size_t i = 0; i < node.inputs.size(); ++i) {
      std::cout << (i ? ", " : "") << g.value(node.inputs[i]).debugName;
    }
    std::cout << ") : " << describeType(out.type, types);
    if (explainMode) {
      // Names the operator contract that produced this node (whitepaper
      // Section 5), by which RM section governs it. That RM section is
      // the contract's identity: there is one contract function per
      // OpKind.
      std::cout << "   # RM " << opKindName(node.op);
    }
    std::cout << "\n";
  }
  std::cout << "  return " << g.value(g.output).debugName << "\n}\n";
}

const char* symbolKindName(Symbol::Kind k) {
  switch (k) {
    case Symbol::Kind::Input: return "input";
    case Symbol::Kind::Weight: return "weight";
    case Symbol::Kind::Value: return "local";
  }
  return "?";
}

// --emit-symbols: renders SemanticAnalyzer::symbolTable() (RM 3.6).
// This prints exactly the declaration-order entries SymbolTable
// actually accepted during name resolution (see Symbol.h's
// orderedGlobals/orderedLocals). It is not a second, separately built
// symbol table. There is one GLOBAL section (input/weight
// declarations, shared across every graph), plus one GRAPH section
// per graph (that graph's final local scope). This matches the two
// scope levels RM 3.6 defines. A trailing RESOLVED section per graph
// shows what each name reference in that graph's body actually
// resolved to: the real resolution log SemanticAnalyzer::resolveOperand
// produced during name resolution, not a replay of it.
void printSymbolTable(const SymbolTableReport& report, TypeContext& types) {
  std::cout << "=== Symbol Table ===\n\nGLOBAL\n";
  for (const Symbol& s : report.globals) {
    std::cout << "  " << s.name << "  " << symbolKindName(s.kind) << "  " << describeType(s.type, types) << "  ("
               << s.span.line << ":" << s.span.col << ")\n";
  }
  for (const SymbolTableReport::GraphScope& scope : report.graphs) {
    std::cout << "\nGRAPH " << scope.graphName << "\n";
    for (const Symbol& s : scope.locals) {
      std::cout << "  " << s.name << "  " << symbolKindName(s.kind) << "  " << describeType(s.type, types) << "  ("
                 << s.span.line << ":" << s.span.col << ")\n";
    }
    if (!scope.resolutions.empty()) {
      std::cout << "  RESOLVED\n";
      for (const auto& [name, resolvedAs] : scope.resolutions) {
        std::cout << "    " << name << " -> " << resolvedAs << "\n";
      }
    }
  }
  std::cout << "\n=== End Symbol Table ===\n";
}

// --emit-ast-dot: a Graphviz DOT dump of the AST, for demos and for
// visually checking that the parser built the tree a program expects
// (e.g. `tensora --emit-ast-dot f.tns | dot -Tpng -o ast.png`).
class AstDotWriter {
 public:
  std::string build(const ast::AstProgram& program) {
    for (const ast::Decl& d : program.decls) {
      node((d.kind == ast::Decl::Kind::Input ? "input " : "weight ") + d.name);
    }
    for (const ast::GraphDecl& g : program.graphs) {
      std::string graphId = node("graph " + g.name);
      for (const ast::Stmt& s : g.stmts) {
        std::string assignId = node("Assignment " + s.name);
        edge(graphId, assignId);
        edge(assignId, emitExpr(*s.rhs));
      }
      edge(graphId, node("return " + g.returnName));
    }
    return "digraph AST {\n" + out_.str() + "}\n";
  }

 private:
  std::string node(const std::string& label) {
    std::string id = "n" + std::to_string(counter_++);
    out_ << "  " << id << " [label=\"" << label << "\"];\n";
    return id;
  }
  void edge(const std::string& from, const std::string& to) { out_ << "  " << from << " -> " << to << ";\n"; }

  std::string emitExpr(const ast::Expr& e) {
    if (const auto* n = dynamic_cast<const ast::NameExpr*>(&e)) return node("Identifier " + n->name);
    if (const auto* s = dynamic_cast<const ast::ScalarExpr*>(&e)) return node("Scalar " + std::to_string(s->value));
    if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(&e)) {
      std::string id = node("MatMul");
      edge(id, emitExpr(*mm->lhs));
      edge(id, emitExpr(*mm->rhs));
      return id;
    }
    if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(&e)) {
      static const char* ops[] = {"Add", "Sub", "Mul", "Div"};
      std::string id = node(ops[static_cast<int>(bin->op)]);
      edge(id, emitExpr(*bin->lhs));
      edge(id, emitExpr(*bin->rhs));
      return id;
    }
    if (const auto* call = dynamic_cast<const ast::CallExpr*>(&e)) {
      std::string id = node("Call " + call->callee);
      for (const ast::Arg& a : call->args) edge(id, node(argToString(a)));
      return id;
    }
    return node("?");
  }

  std::ostringstream out_;
  int counter_ = 0;
};

std::string astToDot(const ast::AstProgram& program) { return AstDotWriter().build(program); }

// --emit-graph-dot: a Graphviz DOT dump of the Typed Computation Graph,
// one node per param/graph-node, edges following producer -> consumer.
std::string graphToDot(const TypedGraph& g, TypeContext& types) {
  std::ostringstream out;
  out << "digraph TypedGraph_" << g.name << " {\n";
  for (ValueId p : g.params) {
    const Value& v = g.value(p);
    out << "  v" << p << " [shape=box, style=dashed, label=\"" << v.debugName << "\\n"
        << describeType(v.type, types) << "\"];\n";
  }
  for (const GraphNode& node : g.nodes()) {
    const Value& out_v = g.value(node.output);
    out << "  n" << node.id << " [label=\"" << opKindName(node.op) << "\\n" << describeType(out_v.type, types)
        << "\"];\n";
    for (ValueId in : node.inputs) {
      NodeId producer = g.producerOf(in);
      std::string fromId = (producer == kNoProducer) ? ("v" + std::to_string(in)) : ("n" + std::to_string(producer));
      out << "  " << fromId << " -> n" << node.id << ";\n";
    }
  }
  NodeId outputProducer = g.producerOf(g.output);
  out << "  n" << outputProducer << " -> ret [label=\"return\"];\n";
  out << "  ret [shape=doublecircle, label=\"" << g.value(g.output).debugName << "\"];\n";
  out << "}\n";
  return out.str();
}

// Returns false, with a message on stderr, if any param or output has
// a symbolic dimension. --run and -o need concrete sizes, to build
// input data or to emit a fixed-shape function.
bool allConcreteShapes(const TypedGraph& g) {
  auto checkOne = [&](ValueId id) {
    const auto* t = static_cast<const TensorType*>(g.value(id).type);
    for (const Axis& ax : t->axes()) {
      if (ax.size.kind != Dimension::Kind::Const) return false;
    }
    return true;
  };
  for (ValueId p : g.params) {
    if (!checkOne(p)) return false;
  }
  return checkOne(g.output);
}

int64_t elementCount(const TensorType* t) {
  int64_t n = 1;
  for (const Axis& ax : t->axes()) n *= ax.size.constant;
  return n;
}

// A byte buffer compatible with StridedMemRefType at any runtime rank.
// MLIR's own C-interface ABI fixes this struct layout: {T* basePtr;
// T* data; int64_t offset; int64_t sizes[N]; int64_t strides[N];}
// (mlir/ExecutionEngine/CRunnerUtils.h). Building it through
// StridedMemRefType<float, N>, for a compile-time N, and copying its
// bytes, keeps the layout exactly right, without hand-writing the
// offsets.
struct MemRefBuffer {
  std::vector<char> bytes;
  void* ptr() { return bytes.data(); }
};

// One function per rank (1-4), instead of a template. StridedMemRefType
// itself needs a compile-time rank, since its `sizes` and `strides`
// arrays are fixed-size C arrays. So the rank must be a literal
// number somewhere. Writing it out four times here keeps this
// CLI-only helper readable, with no template code.
MemRefBuffer makeMemRefBuffer1(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 1> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.strides[0] = 1;
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer2(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 2> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.sizes[1] = sizes[1];
  m.strides[1] = 1;
  m.strides[0] = sizes[1];
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer3(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 3> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.sizes[1] = sizes[1];
  m.sizes[2] = sizes[2];
  m.strides[2] = 1;
  m.strides[1] = sizes[2];
  m.strides[0] = sizes[1] * sizes[2];
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer4(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 4> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.sizes[1] = sizes[1];
  m.sizes[2] = sizes[2];
  m.sizes[3] = sizes[3];
  m.strides[3] = 1;
  m.strides[2] = sizes[3];
  m.strides[1] = sizes[2] * sizes[3];
  m.strides[0] = sizes[1] * sizes[2] * sizes[3];
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer(float* data, const std::vector<int64_t>& sizes, bool& ok) {
  ok = true;
  switch (sizes.size()) {
    case 1: return makeMemRefBuffer1(data, sizes);
    case 2: return makeMemRefBuffer2(data, sizes);
    case 3: return makeMemRefBuffer3(data, sizes);
    case 4: return makeMemRefBuffer4(data, sizes);
    default:
      ok = false;
      return {};
  }
}

// Reads back element 0, and reports the total count, from a result
// descriptor, after the callee has overwritten it with its own
// allocation's data pointer. A returned-by-value memref is *not*
// written into the caller's original buffer. The ciface wrapper
// stores the callee's new pointer into the descriptor the caller
// passed in. So the descriptor's *fields*, not the original buffer,
// hold the real answer.
float readMemRefElement0(const MemRefBuffer& buf, int rank) {
  switch (rank) {
    case 1: return reinterpret_cast<const StridedMemRefType<float, 1>*>(buf.bytes.data())->data[0];
    case 2: return reinterpret_cast<const StridedMemRefType<float, 2>*>(buf.bytes.data())->data[0];
    case 3: return reinterpret_cast<const StridedMemRefType<float, 3>*>(buf.bytes.data())->data[0];
    case 4: return reinterpret_cast<const StridedMemRefType<float, 4>*>(buf.bytes.data())->data[0];
    default: return NAN;
  }
}

// Copies all `total` result elements out of a result descriptor, for
// --output. Like readMemRefElement0, it reads the callee's own `data`
// pointer (not the caller's original buffer). The result of these
// graphs is a fresh C-contiguous allocation, so reading data[0..total)
// gives the values in row-major order.
void readMemRefData(const MemRefBuffer& buf, int rank, int64_t total, float* out) {
  const float* data = nullptr;
  switch (rank) {
    case 1: data = reinterpret_cast<const StridedMemRefType<float, 1>*>(buf.bytes.data())->data; break;
    case 2: data = reinterpret_cast<const StridedMemRefType<float, 2>*>(buf.bytes.data())->data; break;
    case 3: data = reinterpret_cast<const StridedMemRefType<float, 3>*>(buf.bytes.data())->data; break;
    case 4: data = reinterpret_cast<const StridedMemRefType<float, 4>*>(buf.bytes.data())->data; break;
    default: return;
  }
  for (int64_t i = 0; i < total; ++i) out[static_cast<size_t>(i)] = data[i];
}

std::vector<int64_t> shapeOf(const TensorType* t) {
  std::vector<int64_t> shape;
  for (const Axis& ax : t->axes()) shape.push_back(ax.size.constant);
  return shape;
}

// Formats a shape as "[a, b, c]" for diagnostics.
std::string shapeStr(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + "]";
}

// Runs the Level-1, team-authored graph optimization passes on every
// graph in the program, in place. This is a separate function from
// lowerToModule below, so the CLI can show the optimization phase's
// own output (--emit-optimized-graph) before lowering ever runs.
// Otherwise the optimization phase's effect would stay invisible,
// hidden inside the MLIR the lowering phase prints.
void runOptimizationPasses(TypedProgram& program) {
  DiagnosticEngine passDiags;
  PassContext passCtx{program.types(), passDiags};
  PassManager passManager;
  passManager.add(createShapeFoldingPass());
  passManager.add(createReshapeCancellationPass());
  passManager.add(createTransposeCancellationPass());
  passManager.add(createElementwiseFusionPass());
  passManager.add(createLinearBiasActivationFusionPass());
  passManager.add(createDeadOpEliminationPass());
  // TypedProgram stores each graph by value. Each pass mutates the
  // graph in place, through the same TypedGraph& the program already
  // owns.
  for (const TypedGraph& g : program.graphs()) {
    passManager.runAll(const_cast<TypedGraph&>(g), passCtx);
  }
}

// Lowers an already-optimized program to MLIR. Call
// runOptimizationPasses first. This function does not run the
// Level-1 passes itself, so the lowering phase's own output
// (--emit-mlir) shows only what lowering itself contributes.
mlir::OwningOpRef<mlir::ModuleOp> lowerToModule(mlir::MLIRContext& ctx, TypedProgram& program) {
  MlirLowerer lowerer(ctx, kTilingPolicy);
  mlir::ModuleOp module = lowerer.lower(program);
  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

void printUsage(const char* prog) {
  std::cerr << "usage: " << prog << " <stage-flag> <file.tns>\n"
            << "  --emit-tokens          lexer: print the token stream\n"
            << "  --emit-parse-tree      parser: print the raw ANTLR parse tree (LISP-style)\n"
            << "  --emit-ast             parser: print the parsed AST\n"
            << "  --emit-ast-dot         parser: print the parsed AST as Graphviz DOT\n"
            << "  --check                semantic analysis: report OK or diagnostics\n"
            << "  --emit-symbols         semantic analysis: print the symbol table (RM 3.6)\n"
            << "  --emit-graph           semantic analysis: print the Typed Computation Graph\n"
            << "  --emit-graph-dot       semantic analysis: print the Typed Computation Graph as Graphviz DOT\n"
            << "  --explain              semantic analysis: like --emit-graph, annotated per RM section\n"
            << "  --emit-optimized-graph optimization: print the graph after the Level-1 passes\n"
            << "  --emit-optimized-graph-dot  optimization: the post-optimization graph as Graphviz DOT\n"
            << "  --emit-mlir            lowering: print the MLIR module\n"
            << "  --emit-llvm            backend: print the translated LLVM IR\n"
            << "  --run                  full pipeline: JIT-execute with synthetic all-ones input data\n"
            << "  -o <out.o>             full pipeline: emit a native object file (ahead-of-time)\n"
            << "\n"
            << "  --data-dir <dir>       with --run: load each input/weight from <dir>/<name>.npy\n"
            << "                         (float32) instead of synthetic all-ones data\n"
            << "  --output <out.npy>     with --run: write the full result tensor to a .npy file\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string stage;
  std::string path;
  std::string outPath;
  std::string dataDir;   // --data-dir: where to read real input/weight .npy files
  std::string outputNpy; // --output: where to write the result tensor as .npy
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      stage = "-o";
      outPath = argv[++i];
    } else if (arg == "--data-dir" && i + 1 < argc) {
      dataDir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      outputNpy = argv[++i];
    } else if (!arg.empty() && arg[0] == '-') {
      stage = arg;
    } else {
      path = arg;
    }
  }

  if (path.empty() || stage.empty()) {
    printUsage(argv[0]);
    return 2;
  }

  bool ok = false;
  std::string text = readFile(path, ok);
  if (!ok) {
    std::cerr << "cannot open " << path << "\n";
    return 2;
  }

  SourceBuffer src(path, text);
  DiagnosticEngine diags;

  if (stage == "--emit-tokens") {
    Lexer lexer(src, diags);
    std::vector<Token> tokens = lexer.tokenize();
    for (const Token& t : tokens) {
      std::cout << t.span.line << ":" << t.span.col << "  " << tokenKindName(t.kind);
      if (t.kind == TokenKind::Identifier || t.kind == TokenKind::IntLiteral || t.kind == TokenKind::FloatLiteral) {
        std::cout << "  '" << std::string(t.lexeme) << "'";
      }
      std::cout << "\n";
    }
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }

  if (stage == "--emit-parse-tree") {
    std::cout << dumpParseTree(src) << "\n";
    return 0;
  }

  Parser parser(src, diags);
  Result<ast::AstProgram> parsed = parser.parseProgram();

  if (stage == "--emit-ast") {
    printAstTree(parsed.value());
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (stage == "--emit-ast-dot") {
    std::cout << astToDot(parsed.value());
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (diags.hasErrors()) {
    std::cerr << diags.render(src);
    return 1;
  }

  TypeContext types;
  SemanticAnalyzer analyzer(types, diags);
  Result<TypedProgram> analyzed = analyzer.analyze(parsed.value());

  if (stage == "--check") {
    std::cerr << diags.render(src);
    if (diags.hasErrors()) return 1;
    std::cout << "OK: " << path << " is a well-formed Tensora program\n";
    return 0;
  }
  if (stage == "--emit-symbols") {
    printSymbolTable(analyzer.symbolTable(), types);
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (stage == "--emit-graph" || stage == "--explain") {
    for (const TypedGraph& g : analyzed.value().graphs()) printTypedGraph(g, types, stage == "--explain");
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (stage == "--emit-graph-dot") {
    for (const TypedGraph& g : analyzed.value().graphs()) std::cout << graphToDot(g, types);
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (diags.hasErrors()) {
    std::cerr << diags.render(src);
    return 1;
  }

  TypedProgram& program = analyzed.value();

  if (stage == "--emit-optimized-graph") {
    runOptimizationPasses(program);
    for (const TypedGraph& g : program.graphs()) printTypedGraph(g, types, /*explainMode=*/false);
    return 0;
  }

  // Same DOT view as --emit-graph-dot, but after the Level-1 passes, so
  // a before/after render shows what optimization removed or fused.
  if (stage == "--emit-optimized-graph-dot") {
    runOptimizationPasses(program);
    for (const TypedGraph& g : program.graphs()) std::cout << graphToDot(g, types);
    return 0;
  }

  // This driver found a known limitation by running against a real
  // symbolic-batch source file (mlp.tns): MlirLowerer's
  // `tensor.empty()` calls do not yet supply dynamic-size operands for
  // a symbolic dimension. ARCH-005 says symbolic maps to dynamic `?`,
  // but only the *type* lowering follows that rule; the *value*
  // lowering for every op that builds a fresh tensor (elementwise,
  // relu, matmul's and conv2d's accumulator, and so on) assumes every
  // dim is static. Fixing this touches every lowering function, and
  // deserves its own pass and its own tests, not a rushed patch here.
  // Until then, every stage past semantic analysis needs concrete
  // shapes throughout.
  for (const TypedGraph& g : program.graphs()) {
    if (!allConcreteShapes(g)) {
      std::cerr << "error: " << stage << " requires every input/weight and the output to have concrete "
                   "(non-symbolic) sizes; MLIR lowering does not yet support symbolic dimensions past "
                   "type-checking (known limitation -- see docs/testing/TEST_MATRIX.md LOW-005)\n";
      return 1;
    }
  }

  // Runs the optimization phase, then the lowering phase (two
  // separate CLI-visible stages: --emit-optimized-graph and
  // --emit-mlir). Both run here for every later stage
  // (--emit-mlir onward), since each stage needs the one before it.
  runOptimizationPasses(program);

  mlir::DialectRegistry registry = makeTensoraDialectRegistry();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module = lowerToModule(ctx, program);

  // Level-2 / USP: runs the project-authored shape-guided matmul
  // tiling pass on every function. This is a real MLIR pass, distinct
  // from the Level-1 TCG passes already applied above, and from the
  // standard MLIR/LLVM infrastructure below. The default policy (see
  // LoopTiling.h) only tiles a matmul at or above its size threshold.
  // So this is a genuine no-op on small demo graphs, not a forced
  // effect.
  {
    mlir::PassManager tilingPm(&ctx);
    tilingPm.addNestedPass<mlir::func::FuncOp>(createMatmulTilingPass(kTilingPolicy));
    if (failed(tilingPm.run(*module))) {
      std::cerr << "error: MatmulTilingPass failed\n";
      return 1;
    }
  }
  if (failed(mlir::verify(*module))) {
    std::cerr << "internal error: lowered module does not verify\n";
    module->dump();
    return 1;
  }

  if (stage == "--emit-mlir") {
    module->print(llvm::outs());
    return 0;
  }

  // The standard MLIR/LLVM lowering pipeline. Every pass here is
  // stock MLIR infrastructure, not written by this project. This
  // always runs the full set of passes every workload in this phase
  // has needed (subview-bearing tiled loops, softmax's math.exp,
  // plain elementwise/matmul/conv graphs). Each pass is a no-op when
  // its input pattern is not present.
  mlir::PassManager pm(&ctx);
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (failed(pm.run(*module))) {
    std::cerr << "error: standard MLIR/LLVM lowering pipeline failed\n";
    module->dump();
    return 1;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext, path);
  if (!llvmModule) {
    std::cerr << "error: MLIR->LLVM IR translation failed\n";
    return 1;
  }

  if (stage == "--emit-llvm") {
    llvmModule->print(llvm::outs(), nullptr);
    return 0;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  llvmModule->setTargetTriple(targetTriple);
  std::string lookupError;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple.str(), lookupError);
  if (!target) {
    std::cerr << "error: TargetRegistry::lookupTarget: " << lookupError << "\n";
    return 1;
  }
  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, "generic", "", targetOptions, llvm::Reloc::PIC_));
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  if (stage == "-o") {
    std::error_code ec;
    llvm::raw_fd_ostream dest(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      std::cerr << "error: could not open '" << outPath << "': " << ec.message() << "\n";
      return 1;
    }
    llvm::legacy::PassManager codegenPm;
    if (targetMachine->addPassesToEmitFile(codegenPm, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
      std::cerr << "error: TargetMachine can't emit an object file for this target\n";
      return 1;
    }
    codegenPm.run(*llvmModule);
    dest.flush();
    std::cout << "wrote native object file: " << outPath << "\n";
    return 0;
  }

  if (stage == "--run") {
    if (program.graphs().size() != 1) {
      std::cerr << "error: --run requires exactly one graph in the file (found " << program.graphs().size()
                << "); disambiguation across multiple graphs is not supported yet\n";
      return 1;
    }
    const TypedGraph& graph = program.graphs().front();
    if (!allConcreteShapes(graph)) {
      std::cerr << "error: --run requires every input/weight and the output to have concrete (non-symbolic) "
                   "sizes, since it must synthesize input data and cannot allocate a symbolic-sized buffer\n";
      return 1;
    }

    mlir::ExecutionEngineOptions engineOpts;
    static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
    static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
    engineOpts.sharedLibPaths = kSharedLibs;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOpts);
    if (!maybeEngine) {
      std::cerr << "error: ExecutionEngine::create failed\n";
      return 1;
    }
    auto engine = std::move(*maybeEngine);

    // Without --data-dir, uses synthetic all-ones data for every
    // declared input and weight. Tensora source declares shapes, not
    // values (there is no tensor-literal syntax), so a plain CLI run
    // has no other source of data. All-ones is a deliberate, checkable
    // choice, not an arbitrary one: for example, in a matmul, every
    // output element then equals the contracted dimension K, which you
    // can check by hand. With --data-dir, each param's real values are
    // read from <dir>/<name>.npy instead (see the loop below).
    std::vector<std::vector<float>> storage;
    std::vector<MemRefBuffer> argBufs;
    std::vector<void*> packedArgs;

    const auto* outType = static_cast<const TensorType*>(graph.value(graph.output).type);
    std::vector<int64_t> outShape = shapeOf(outType);
    std::vector<float> outStorage(static_cast<size_t>(elementCount(outType)), 0.0f);
    bool rankOk = false;
    MemRefBuffer resultBuf = makeMemRefBuffer(outStorage.data(), outShape, rankOk);
    if (!rankOk) {
      std::cerr << "error: --run only supports rank 1-4 tensors; output has rank " << outShape.size() << "\n";
      return 1;
    }
    packedArgs.push_back(resultBuf.ptr());

    for (ValueId p : graph.params) {
      const auto* pType = static_cast<const TensorType*>(graph.value(p).type);
      std::vector<int64_t> pShape = shapeOf(pType);
      const std::string& pName = graph.value(p).debugName;
      if (dataDir.empty()) {
        storage.emplace_back(static_cast<size_t>(elementCount(pType)), 1.0f);
      } else {
        // Load this param's real values from <dir>/<name>.npy and check
        // the file's shape matches what the .tns program declared.
        std::string npyPath = dataDir + "/" + pName + ".npy";
        npy::NpyArray arr;
        std::string npyErr;
        if (!npy::read(npyPath, arr, npyErr)) {
          std::cerr << "error: --data-dir: " << npyErr << "\n";
          return 1;
        }
        if (arr.shape != pShape) {
          std::cerr << "error: --data-dir: '" << pName << "' shape mismatch: declared " << shapeStr(pShape)
                     << ", file " << npyPath << " has " << shapeStr(arr.shape) << "\n";
          return 1;
        }
        storage.emplace_back(std::move(arr.data));
      }
      MemRefBuffer buf = makeMemRefBuffer(storage.back().data(), pShape, rankOk);
      if (!rankOk) {
        std::cerr << "error: --run only supports rank 1-4 tensors; '" << pName << "' has rank " << pShape.size()
                   << "\n";
        return 1;
      }
      argBufs.push_back(std::move(buf));
    }
    for (MemRefBuffer& buf : argBufs) packedArgs.push_back(buf.ptr());

    // Invokes the compiled graph through MLIR's own packed calling
    // convention (ExecutionEngine::invokePacked), the same path the
    // library's `invoke()` helper uses. The packed C-interface wrapper
    // reads its arguments from one `void*[]`, so there is no per-arity
    // code and no cap on the number of inputs/weights: each element
    // points to one descriptor pointer in `packedArgs` (result first,
    // then params in declared order), matching how `invoke()` passes
    // the address of each argument value.
    std::vector<void*> argAddrs;
    argAddrs.reserve(packedArgs.size());
    for (void*& p : packedArgs) argAddrs.push_back(&p);
    if (llvm::Error err = engine->invokePacked("_mlir_ciface_" + graph.name, argAddrs)) {
      std::cerr << "error: JIT invocation of '" << graph.name << "' failed: " << llvm::toString(std::move(err)) << "\n";
      return 1;
    }

    int64_t total = elementCount(outType);
    float first = readMemRefElement0(resultBuf, static_cast<int>(outShape.size()));
    if (dataDir.empty()) {
      std::cout << "ran '" << graph.name << "' with synthetic all-ones inputs\n";
    } else {
      std::cout << "ran '" << graph.name << "' with inputs from " << dataDir << "\n";
    }
    std::cout << "output shape: [";
    for (size_t i = 0; i < outShape.size(); ++i) std::cout << (i ? ", " : "") << outShape[i];
    std::cout << "], " << total << " elements; element[0] = " << first << "\n";

    // --output writes the whole result tensor (not just element[0]) so a
    // demo wrapper can read the full logit vector and compare against a
    // reference. stdout above is unchanged either way.
    if (!outputNpy.empty()) {
      std::vector<float> result(static_cast<size_t>(total));
      readMemRefData(resultBuf, static_cast<int>(outShape.size()), total, result.data());
      std::string npyErr;
      if (!npy::write(outputNpy, result.data(), outShape, npyErr)) {
        std::cerr << "error: --output: " << npyErr << "\n";
        return 1;
      }
    }
    return 0;
  }

  printUsage(argv[0]);
  return 2;
}
