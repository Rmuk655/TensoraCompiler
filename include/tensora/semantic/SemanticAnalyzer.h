#pragma once
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Ir.h"
#include "tensora/shared/Result.h"
#include "tensora/semantic/Symbol.h"
#include "tensora/shared/DimSolver.h"
#include "tensora/shared/Type.h"

// Drives semantic analysis (ARCHITECTURE.md §6). It resolves names,
// infers types and shapes through the operator contracts
// (semantic/Contracts.h), and builds a Typed Computation Graph. It
// never depends on MLIR or LLVM.
namespace tensora {

// A declaration-order snapshot of the entries SymbolTable accepted,
// captured while `analyze()` runs (--emit-symbols in main.cpp renders
// this; see SymbolTable::orderedGlobals/orderedLocals for why this is a
// snapshot of the real resolution state, not a second, separate
// rebuild of it). There is one GraphScope per graph, in program order.
// `locals` is that graph's final local scope: a rebind that shadows an
// earlier local updates the same entry in place, so only the last
// value stays, matching what name resolution itself would see at the
// end of that graph.
struct SymbolTableReport {
  std::vector<Symbol> globals;
  struct GraphScope {
    std::string graphName;
    std::vector<Symbol> locals;
    // One entry per distinct name referenced in this graph's body, in
    // first-reference order: (referenced name, "global::name" /
    // "local::name" / "undefined"). This is a real log of what
    // SymbolTable::resolve() returned for that reference, not a
    // recomputed value.
    std::vector<std::pair<std::string, std::string>> resolutions;
  };
  std::vector<GraphScope> graphs;
};

class SemanticAnalyzer {
 public:
  SemanticAnalyzer(TypeContext& types, DiagnosticEngine& diags) : types_(types), diags_(diags) {}

  Result<TypedProgram> analyze(const ast::AstProgram& program);

  // Valid after analyze() returns. It reflects exactly the symbols
  // `analyze()` inserted into `symbols_` (it skips rejected
  // redeclarations, since those never entered the table).
  const SymbolTableReport& symbolTable() const { return symbolReport_; }

 private:
  // Per-graph state: a fresh GraphBuilder and DimSolver for each graph
  // (RM 3.4 scopes dimension reasoning to "within one graph"), plus the
  // name -> ValueId maps for graph-local bindings and lazily-added
  // global params.
  struct GraphCtx {
    GraphBuilder builder;
    DimSolver solver;
    std::unordered_map<std::string, ValueId> localValues;
    std::unordered_map<std::string, ValueId> globalValues;
    // First-reference-order resolution log for this graph (see
    // SymbolTableReport::GraphScope::resolutions). loggedNames removes
    // duplicates, so a name used on several lines gets one entry, not
    // one per use.
    std::vector<std::pair<std::string, std::string>> resolutionLog;
    std::unordered_set<std::string> loggedNames;
  };

  TypedGraph analyzeGraph(const ast::GraphDecl& g);

  // Resolves a plain identifier operand. If it is undefined, reports
  // SemaUndefinedName and returns an InvalidType poison value. Adds a
  // global input/weight as this graph's param the first time the graph
  // references it.
  std::pair<ValueId, const Type*> resolveOperand(GraphCtx& gc, const std::string& name, SourceSpan span);
  // RM 3.24 `operand ::= identifier | floatLit`.
  std::pair<ValueId, const Type*> resolveExprOperand(GraphCtx& gc, const ast::Expr& e);

  std::pair<ValueId, const Type*> analyzeCall(GraphCtx& gc, const ast::CallExpr& call);
  std::pair<ValueId, const Type*> analyzeMatMul(GraphCtx& gc, const ast::MatMulExpr& mm);
  std::pair<ValueId, const Type*> analyzeBinary(GraphCtx& gc, const ast::BinaryExpr& bin);

  TypeContext& types_;
  DiagnosticEngine& diags_;
  SymbolTable symbols_;
  SymbolTableReport symbolReport_;
};

}  // namespace tensora
