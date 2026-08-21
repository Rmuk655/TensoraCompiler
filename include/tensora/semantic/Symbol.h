#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tensora/shared/SourceSpan.h"
#include "tensora/shared/Type.h"

// Name resolution and scope (RM 3.6). There are only two scope levels,
// matching the grammar: a flat global scope (input/weight declarations,
// shared by every graph) and one flat local scope per graph (value
// bindings from statements). The local scope resets at the start of
// each graph.
namespace tensora {

struct Symbol {
  enum class Kind { Input, Weight, Value };
  std::string name;
  Kind kind = Kind::Value;
  const Type* type = nullptr;  // verified decl type, or the RHS's inferred type
  SourceSpan span;             // declaration/binding site
};

class SymbolTable {
 public:
  // Returns false, and does not insert, if `name` is already declared
  // globally -- RM 3.6 says redeclaring a top-level name is rejected.
  bool declareGlobal(const Symbol& sym);
  const Symbol* lookupGlobal(std::string_view name) const;

  // Resets the local (per-graph) scope. Call this once at the start of
  // each graph.
  void enterGraph();

  enum class BindOutcome {
    Ok,           // A fresh local bind, or a reassignment of an existing local Value (shadows, RM 3.6).
    KindConflict  // `name` already exists globally as Input/Weight (see the note below).
  };
  // RM 3.6 says a rebind, in the same graph, of a name already bound to
  // a different declaration kind is rejected. It also says reassigning
  // an already-bound value name within a graph is allowed and shadows
  // the old value. The only way a name can have two different *kinds*
  // is Input/Weight (global) vs. Value (local); a purely local rebind
  // is always Value -> Value, which the second rule allows. So this
  // class treats a local assignment that shadows a GLOBAL input/weight
  // name as the kind conflict the first rule forbids. A local
  // assignment that only shadows an earlier LOCAL binding is always
  // Ok. (The RM text does not force this reading; see the semantic
  // implementation report's Specification/Ambiguity Issues section.)
  BindOutcome bindLocal(const std::string& name, Symbol sym);

  // Resolves a name for use inside the current graph. It checks the
  // most recent local binding first, then the global scope. Returns
  // nullptr if the name is undefined (RM 3.6).
  const Symbol* resolve(std::string_view name) const;

  // Declaration-order views of the two scopes. These hold the same
  // Symbol objects `resolve` reads, in the order they were first
  // accepted. (A rebind that shadows an earlier local does not add a
  // second entry; it updates the existing one in place, the same way
  // `locals_` does.) --emit-symbols (main.cpp) renders this data, so
  // its output shows the real resolution state, not a second, separate
  // rebuild of it. `orderedLocals` reflects the *current* graph's
  // scope. The next call to `enterGraph` clears it.
  const std::vector<const Symbol*>& orderedGlobals() const { return globalOrder_; }
  const std::vector<const Symbol*>& orderedLocals() const { return localOrder_; }

 private:
  std::unordered_map<std::string, Symbol> globals_;
  std::unordered_map<std::string, Symbol> locals_;
  std::vector<const Symbol*> globalOrder_;
  std::vector<const Symbol*> localOrder_;
};

}  // namespace tensora
