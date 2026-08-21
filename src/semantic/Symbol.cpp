// This file implements the symbol table.
// Semantic analysis uses it to resolve names and enforce scope rules
// (RM 3.6).
#include "tensora/semantic/Symbol.h"

namespace tensora {

bool SymbolTable::declareGlobal(const Symbol& sym) {
  auto [it, inserted] = globals_.emplace(sym.name, sym);
  if (inserted) globalOrder_.push_back(&it->second);
  return inserted;
}

const Symbol* SymbolTable::lookupGlobal(std::string_view name) const {
  auto it = globals_.find(std::string(name));
  return it == globals_.end() ? nullptr : &it->second;
}

void SymbolTable::enterGraph() {
  locals_.clear();
  localOrder_.clear();
}

SymbolTable::BindOutcome SymbolTable::bindLocal(const std::string& name, Symbol sym) {
  if (locals_.find(name) == locals_.end() && lookupGlobal(name) != nullptr) {
    return BindOutcome::KindConflict;
  }
  auto it = locals_.find(name);
  if (it == locals_.end()) {
    auto [newIt, inserted] = locals_.emplace(name, std::move(sym));
    localOrder_.push_back(&newIt->second);
  } else {
    it->second = std::move(sym);  // shadows the earlier local in place, same entry
  }
  return BindOutcome::Ok;
}

const Symbol* SymbolTable::resolve(std::string_view name) const {
  auto it = locals_.find(std::string(name));
  if (it != locals_.end()) return &it->second;
  return lookupGlobal(name);
}

}  // namespace tensora
