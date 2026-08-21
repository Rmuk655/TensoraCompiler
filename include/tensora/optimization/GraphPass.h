#pragma once
#include <memory>
#include <string_view>
#include <vector>

#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Ir.h"
#include "tensora/shared/Type.h"

// Level-1 graph optimization (RM 3.21; ARCHITECTURE.md §6a): the six
// frozen v1 passes over the SSA Typed Computation Graph. Every pass
// must preserve the observable `return` value of the graph, for all
// inputs that satisfy the checked program's semantics (RM 3.21). See
// each pass's own precondition comment in optimization/GraphPasses.cpp
// for why its transformation is legal.
namespace tensora {

struct PassStats {
  size_t numChanges = 0;  // e.g. nodes removed/fused/folded; pass-specific meaning, always 0 == no-op
  size_t iterations = 0;  // set by PassManager::runAll: how many full sweeps over the pass list ran
};

struct PassResult {
  bool changed = false;
  DiagnosticList diagnostics;
  PassStats stats;
};

struct PassContext {
  TypeContext& types;
  DiagnosticEngine& diags;
};

class GraphPass {
 public:
  virtual ~GraphPass() = default;
  virtual std::string_view name() const = 0;
  // Structural passes rebuild `graph` in place (via a fresh GraphBuilder
  // internally, then move-assign) rather than mutating it directly,
  // since GraphBuilder remains "the only sanctioned way to create
  // nodes/values" (ARCH-008) even during optimization. Pure annotation
  // passes (tagging via TypedGraph::addFact) mutate in place directly.
  virtual PassResult run(TypedGraph& graph, PassContext& ctx) = 0;
};

// Runs the fixed sequence of passes to a fixed point: one "sweep" runs
// every pass once, in order; runAll repeats the sweep as long as the
// last sweep changed the graph, up to kMaxSweeps. This is what lets one
// pass's output create a new opportunity for an earlier pass -- e.g.
// DeadOpElimination dropping a value can bring a Linear node's
// consumer count down to one, which is exactly what
// LinearBiasActivationFusion requires but did not see in the sweep
// before. Every pass here is monotonic (a structural pass only removes
// or merges nodes; a tagging pass only adds a fact, and never re-tags a
// node it already tagged -- see GraphPasses.cpp), so a sweep with no
// change means every pass has actually reached its own fixed point, not
// just paused. kMaxSweeps is a defensive cap, not something a correct
// pass set should ever hit; it exists so a future pass bug (one that
// reports `changed` without making real progress) fails safe -- it
// stops, instead of hanging the compiler.
class PassManager {
 public:
  void add(std::unique_ptr<GraphPass> pass) { passes_.push_back(std::move(pass)); }

  PassResult runAll(TypedGraph& graph, PassContext& ctx) {
    constexpr size_t kMaxSweeps = 10;
    PassResult combined;
    for (; combined.stats.iterations < kMaxSweeps; ++combined.stats.iterations) {
      bool sweepChanged = false;
      for (auto& pass : passes_) {
        PassResult r = pass->run(graph, ctx);
        sweepChanged = sweepChanged || r.changed;
        combined.changed = combined.changed || r.changed;
        combined.stats.numChanges += r.stats.numChanges;
        combined.diagnostics.insert(combined.diagnostics.end(), r.diagnostics.begin(), r.diagnostics.end());
      }
      if (!sweepChanged) {
        ++combined.stats.iterations;  // count the final, no-change confirmation sweep too
        break;
      }
    }
    return combined;
  }

  const std::vector<std::unique_ptr<GraphPass>>& passes() const { return passes_; }

 private:
  std::vector<std::unique_ptr<GraphPass>> passes_;
};

// The six frozen v1 passes (RM 3.21). Constant folding was retired: its
// same-kind scalar-chain combining cannot be proven to preserve observable
// output under IEEE-754 f32 non-associativity, which RM 3.21 requires of
// every enabled pass.
std::unique_ptr<GraphPass> createShapeFoldingPass();
std::unique_ptr<GraphPass> createDeadOpEliminationPass();
std::unique_ptr<GraphPass> createReshapeCancellationPass();
std::unique_ptr<GraphPass> createTransposeCancellationPass();
std::unique_ptr<GraphPass> createElementwiseFusionPass();
std::unique_ptr<GraphPass> createLinearBiasActivationFusionPass();

}  // namespace tensora
