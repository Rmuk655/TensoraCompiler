#pragma once
#include <cstdint>
#include <unordered_map>

#include "tensora/shared/Ir.h"

// Tensor liveness (ARCH-009; INTERFACES.md §4a): a real data-flow
// analysis over the SSA Typed Computation Graph. It is read-only, and
// its result feeds real optimization/codegen decisions (e.g.
// codegen/LoopTiling.h), not just a report number.
namespace tensora {

// f32 byte size of a tensor type (RM 3.23: no dtype, all f32). An unbound
// symbolic dimension is conservatively treated as size 1 -- callers that
// know a concrete size (e.g. a benchmark's chosen batch size) should
// supply it via `symbolBindings`.
int64_t tensorByteSize(const TensorType* t, const std::unordered_map<SymbolId, int64_t>& symbolBindings = {});

struct Interval {
  // kNoProducer for a graph parameter (live from graph entry, i.e.
  // before any node).
  NodeId firstDef = kNoProducer;
  // The NodeId of the last node that consumes this value, or
  // graph.nodes().size() (one past the last real node) if the value is
  // (also) the graph's `return`ed output.
  NodeId lastUse = 0;
  int64_t bytes = 0;
};

struct LivenessResult {
  std::unordered_map<ValueId, Interval> intervals;
  int64_t peakLiveBytes = 0;  // max total size of simultaneously-live tensors, anywhere in the graph
};

class LivenessAnalysis {
 public:
  LivenessResult run(const TypedGraph& graph,
                      const std::unordered_map<SymbolId, int64_t>& symbolBindings = {}) const;
};

// A minimal real "decision that consumes the estimate" (ARCH-009,
// INTERFACES.md §4a): would a schedule with this peak live-memory
// footprint exceed a given budget? Used directly by TilingDecision
// (codegen/LoopTiling.h) and independently unit-tested (LIVE-004).
bool exceedsMemoryBudget(const LivenessResult& result, int64_t budgetBytes);

}  // namespace tensora
