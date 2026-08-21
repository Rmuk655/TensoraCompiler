// This file implements the tensor-liveness analysis.
// It computes each value's live interval and a peak-live-byte estimate
// from the verified Typed Computation Graph. Codegen's Level-2 tiling
// decision reads this estimate (see codegen/LoopTiling.cpp).
#include "tensora/optimization/Liveness.h"

#include <algorithm>
#include <vector>

namespace tensora {

int64_t tensorByteSize(const TensorType* t, const std::unordered_map<SymbolId, int64_t>& symbolBindings) {
  int64_t product = 1;
  for (const Axis& ax : t->axes()) {
    if (ax.size.kind == Dimension::Kind::Const) {
      product *= ax.size.constant;
    } else {
      auto it = symbolBindings.find(ax.size.symbol);
      product *= (it != symbolBindings.end()) ? it->second : 1;
    }
  }
  return product * 4;  // f32 (RM 3.23): 4 bytes/element
}

LivenessResult LivenessAnalysis::run(const TypedGraph& graph,
                                      const std::unordered_map<SymbolId, int64_t>& symbolBindings) const {
  LivenessResult result;
  const auto numNodes = static_cast<long>(graph.nodes().size());

  std::vector<ValueId> allValues;
  allValues.reserve(graph.params.size() + graph.nodes().size());
  for (ValueId p : graph.params) allValues.push_back(p);
  for (const GraphNode& n : graph.nodes()) allValues.push_back(n.output);

  for (ValueId v : allValues) {
    NodeId producer = graph.producerOf(v);
    long firstPos = (producer == kNoProducer) ? -1 : static_cast<long>(producer);
    long lastPos = firstPos;
    for (NodeId c : graph.consumersOf(v)) lastPos = std::max(lastPos, static_cast<long>(c));
    if (v == graph.output) lastPos = std::max(lastPos, numNodes);  // `return` is a use

    Interval interval;
    interval.firstDef = producer;
    interval.lastUse = static_cast<NodeId>(lastPos);
    interval.bytes = tensorByteSize(static_cast<const TensorType*>(graph.value(v).type), symbolBindings);
    result.intervals[v] = interval;
  }

  // A sweep line over "position" (a node index, or -1 for params, or
  // numNodes for `return`). This tracks the net byte change at each
  // position. The max of the running sum is the peak footprint of all
  // values live at the same time.
  std::unordered_map<long, int64_t> delta;
  for (ValueId v : allValues) {
    const Interval& iv = result.intervals.at(v);
    long firstPos = (iv.firstDef == kNoProducer) ? -1 : static_cast<long>(iv.firstDef);
    long lastPos = static_cast<long>(iv.lastUse);
    delta[firstPos] += iv.bytes;
    delta[lastPos + 1] -= iv.bytes;
  }
  std::vector<long> positions;
  positions.reserve(delta.size());
  for (const auto& [pos, d] : delta) positions.push_back(pos);
  std::sort(positions.begin(), positions.end());

  int64_t running = 0, peak = 0;
  for (long pos : positions) {
    running += delta.at(pos);
    peak = std::max(peak, running);
  }
  result.peakLiveBytes = peak;
  return result;
}

bool exceedsMemoryBudget(const LivenessResult& result, int64_t budgetBytes) {
  return result.peakLiveBytes > budgetBytes;
}

}  // namespace tensora
