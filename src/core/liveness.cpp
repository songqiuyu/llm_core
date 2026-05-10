#include "axonforge/graph.hpp"
#include "axonforge/memory.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace axonforge {
// ============================================================
// Liveness Analysis + Memory Planning
//
// Algorithm:
//   1. Walk nodes in topological order.
//   2. For each tensor, record birth_step (first producer) and
//      death_step (last consumer).
//   3. Greedy coloring: assign slots to tensors such that tensors
//      with non-overlapping lifetimes share the same slot.
//      Slots are sized to the largest tensor that uses them.
//
// TODO(Phase1): Implement full liveness analysis.
// ============================================================

struct TensorLifetime {
    TensorId id;
    int      birth_step; // index in topo_order where tensor is first produced
    int      death_step; // index in topo_order where tensor is last consumed
    size_t   nbytes;
};

struct AllocationSlot {
    size_t   offset;     // offset within the arena buffer
    size_t   size;       // slot size in bytes
};

struct MemoryPlan {
    size_t                                  total_bytes{0};
    std::unordered_map<TensorId, size_t>    tensor_offsets; // tensor → arena offset
};

// TODO(Phase1): implement
MemoryPlan plan_memory(const ComputeGraph& /*graph*/,
                       const std::vector<size_t>& /*tensor_nbytes*/) {
    return MemoryPlan{};
}

} // namespace axonforge
