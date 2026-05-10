#include "axonforge/memory.hpp"

// CpuTensorStorage and MemoryArena are implemented in src/core/tensor.cpp.
// This file is reserved for any x86-specific allocation extensions
// (e.g., huge-page backed arenas, NUMA-aware allocation).

namespace axonforge::cpu_x86 {
// TODO(Phase1): huge-page arena using mmap(MAP_HUGETLB) for large weight tensors
} // namespace axonforge::cpu_x86
