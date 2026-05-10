#include "axonforge/graph.hpp"

namespace axonforge {
// ============================================================
// Operator Fusion Pass
//
// Patterns (in priority order):
//   [P1] MATMUL + ADD                       → FUSED_MATMUL_BIAS
//   [P1] MATMUL + ADD + {RELU|SILU|GELU}    → FUSED_MATMUL_BIAS_ACT
//   [P1] RMS_NORM + MUL (scale)             → FUSED_RMS_NORM_SCALE
//   [P2] consecutive element-wise unary ops → single kernel loop
//
// TODO(Phase1): implement pattern matching and graph rewrite.
// ============================================================

struct FusePattern {
    std::vector<OpType> ops;   // sequence to match
    OpType              fused; // replacement op
};

static const std::vector<FusePattern> kFusePatterns = {
    {{OpType::MATMUL, OpType::ADD},               OpType::FUSED_MATMUL_BIAS},
    {{OpType::MATMUL, OpType::ADD, OpType::RELU},  OpType::FUSED_MATMUL_BIAS_ACT},
    {{OpType::MATMUL, OpType::ADD, OpType::SILU},  OpType::FUSED_MATMUL_BIAS_ACT},
    {{OpType::RMS_NORM, OpType::MUL},              OpType::FUSED_RMS_NORM_SCALE},
};

void run_fuse_pass(ComputeGraph& /*graph*/) {
    // TODO(Phase1):
    // 1. Walk nodes in topo order
    // 2. For each node, try to match a FusePattern starting from it
    // 3. If match found, replace the matched nodes with one fused node
    // 4. Repeat until no more matches
}

} // namespace axonforge
