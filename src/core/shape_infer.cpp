#include "axonforge/graph.hpp"
#include <stdexcept>

namespace axonforge {
// ============================================================
// Shape Inference
//
// Given a ComputeGraph with concrete TensorDescs (all DimExpr are
// int64_t after instantiate()), propagate output shapes through
// every node using per-op shape inference rules.
//
// TODO(Phase0): implement per-op rules.
// ============================================================

// Infer the output shape of a MATMUL node.
// a: [..., M, K]  b: [..., K, N]  → [..., M, N]
static SymbolicShape infer_matmul(const SymbolicShape& a,
                                   const SymbolicShape& b) {
    if (a.size() < 2 || b.size() < 2) {
        throw std::runtime_error("ShapeInfer: matmul requires at least 2D tensors");
    }
    SymbolicShape out = a;
    out.back() = b.back();
    return out;
}

// TODO: implement full shape inference pass over ComputeGraph
void infer_shapes(ComputeGraph& /*graph*/) {
    // Walk topo_order, call per-op infer function, update tensor descs
}

} // namespace axonforge
