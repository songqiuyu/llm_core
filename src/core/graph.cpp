#include "axonforge/graph.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace axonforge {

// ---- op_type_name ----

std::string_view op_type_name(OpType op) noexcept {
    switch (op) {
        case OpType::NEG:                  return "neg";
        case OpType::ABS:                  return "abs";
        case OpType::SQRT:                 return "sqrt";
        case OpType::EXP:                  return "exp";
        case OpType::LOG:                  return "log";
        case OpType::RELU:                 return "relu";
        case OpType::SILU:                 return "silu";
        case OpType::GELU:                 return "gelu";
        case OpType::SIGMOID:              return "sigmoid";
        case OpType::TANH:                 return "tanh";
        case OpType::CAST:                 return "cast";
        case OpType::ADD:                  return "add";
        case OpType::SUB:                  return "sub";
        case OpType::MUL:                  return "mul";
        case OpType::DIV:                  return "div";
        case OpType::POW:                  return "pow";
        case OpType::MAX_ELEM:             return "max_elem";
        case OpType::MIN_ELEM:             return "min_elem";
        case OpType::SUM:                  return "sum";
        case OpType::MEAN:                 return "mean";
        case OpType::MAX_REDUCE:           return "max_reduce";
        case OpType::MIN_REDUCE:           return "min_reduce";
        case OpType::RESHAPE:              return "reshape";
        case OpType::PERMUTE:              return "permute";
        case OpType::SLICE:                return "slice";
        case OpType::PAD:                  return "pad";
        case OpType::CONTIGUOUS:           return "contiguous";
        case OpType::BROADCAST_TO:         return "broadcast_to";
        case OpType::MATMUL:               return "matmul";
        case OpType::MATMUL_SCALED:        return "matmul_scaled";
        case OpType::LAYER_NORM:           return "layer_norm";
        case OpType::RMS_NORM:             return "rms_norm";
        case OpType::SOFTMAX:              return "softmax";
        case OpType::ROPE:                 return "rope";
        case OpType::EMBEDDING_LOOKUP:     return "embedding_lookup";
        case OpType::FUSED_MATMUL_BIAS:    return "fused_matmul_bias";
        case OpType::FUSED_MATMUL_BIAS_ACT:return "fused_matmul_bias_act";
        case OpType::FUSED_RMS_NORM_SCALE: return "fused_rms_norm_scale";
        default:                           return "custom";
    }
}

// ---- AttributeMap ----

void AttributeMap::set(std::string_view key, AttrValue value) {
    data_.insert_or_assign(std::string(key), std::move(value));
}

const AttrValue* AttributeMap::get(std::string_view key) const noexcept {
    auto it = data_.find(std::string(key));
    return (it != data_.end()) ? &it->second : nullptr;
}

// ---- ComputeGraph ----

TensorId ComputeGraph::add_tensor(TensorDesc desc) {
    TensorId id = static_cast<TensorId>(tensors_.size());
    tensors_.push_back(std::move(desc));
    topo_dirty_ = true;
    return id;
}

NodeId ComputeGraph::add_node(OpNode node) {
    NodeId id = static_cast<NodeId>(nodes_.size());
    node.id   = id;
    nodes_.push_back(std::move(node));
    topo_dirty_ = true;
    return id;
}

void ComputeGraph::set_inputs(std::vector<TensorId> ids)  { graph_inputs_  = std::move(ids); }
void ComputeGraph::set_outputs(std::vector<TensorId> ids) { graph_outputs_ = std::move(ids); }

const TensorDesc& ComputeGraph::tensor_desc(TensorId id) const {
    return tensors_.at(id);
}
const OpNode& ComputeGraph::node(NodeId id) const {
    return nodes_.at(id);
}
const std::vector<OpNode>&    ComputeGraph::nodes()   const noexcept { return nodes_; }
const std::vector<TensorId>&  ComputeGraph::inputs()  const noexcept { return graph_inputs_; }
const std::vector<TensorId>&  ComputeGraph::outputs() const noexcept { return graph_outputs_; }

void ComputeGraph::clear() {
    tensors_.clear();
    nodes_.clear();
    graph_inputs_.clear();
    graph_outputs_.clear();
    topo_cache_.clear();
    topo_dirty_ = true;
}

const std::vector<NodeId>& ComputeGraph::topo_order() const {
    if (topo_dirty_) compute_topo_order();
    return topo_cache_;
}

void ComputeGraph::compute_topo_order() const {
    // Kahn's algorithm (BFS-based topological sort).
    // Build: tensor_id → set of nodes that produce it.
    std::unordered_map<TensorId, NodeId> tensor_producer; // tensor → producing node
    for (const auto& n : nodes_) {
        for (TensorId out : n.outputs) {
            tensor_producer[out] = n.id;
        }
    }

    // Compute in-degrees per node.
    std::vector<int> in_degree(nodes_.size(), 0);
    for (const auto& n : nodes_) {
        for (TensorId inp : n.inputs) {
            auto it = tensor_producer.find(inp);
            if (it != tensor_producer.end()) {
                in_degree[n.id]++;
            }
        }
    }

    std::queue<NodeId> q;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (in_degree[i] == 0) q.push(static_cast<NodeId>(i));
    }

    topo_cache_.clear();
    topo_cache_.reserve(nodes_.size());

    while (!q.empty()) {
        NodeId id = q.front(); q.pop();
        topo_cache_.push_back(id);
        for (TensorId out : nodes_[id].outputs) {
            for (const auto& n2 : nodes_) {
                for (TensorId inp : n2.inputs) {
                    if (inp == out) {
                        if (--in_degree[n2.id] == 0) q.push(n2.id);
                    }
                }
            }
        }
    }

    topo_dirty_ = false;
}

ComputeGraph ComputeGraph::instantiate(
    const std::unordered_map<std::string, int64_t>& bindings) const {
    ComputeGraph out;
    // Resolve symbolic dims to concrete values.
    for (const auto& td : tensors_) {
        TensorDesc resolved;
        resolved.name  = td.name;
        resolved.dtype = td.dtype;
        for (const auto& d : td.shape) {
            if (std::holds_alternative<int64_t>(d)) {
                resolved.shape.push_back(d);
            } else {
                const auto& sym = std::get<std::string>(d);
                auto it = bindings.find(sym);
                if (it == bindings.end()) {
                    throw std::runtime_error("ComputeGraph::instantiate: unbound symbol '" + sym + "'");
                }
                resolved.shape.push_back(it->second);
            }
        }
        out.add_tensor(std::move(resolved));
    }
    for (const auto& n : nodes_) out.add_node(n);
    out.set_inputs(graph_inputs_);
    out.set_outputs(graph_outputs_);
    return out;
}

// ---- GraphBuilder ----

GraphBuilder::GraphBuilder(ComputeGraph& graph) : graph_(graph) {}

TensorId GraphBuilder::input(SymbolicShape shape, DType dtype, std::string name) {
    TensorDesc td;
    td.shape = std::move(shape);
    td.dtype = dtype;
    td.name  = std::move(name);
    TensorId id = graph_.add_tensor(std::move(td));
    auto inputs = graph_.inputs();
    inputs.push_back(id);
    graph_.set_inputs(std::move(inputs));
    return id;
}

TensorId GraphBuilder::emit_unary(OpType op, TensorId x,
                                   AttributeMap attrs, std::string name) {
    TensorDesc out_desc = graph_.tensor_desc(x);
    out_desc.name = name.empty() ? std::string(op_type_name(op)) : name;
    TensorId out_id = graph_.add_tensor(std::move(out_desc));
    OpNode node;
    node.op      = op;
    node.inputs  = {x};
    node.outputs = {out_id};
    node.attrs   = std::move(attrs);
    node.name    = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::emit_binary(OpType op, TensorId a, TensorId b,
                                    AttributeMap attrs, std::string name) {
    TensorDesc out_desc = graph_.tensor_desc(a); // output shape matches a (broadcast handled at runtime)
    out_desc.name = name.empty() ? std::string(op_type_name(op)) : name;
    TensorId out_id = graph_.add_tensor(std::move(out_desc));
    OpNode node;
    node.op      = op;
    node.inputs  = {a, b};
    node.outputs = {out_id};
    node.attrs   = std::move(attrs);
    node.name    = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::matmul(TensorId a, TensorId b, std::string name) {
    return emit_binary(OpType::MATMUL, a, b, {}, std::move(name));
}
TensorId GraphBuilder::add(TensorId a, TensorId b, std::string name) {
    return emit_binary(OpType::ADD, a, b, {}, std::move(name));
}
TensorId GraphBuilder::mul(TensorId a, TensorId b, std::string name) {
    return emit_binary(OpType::MUL, a, b, {}, std::move(name));
}
TensorId GraphBuilder::silu(TensorId x, std::string name) {
    return emit_unary(OpType::SILU, x, {}, std::move(name));
}

TensorId GraphBuilder::rms_norm(TensorId x, TensorId weight, float eps, std::string name) {
    AttributeMap attrs;
    attrs.set("eps", eps);
    TensorDesc out_desc = graph_.tensor_desc(x);
    out_desc.name = name.empty() ? "rms_norm" : name;
    TensorId out_id = graph_.add_tensor(std::move(out_desc));
    OpNode node;
    node.op = OpType::RMS_NORM;
    node.inputs = {x, weight};
    node.outputs = {out_id};
    node.attrs = std::move(attrs);
    node.name  = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::layer_norm(TensorId x, TensorId w, TensorId b, float eps, std::string name) {
    AttributeMap attrs;
    attrs.set("eps", eps);
    TensorDesc out_desc = graph_.tensor_desc(x);
    out_desc.name = name.empty() ? "layer_norm" : name;
    TensorId out_id = graph_.add_tensor(std::move(out_desc));
    OpNode node;
    node.op = OpType::LAYER_NORM;
    node.inputs  = {x, w, b};
    node.outputs = {out_id};
    node.attrs   = std::move(attrs);
    node.name    = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::softmax(TensorId x, int axis, std::string name) {
    AttributeMap attrs;
    attrs.set("axis", static_cast<int64_t>(axis));
    return emit_unary(OpType::SOFTMAX, x, std::move(attrs), std::move(name));
}

TensorId GraphBuilder::rope(TensorId x, TensorId freqs, int head_dim, std::string name) {
    AttributeMap attrs;
    attrs.set("head_dim", static_cast<int64_t>(head_dim));
    TensorDesc out_desc = graph_.tensor_desc(x);
    out_desc.name = name.empty() ? "rope" : name;
    TensorId out_id = graph_.add_tensor(std::move(out_desc));
    OpNode node;
    node.op = OpType::ROPE;
    node.inputs  = {x, freqs};
    node.outputs = {out_id};
    node.attrs   = std::move(attrs);
    node.name    = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::embedding(TensorId indices, TensorId weight, std::string name) {
    return emit_binary(OpType::EMBEDDING_LOOKUP, indices, weight, {}, std::move(name));
}

TensorId GraphBuilder::reshape(TensorId x, SymbolicShape new_shape, std::string name) {
    AttributeMap attrs;
    TensorDesc out_desc;
    out_desc.shape = std::move(new_shape);
    out_desc.dtype = graph_.tensor_desc(x).dtype;
    out_desc.name  = name.empty() ? "reshape" : name;
    TensorId out_id = graph_.add_tensor(std::move(out_desc));
    OpNode node;
    node.op = OpType::RESHAPE;
    node.inputs  = {x};
    node.outputs = {out_id};
    node.attrs   = std::move(attrs);
    node.name    = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::permute(TensorId x, std::vector<int> perm, std::string name) {
    const auto& src = graph_.tensor_desc(x);
    TensorDesc out_desc;
    out_desc.dtype = src.dtype;
    out_desc.name  = name.empty() ? "permute" : name;
    for (int p : perm) out_desc.shape.push_back(src.shape.at(p));
    TensorId out_id = graph_.add_tensor(std::move(out_desc));

    AttributeMap attrs;
    std::vector<int64_t> perm64(perm.begin(), perm.end());
    attrs.set("perm", perm64);
    OpNode node;
    node.op = OpType::PERMUTE;
    node.inputs  = {x};
    node.outputs = {out_id};
    node.attrs   = std::move(attrs);
    node.name    = name;
    graph_.add_node(std::move(node));
    return out_id;
}

TensorId GraphBuilder::slice(TensorId x, int dim, int64_t start, int64_t end, std::string name) {
    AttributeMap attrs;
    attrs.set("dim",   static_cast<int64_t>(dim));
    attrs.set("start", start);
    attrs.set("end",   end);
    return emit_unary(OpType::SLICE, x, std::move(attrs), std::move(name));
}

TensorId GraphBuilder::cast(TensorId x, DType target_dtype, std::string name) {
    AttributeMap attrs;
    attrs.set("target_dtype", static_cast<int64_t>(static_cast<uint8_t>(target_dtype)));
    return emit_unary(OpType::CAST, x, std::move(attrs), std::move(name));
}

void GraphBuilder::output(TensorId id) {
    auto outputs = graph_.outputs();
    outputs.push_back(id);
    graph_.set_outputs(std::move(outputs));
}

} // namespace axonforge
