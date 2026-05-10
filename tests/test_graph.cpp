#include <catch2/catch_test_macros.hpp>
#include "axonforge/graph.hpp"

using namespace axonforge;

// ---- AttributeMap ----

TEST_CASE("AttributeMap set/get round-trip", "[graph][attrs]") {
    AttributeMap m;
    m.set("eps",   1e-5f);
    m.set("axis",  int64_t(-1));
    m.set("name",  std::string("layer0"));

    REQUIRE(m.get("eps")   != nullptr);
    REQUIRE(m.get("axis")  != nullptr);
    REQUIRE(m.get("missing") == nullptr);

    REQUIRE(m.get_or<float>("eps", 0.f) == 1e-5f);
    REQUIRE(m.get_or<int64_t>("axis", 0) == -1);
    REQUIRE(m.get_or<std::string>("name", "") == "layer0");
    REQUIRE(m.get_or<float>("missing", 42.f) == 42.f);
}

// ---- ComputeGraph ----

TEST_CASE("ComputeGraph add tensor/node and lookup", "[graph]") {
    ComputeGraph g;

    TensorDesc td1;
    td1.shape = {int64_t(4), std::string("seq")};
    td1.dtype = DType::F32;
    td1.name  = "x";
    TensorId t1 = g.add_tensor(td1);

    TensorDesc td2;
    td2.shape = {int64_t(4), std::string("seq")};
    td2.dtype = DType::F32;
    td2.name  = "y";
    TensorId t2 = g.add_tensor(td2);

    OpNode n;
    n.op      = OpType::ADD;
    n.inputs  = {t1};
    n.outputs = {t2};
    NodeId nid = g.add_node(n);

    REQUIRE(g.tensor_desc(t1).name == "x");
    REQUIRE(g.tensor_desc(t2).name == "y");
    REQUIRE(g.node(nid).op == OpType::ADD);
    REQUIRE(g.nodes().size() == 1);
}

TEST_CASE("ComputeGraph topological order single chain", "[graph][topo]") {
    // Build:  a → add → b → silu → c
    ComputeGraph g;
    TensorDesc td; td.dtype = DType::F32; td.shape = {int64_t(8)};

    td.name = "a"; TensorId a = g.add_tensor(td);
    td.name = "b"; TensorId b = g.add_tensor(td);
    td.name = "c"; TensorId c = g.add_tensor(td);

    OpNode add_node; add_node.op = OpType::ADD;
    add_node.inputs = {a}; add_node.outputs = {b};
    NodeId n0 = g.add_node(add_node);

    OpNode silu_node; silu_node.op = OpType::SILU;
    silu_node.inputs = {b}; silu_node.outputs = {c};
    NodeId n1 = g.add_node(silu_node);

    g.set_inputs({a});
    g.set_outputs({c});

    const auto& order = g.topo_order();
    REQUIRE(order.size() == 2);
    // ADD must come before SILU
    auto pos_n0 = std::find(order.begin(), order.end(), n0);
    auto pos_n1 = std::find(order.begin(), order.end(), n1);
    REQUIRE(pos_n0 < pos_n1);
}

TEST_CASE("ComputeGraph::instantiate resolves symbolic dims", "[graph][symbolic]") {
    ComputeGraph g;
    TensorDesc td;
    td.dtype = DType::F32;
    td.shape = {std::string("batch"), std::string("seq"), int64_t(128)};
    td.name  = "hidden";
    TensorId id = g.add_tensor(td);

    auto resolved = g.instantiate({{"batch", 1}, {"seq", 512}});

    const auto& s = resolved.tensor_desc(id).shape;
    REQUIRE(std::get<int64_t>(s[0]) == 1);
    REQUIRE(std::get<int64_t>(s[1]) == 512);
    REQUIRE(std::get<int64_t>(s[2]) == 128);
}

TEST_CASE("ComputeGraph::instantiate throws on unbound symbol", "[graph][symbolic]") {
    ComputeGraph g;
    TensorDesc td;
    td.dtype = DType::F32;
    td.shape = {std::string("unbound_dim")};
    g.add_tensor(td);

    REQUIRE_THROWS_AS(g.instantiate({}), std::runtime_error);
}

// ---- GraphBuilder ----

TEST_CASE("GraphBuilder builds a two-op graph", "[graph][builder]") {
    ComputeGraph g;
    GraphBuilder b(g);

    TensorId x = b.input({int64_t(4), int64_t(8)}, DType::F32, "x");
    TensorId w = b.input({int64_t(8), int64_t(4)}, DType::F32, "w");
    TensorId y = b.matmul(x, w, "y");
    TensorId z = b.silu(y, "z");
    b.output(z);

    REQUIRE(g.nodes().size() == 2);
    REQUIRE(g.inputs().size() == 2);
    REQUIRE(g.outputs().size() == 1);
    REQUIRE(g.nodes()[0].op == OpType::MATMUL);
    REQUIRE(g.nodes()[1].op == OpType::SILU);
}

TEST_CASE("op_type_name returns non-empty strings", "[graph]") {
    REQUIRE(op_type_name(OpType::MATMUL)  == "matmul");
    REQUIRE(op_type_name(OpType::RMS_NORM) == "rms_norm");
    REQUIRE(op_type_name(OpType::ADD)     == "add");
    REQUIRE(op_type_name(OpType::SOFTMAX) == "softmax");
    REQUIRE(op_type_name(OpType::CUSTOM)  == "custom");
}
