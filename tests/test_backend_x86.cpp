#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "axonforge/backend.hpp"
#include "axonforge/tensor.hpp"
#include <cmath>
#include <cstring>

using namespace axonforge;

// ---- BackendRegistry ----

TEST_CASE("BackendRegistry has cpu_x86 registered", "[backend][registry]") {
    auto& reg = BackendRegistry::instance();
    REQUIRE(reg.has_backend("cpu_x86"));
    REQUIRE(reg.has_backend("cpu"));
}

TEST_CASE("BackendRegistry create returns non-null backend", "[backend][registry]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    REQUIRE(b != nullptr);
    REQUIRE(b->name() == "cpu_x86");
    REQUIRE(b->device_id().is_cpu());
}

TEST_CASE("BackendRegistry throws on unknown backend", "[backend][registry]") {
    REQUIRE_THROWS_AS(
        BackendRegistry::instance().create("nonexistent_backend"),
        std::runtime_error);
}

TEST_CASE("BackendRegistry available_backends is non-empty", "[backend][registry]") {
    auto ids = BackendRegistry::instance().available_backends();
    REQUIRE(!ids.empty());
}

// ---- x86 Backend lifecycle ----

TEST_CASE("X86Backend initialize/finalize succeeds", "[backend][x86]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    REQUIRE(b->initialize());
    b->finalize();  // must not throw
}

// ---- Memory allocation ----

TEST_CASE("X86Backend allocate returns valid storage", "[backend][x86][memory]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    b->initialize();

    auto storage = b->allocate(1024, 64);
    REQUIRE(storage != nullptr);
    REQUIRE(storage->byte_size() == 1024);
    REQUIRE(storage->data()      != nullptr);
    REQUIRE(storage->device().is_cpu());

    // Check 64-byte alignment
    REQUIRE(reinterpret_cast<uintptr_t>(storage->data()) % 64 == 0);
}

TEST_CASE("X86Backend allocate zero bytes", "[backend][x86][memory]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    b->initialize();
    // Zero-byte alloc should not crash
    REQUIRE_NOTHROW(b->allocate(0, 64));
}

// ---- can_handle ----

TEST_CASE("X86Backend can_handle common ops", "[backend][x86][dispatch]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    b->initialize();

    std::vector<DType> f32_inputs = {DType::F32, DType::F32};
    REQUIRE(b->can_handle(OpType::MATMUL,   f32_inputs));
    REQUIRE(b->can_handle(OpType::ADD,      f32_inputs));
    REQUIRE(b->can_handle(OpType::SOFTMAX,  f32_inputs));
    REQUIRE(b->can_handle(OpType::RMS_NORM, f32_inputs));
    REQUIRE(b->can_handle(OpType::SILU,     f32_inputs));
    REQUIRE(b->can_handle(OpType::RESHAPE,  f32_inputs));
}

// ---- Kernel execution (scalar reference path) ----

TEST_CASE("RMSNorm kernel produces correct output", "[backend][x86][kernel]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    REQUIRE(b->initialize());

    const int64_t N = 8;
    auto x = Tensor::zeros({N}, DType::F32);
    auto w = Tensor::zeros({N}, DType::F32);
    auto y = Tensor::zeros({N}, DType::F32);

    // x = [1, 2, 3, 4, 5, 6, 7, 8],  w = all 1s
    auto xd = x.data_as<float>();
    auto wd = w.data_as<float>();
    for (int64_t i = 0; i < N; ++i) { xd[i] = static_cast<float>(i + 1); wd[i] = 1.0f; }

    std::vector<DType> dtypes = {DType::F32, DType::F32};
    AttributeMap attrs;
    attrs.set("eps", 1e-5f);

    auto kernel = b->get_kernel(OpType::RMS_NORM, dtypes, attrs);
    REQUIRE(kernel != nullptr);

    std::vector<Tensor> inputs  = {x, w};
    std::vector<Tensor> outputs = {y};
    kernel->execute(inputs, outputs, attrs);

    // RMS of [1..8]: sqrt((1+4+9+16+25+36+49+64)/8) = sqrt(204/8) = sqrt(25.5)
    const float rms     = std::sqrt(204.f / 8.f + 1e-5f);
    auto        yd      = y.data_as<float>();
    for (int64_t i = 0; i < N; ++i) {
        float expected = static_cast<float>(i + 1) / rms;
        REQUIRE_THAT(yd[i], Catch::Matchers::WithinRel(expected, 1e-4f));
    }
}

TEST_CASE("Softmax kernel output sums to 1", "[backend][x86][kernel]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    REQUIRE(b->initialize());

    const int64_t N = 16;
    auto logits = Tensor::zeros({N}, DType::F32);
    auto probs  = Tensor::zeros({N}, DType::F32);
    auto ld = logits.data_as<float>();
    for (int64_t i = 0; i < N; ++i) ld[i] = static_cast<float>(i);

    std::vector<DType> dtypes = {DType::F32};
    AttributeMap attrs;
    auto kernel = b->get_kernel(OpType::SOFTMAX, dtypes, attrs);

    std::vector<Tensor> inputs  = {logits};
    std::vector<Tensor> outputs = {probs};
    kernel->execute(inputs, outputs, attrs);

    float sum = 0.f;
    for (float v : probs.data_as<float>()) sum += v;
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.f, 1e-5f));
}

TEST_CASE("GEMM F32 scalar reference path correct", "[backend][x86][kernel]") {
    auto b = BackendRegistry::instance().create("cpu_x86");
    REQUIRE(b->initialize());

    // A [2,3] × B [3,2] = C [2,2]
    auto A = Tensor::zeros({2, 3}, DType::F32);
    auto B = Tensor::zeros({3, 2}, DType::F32);
    auto C = Tensor::zeros({2, 2}, DType::F32);

    // A = [[1,2,3],[4,5,6]]   B = [[1,2],[3,4],[5,6]]
    // C = [[22,28],[49,64]]
    float a[] = {1,2,3,4,5,6};
    float b2[] = {1,2,3,4,5,6};
    std::memcpy(A.raw_data(), a,  sizeof(a));
    std::memcpy(B.raw_data(), b2, sizeof(b2));

    std::vector<DType> dtypes = {DType::F32, DType::F32};
    AttributeMap attrs;
    auto kernel = b->get_kernel(OpType::MATMUL, dtypes, attrs);

    std::vector<Tensor> inputs  = {A, B};
    std::vector<Tensor> outputs = {C};
    kernel->execute(inputs, outputs, attrs);

    auto cd = C.data_as<float>();
    REQUIRE_THAT(cd[0], Catch::Matchers::WithinAbs(22.f, 1e-4f));
    REQUIRE_THAT(cd[1], Catch::Matchers::WithinAbs(28.f, 1e-4f));
    REQUIRE_THAT(cd[2], Catch::Matchers::WithinAbs(49.f, 1e-4f));
    REQUIRE_THAT(cd[3], Catch::Matchers::WithinAbs(64.f, 1e-4f));
}
