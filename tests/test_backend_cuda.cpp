#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "axonforge/backend.hpp"
#include "axonforge/tensor.hpp"
#include <array>
#include <cmath>
#include <vector>

using namespace axonforge;
using Catch::Matchers::WithinRel;

TEST_CASE("CUDA backend registers and copies tensors", "[backend][cuda]") {
    auto& reg = BackendRegistry::instance();
    REQUIRE(reg.has_backend("cuda"));

    auto b = reg.create("cuda");
    REQUIRE(b->initialize());
    REQUIRE(b->device_id().is_cuda());

    auto h = Tensor::zeros({4}, DType::F32);
    auto hd = h.data_as<float>();
    for (int i = 0; i < 4; ++i) hd[i] = static_cast<float>(i + 1);

    auto d_storage = b->allocate(h.nbytes(), 256);
    auto d = Tensor::from_storage(d_storage, {4}, DType::F32);
    b->transfer(h, d);

    auto out = Tensor::zeros({4}, DType::F32);
    b->transfer(d, out);
    auto od = out.data_as<float>();
    for (int i = 0; i < 4; ++i) REQUIRE(od[i] == hd[i]);

    b->finalize();
}

TEST_CASE("CUDA RMSNorm kernel matches scalar reference", "[backend][cuda][kernel]") {
    auto b = BackendRegistry::instance().create("cuda");
    REQUIRE(b->initialize());

    constexpr int N = 8;
    auto xh = Tensor::zeros({N}, DType::F32);
    auto wh = Tensor::zeros({N}, DType::F32);
    auto yh = Tensor::zeros({N}, DType::F32);
    auto xd = xh.data_as<float>();
    auto wd = wh.data_as<float>();
    for (int i = 0; i < N; ++i) {
        xd[i] = static_cast<float>(i + 1);
        wd[i] = 1.0f + 0.1f * static_cast<float>(i);
    }

    auto x = Tensor::from_storage(b->allocate(xh.nbytes(), 256), {N}, DType::F32);
    auto w = Tensor::from_storage(b->allocate(wh.nbytes(), 256), {N}, DType::F32);
    auto y = Tensor::from_storage(b->allocate(yh.nbytes(), 256), {N}, DType::F32);
    b->transfer(xh, x);
    b->transfer(wh, w);

    std::vector<DType> dtypes = {DType::F32, DType::F32};
    AttributeMap attrs;
    attrs.set("eps", 1e-5f);
    auto k = b->get_kernel(OpType::RMS_NORM, dtypes, attrs);
    std::array<Tensor, 2> inputs = {x, w};
    std::array<Tensor, 1> outputs = {y};
    k->execute(inputs, outputs, attrs);
    b->synchronize();
    b->transfer(y, yh);

    float ss = 0.0f;
    for (float v : xd) ss += v * v;
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(N) + 1e-5f);
    auto yd = yh.data_as<float>();
    for (int i = 0; i < N; ++i) {
        REQUIRE_THAT(yd[i], WithinRel(xd[i] * inv * wd[i], 1e-5f));
    }

    b->finalize();
}

