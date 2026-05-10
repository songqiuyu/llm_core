#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "axonforge/tensor.hpp"

using namespace axonforge;

// ---- Construction ----

TEST_CASE("Tensor::empty allocates correct size", "[tensor]") {
    auto t = Tensor::empty({4, 8}, DType::F32);
    REQUIRE(t.is_valid());
    REQUIRE(t.ndim()   == 2);
    REQUIRE(t.dim(0)   == 4);
    REQUIRE(t.dim(1)   == 8);
    REQUIRE(t.numel()  == 32);
    REQUIRE(t.nbytes() == 32 * sizeof(float));
    REQUIRE(t.dtype()  == DType::F32);
    REQUIRE(t.device().is_cpu());
}

TEST_CASE("Tensor::zeros initialises to 0", "[tensor]") {
    auto t = Tensor::zeros({16}, DType::F32);
    auto data = t.data_as<float>();
    for (float v : data) {
        REQUIRE_THAT(v, Catch::Matchers::WithinAbs(0.f, 1e-9f));
    }
}

TEST_CASE("Tensor 1D default is contiguous", "[tensor]") {
    auto t = Tensor::empty({100}, DType::F32);
    REQUIRE(t.is_contiguous());
}

TEST_CASE("Tensor 3D is contiguous", "[tensor]") {
    auto t = Tensor::empty({2, 3, 4}, DType::F32);
    REQUIRE(t.is_contiguous());
    REQUIRE(t.strides()[0] == 12);
    REQUIRE(t.strides()[1] == 4);
    REQUIRE(t.strides()[2] == 1);
}

// ---- View operations ----

TEST_CASE("Tensor::reshape produces correct shape", "[tensor][view]") {
    auto t  = Tensor::zeros({4, 8}, DType::F32);
    auto r  = t.reshape({2, 4, 4});
    REQUIRE(r.ndim()  == 3);
    REQUIRE(r.numel() == 32);
    // Same underlying storage
    REQUIRE(r.storage().get() == t.storage().get());
}

TEST_CASE("Tensor::permute transposes strides", "[tensor][view]") {
    auto t  = Tensor::empty({3, 5}, DType::F32);
    auto tp = t.permute({1, 0});
    REQUIRE(tp.dim(0) == 5);
    REQUIRE(tp.dim(1) == 3);
    REQUIRE(tp.strides()[0] == 1);   // original stride[1]
    REQUIRE(tp.strides()[1] == 5);   // original stride[0]
}

TEST_CASE("Tensor::slice correct shape", "[tensor][view]") {
    auto t = Tensor::empty({10, 4}, DType::F32);
    auto s = t.slice(0, 2, 7);
    REQUIRE(s.dim(0) == 5);
    REQUIRE(s.dim(1) == 4);
    REQUIRE(s.storage().get() == t.storage().get());
}

TEST_CASE("Tensor::squeeze removes dim-1 axis", "[tensor][view]") {
    auto t = Tensor::empty({1, 8, 1}, DType::F32);
    auto s = t.squeeze(0);
    REQUIRE(s.ndim() == 2);
    REQUIRE(s.dim(0) == 8);
}

TEST_CASE("Tensor::unsqueeze inserts dim", "[tensor][view]") {
    auto t = Tensor::empty({8}, DType::F32);
    auto u = t.unsqueeze(0);
    REQUIRE(u.ndim()  == 2);
    REQUIRE(u.dim(0) == 1);
    REQUIRE(u.dim(1) == 8);
}

TEST_CASE("Tensor::flatten reduces dims", "[tensor][view]") {
    auto t = Tensor::empty({2, 3, 4}, DType::F32);
    auto f = t.flatten(1, 2);   // flatten last two dims
    REQUIRE(f.ndim()  == 2);
    REQUIRE(f.dim(0) == 2);
    REQUIRE(f.dim(1) == 12);
}

// ---- data_as ----

TEST_CASE("Tensor data_as write/read round-trip", "[tensor]") {
    auto t = Tensor::zeros({4}, DType::F32);
    auto w = t.data_as<float>();
    w[0] = 1.f; w[1] = 2.f; w[2] = 3.f; w[3] = 4.f;

    auto r = t.data_as<const float>();
    REQUIRE_THAT(r[0], Catch::Matchers::WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(r[3], Catch::Matchers::WithinAbs(4.f, 1e-6f));
}

// ---- helpers ----

TEST_CASE("contiguous_strides for 3D tensor", "[tensor][helpers]") {
    Shape s   = {2, 3, 4};
    auto  st  = contiguous_strides(s);
    REQUIRE(st[0] == 12);
    REQUIRE(st[1] ==  4);
    REQUIRE(st[2] ==  1);
}

TEST_CASE("numel computes product of dims", "[tensor][helpers]") {
    REQUIRE(numel({3, 4, 5}) == 60);
    REQUIRE(numel({})        == 0);
    REQUIRE(numel({1})       == 1);
}

// ---- dtype helpers ----

TEST_CASE("dtype_element_size", "[dtype]") {
    REQUIRE(dtype_element_size(DType::F32) == 4);
    REQUIRE(dtype_element_size(DType::F16) == 2);
    REQUIRE(dtype_element_size(DType::I8)  == 1);
}

TEST_CASE("dtype_is_float", "[dtype]") {
    REQUIRE( dtype_is_float(DType::F32));
    REQUIRE( dtype_is_float(DType::F16));
    REQUIRE(!dtype_is_float(DType::I8));
    REQUIRE(!dtype_is_float(DType::Q4_K_M));
}

TEST_CASE("dtype_is_quantized", "[dtype]") {
    REQUIRE(!dtype_is_quantized(DType::F32));
    REQUIRE( dtype_is_quantized(DType::Q4_K_M));
    REQUIRE( dtype_is_quantized(DType::Q8_0));
}
