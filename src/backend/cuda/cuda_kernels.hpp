#pragma once
#include "axonforge/tensor.hpp"
#include <cstddef>
#include <cstdint>

namespace axonforge::cuda {

void rms_norm_f32(float* y, const float* x, const float* w, int n, float eps, void* stream);
void silu_f32(float* y, const float* x, int n, void* stream);
void mul_f32(float* y, const float* a, const float* b, int n, void* stream);
void rope_f32(float* q, int n_heads, int head_dim, int pos, float theta, void* stream);

void q4km_gemv_f32(float* y,
                   const uint8_t* w,
                   const float* x,
                   int rows,
                   int cols,
                   size_t row_bytes,
                   void* stream);

} // namespace axonforge::cuda

