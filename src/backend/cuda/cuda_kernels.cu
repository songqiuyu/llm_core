#include "cuda_kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace axonforge::cuda {
namespace {

constexpr int QK_K = 256;
constexpr int Q4K_BYTES = 144;

__global__ void rms_norm_f32_kernel(float* y, const float* x, const float* w,
                                    int n, float eps) {
    extern __shared__ float ssum[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(ssum[0] / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = x[i] * inv * w[i];
    }
}

__global__ void silu_f32_kernel(float* y, const float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = v / (1.0f + expf(-v));
}

__global__ void mul_f32_kernel(float* y, const float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * b[i];
}

__global__ void rope_f32_kernel(float* q, int n_heads, int head_dim, int pos, float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = head_dim / 2;
    const int total = n_heads * half;
    if (i >= total) return;
    const int h = i / half;
    const int j = i - h * half;
    const int base = h * head_dim + j;
    const float freq = powf(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    const float a = static_cast<float>(pos) * freq;
    const float c = cosf(a);
    const float s = sinf(a);
    const float x0 = q[base];
    const float x1 = q[base + half];
    q[base]        = x0 * c - x1 * s;
    q[base + half] = x0 * s + x1 * c;
}

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// GGML Q4_K layout, enough for Q4_K_M rows used by current CPU kernels:
// d, dmin, scales[12], qs[128] per 256 values. This is a correctness-first
// scalar row kernel; v2 should tile rows/cols and dequantize through shared memory.
__global__ void q4km_gemv_f32_kernel(float* y, const uint8_t* w, const float* x,
                                     int rows, int cols, size_t row_bytes) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float partial[];
    float acc = 0.0f;
    const uint8_t* rp = w + static_cast<size_t>(row) * row_bytes;
    const int nb = cols / QK_K;

    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * Q4K_BYTES;
        const float d = fp16_to_f32(*reinterpret_cast<const uint16_t*>(blk + 0));
        const float dmin = fp16_to_f32(*reinterpret_cast<const uint16_t*>(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        const int xbase = b * QK_K;

        for (int i = threadIdx.x; i < QK_K; i += blockDim.x) {
            const int g = i / 32;
            const int si = g < 4 ? g : g + 4;
            const int mi = g < 4 ? g + 4 : g + 8;
            const float scale = static_cast<float>(scales[si] & 0x3f);
            const float minv = static_cast<float>(scales[mi] & 0x3f);
            const uint8_t qbyte = qs[i >> 1];
            const int qv = (i & 1) ? (qbyte >> 4) : (qbyte & 0x0f);
            const float wv = d * scale * static_cast<float>(qv) - dmin * minv;
            acc += wv * x[xbase + i];
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = partial[0];
}

} // namespace

void rms_norm_f32(float* y, const float* x, const float* w, int n, float eps, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    rms_norm_f32_kernel<<<1, 256, 256 * sizeof(float), s>>>(y, x, w, n, eps);
}

void silu_f32(float* y, const float* x, int n, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    silu_f32_kernel<<<(n + block - 1) / block, block, 0, s>>>(y, x, n);
}

void mul_f32(float* y, const float* a, const float* b, int n, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    mul_f32_kernel<<<(n + block - 1) / block, block, 0, s>>>(y, a, b, n);
}

void rope_f32(float* q, int n_heads, int head_dim, int pos, float theta, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int total = n_heads * (head_dim / 2);
    const int block = 256;
    rope_f32_kernel<<<(total + block - 1) / block, block, 0, s>>>(q, n_heads, head_dim, pos, theta);
}

void q4km_gemv_f32(float* y, const uint8_t* w, const float* x,
                   int rows, int cols, size_t row_bytes, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    q4km_gemv_f32_kernel<<<rows, 256, 256 * sizeof(float), s>>>(y, w, x, rows, cols, row_bytes);
}

} // namespace axonforge::cuda
