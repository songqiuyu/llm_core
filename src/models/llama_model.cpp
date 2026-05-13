// ============================================================
// AxonForge — LLaMA-family inference (F16 / Q4_K_M / Q6_K weights)
//
// Covers: LLaMA-2, LLaMA-3, LLaMA-3.2, TinyLlama, Mistral
// Architecture: RMSNorm + RoPE + SwiGLU FFN + GQA attention
//
// Key optimisations:
//   • gemv_f16_dispatch  — AVX2+F16C GEMV via persistent ThreadPool
//   • kQ4kDotRow         — runtime-selected Q4K dot: AVX2 (default) or scalar
//   • q6k_dot_row        — scalar Q6_K row dot (on-the-fly dequant)
//   • LlamaWeights       — pre-built Tensor* pointer table
//   • RoPE cache         — cos/sin precomputed at model load
// ============================================================

#include "axonforge/engine.hpp"
#include "axonforge/models/llama.hpp"
#include "axonforge/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <immintrin.h>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef __linux__
#  include <sched.h>
#endif

// Persistent thread pool (header-only)
#include "../core/thread_pool.hpp"
// CPU feature detection
#include "../backend/cpu_x86/cpuid.hpp"
// Q8_0 block struct (shared with kernel files)
#include "../backend/cpu_x86/kernels/q8_block.hpp"

// ─── AVX2+F16C GEMV forward declaration ──────────────────────────────────────
namespace axonforge::cpu_x86 {
    void gemv_f16_avx2_range(float* y, const uint16_t* W, const float* x,
                             const float* b,
                             int row_start, int row_end, int in_dim) noexcept;
    // Q4K AVX2 kernel (gemv_q4k_avx2.cpp, compiled with -mavx2 -mf16c)
    float q4k_dot_row_avx2(const uint8_t* row, const float* x, int in_dim) noexcept;
    void  gemv_q4k_avx2_range(float* y, const uint8_t* W, const float* x,
                               int o_start, int o_end, int in_dim,
                               size_t row_bytes) noexcept;
    // Q6K AVX2 kernel (gemv_q6k_avx2.cpp, compiled with -mavx2 -mf16c)
    float q6k_dot_row_avx2(const uint8_t* row, const float* x, int in_dim) noexcept;
    void  gemv_q6k_avx2_range(float* y, const uint8_t* W, const float* x,
                               int o_start, int o_end, int in_dim,
                               size_t row_bytes) noexcept;
    // ── Q8_0 kernels (Phase 4) ────────────────────────────────────────────────
    void quantize_q8_avx2(block_q8_0* y, const float* x, int k) noexcept;
    void gemv_q4k_q8_avx2_range(float* y, const uint8_t* W,
                                 const block_q8_0* xq8,
                                 int o_start, int o_end, int in_dim,
                                 size_t row_bytes) noexcept;
    void gemm_q4k_q8_avx2_range(float* Y, const uint8_t* W,
                                 const block_q8_0* XQ8,
                                 int M, int T, int K, size_t row_bytes,
                                 int o_start, int o_end) noexcept;
    void gemv_q6k_q8_avx2_range(float* y, const uint8_t* W,
                                 const block_q8_0* xq8,
                                 int o_start, int o_end, int in_dim,
                                 size_t row_bytes) noexcept;
    // ── AVX-VNNI kernels (gemv_q4k_q8_avxvnni.cpp, -mavxvnni) ───────────────
    void gemv_q4k_q8_avxvnni_range(float* y, const uint8_t* W,
                                    const block_q8_0* xq8,
                                    int o_start, int o_end, int in_dim,
                                    size_t row_bytes) noexcept;
    void gemm_q4k_q8_avxvnni_range(float* Y, const uint8_t* W,
                                    const block_q8_0* XQ8,
                                    int M, int T, int K, size_t row_bytes,
                                    int o_start, int o_end) noexcept;
    // ── 8-row repacked Q4K kernel ──────────────────────────────────────────
    void repack_q4k_8rows(uint8_t* dst, const uint8_t* src,
                          int nrows, int ncols) noexcept;
    void gemv_q4k_r8_avx2_range(float* y, const uint8_t* R8,
                                 const block_q8_0* xq8,
                                 int o_start, int o_end, int in_dim) noexcept;
    // AVX-VNNI 8-row kernel (gemv_q4k_r8_avxvnni.cpp, -mavxvnni)
    void gemv_q4k_r8_avxvnni_range(float* y, const uint8_t* R8,
                                    const block_q8_0* xq8,
                                    int o_start, int o_end, int in_dim) noexcept;
}

namespace axonforge {

// ─── F16 bit-cast helpers ─────────────────────────────────────────────────────

static inline float f16_to_f32(uint16_t h) noexcept {
    uint32_t sign     = (h >> 15) & 1u;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa =  h        & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) { bits = sign << 31; }
        else {
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

// ─── GEMV dispatch (reuse AVX2 kernel from gpt2) ─────────────────────────────

using GemvRangeFn = void(*)(float*, const uint16_t*, const float*, const float*,
                             int, int, int) noexcept;

static void gemv_f16_scalar_range(float* y, const uint16_t* W, const float* x,
                                   const float* b,
                                   int row_start, int row_end, int in_dim) noexcept {
    for (int o = row_start; o < row_end; o++) {
        float acc = b ? b[o] : 0.f;
        const uint16_t* row = W + (size_t)o * in_dim;
        for (int k = 0; k < in_dim; k++) acc += f16_to_f32(row[k]) * x[k];
        y[o] = acc;
    }
}

static const GemvRangeFn kLlamaGemvRange = cpu_x86::cpu_features().avx2
    ? cpu_x86::gemv_f16_avx2_range
    : gemv_f16_scalar_range;

static void gemv_f16_dispatch(float* y, const uint16_t* W, const float* x,
                               const float* b, int out, int in,
                               ThreadPool* pool) noexcept {
    if (pool && out >= 128) {
        pool->parallel_for(out, [&](int s, int e) {
            kLlamaGemvRange(y, W, x, b, s, e, in);
        });
    } else {
        kLlamaGemvRange(y, W, x, b, 0, out, in);
    }
}

static inline float q8_0_weight_dot_row(const uint8_t* row,
                                         const float* x,
                                         int in_dim) noexcept {
    float acc = 0.f;
    const int nb = (in_dim + 31) / 32;
    for (int b = 0; b < nb; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, row + (size_t)b * 34, 2);
        const float d = f16_to_f32(d_bits);
        const int8_t* q = reinterpret_cast<const int8_t*>(row + (size_t)b * 34 + 2);
        const int base = b * 32;
        const int lim = std::min(32, in_dim - base);
        for (int i = 0; i < lim; ++i)
            acc += d * static_cast<float>(q[i]) * x[base + i];
    }
    return acc;
}

static inline float q8_0_weight_dot_q8_row(const uint8_t* row,
                                            const block_q8_0* xq8,
                                            int in_dim) noexcept {
    float acc = 0.f;
    const int nb = in_dim / 32;
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    for (int b = 0; b < nb; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, row + (size_t)b * 34, 2);
        const float scale = f16_to_f32(d_bits) * xq8[b].d;
        const int8_t* q = reinterpret_cast<const int8_t*>(row + (size_t)b * 34 + 2);
#if defined(__AVX2__)
        const __m256i qw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq8[b].qs));
        const __m128i qw_lo_128 = _mm256_castsi256_si128(qw);
        const __m128i qw_hi_128 = _mm256_extracti128_si256(qw, 1);
        const __m128i qx_lo_128 = _mm256_castsi256_si128(qx);
        const __m128i qx_hi_128 = _mm256_extracti128_si256(qx, 1);
        const __m256i qw_lo = _mm256_cvtepi8_epi16(qw_lo_128);
        const __m256i qw_hi = _mm256_cvtepi8_epi16(qw_hi_128);
        const __m256i qx_lo = _mm256_cvtepi8_epi16(qx_lo_128);
        const __m256i qx_hi = _mm256_cvtepi8_epi16(qx_hi_128);
        const __m256i prod_lo = _mm256_mullo_epi16(qw_lo, qx_lo);
        const __m256i prod_hi = _mm256_mullo_epi16(qw_hi, qx_hi);
        const __m256i sum_lo = _mm256_madd_epi16(prod_lo, ones);
        const __m256i sum_hi = _mm256_madd_epi16(prod_hi, ones);
        __m256i sum = _mm256_add_epi32(sum_lo, sum_hi);
        __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                     _mm256_extracti128_si256(sum, 1));
        s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
        s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0xB1));
        const int isum = _mm_cvtsi128_si32(s128);
#else
        int isum = 0;
        for (int i = 0; i < 32; ++i)
            isum += static_cast<int>(q[i]) * static_cast<int>(xq8[b].qs[i]);
#endif
        acc += scale * static_cast<float>(isum);
    }
    return acc;
}

static void gemv_q8_0_weight_q8(float* y,
                                const uint8_t* W,
                                const block_q8_0* xq8,
                                int out,
                                int in,
                                ThreadPool* pool) noexcept {
    const size_t row_bytes = (size_t)(in / 32) * 34;
    if (pool && out >= 32) {
        pool->parallel_for(out, [&](int s, int e) {
            for (int o = s; o < e; ++o)
                y[o] = q8_0_weight_dot_q8_row(W + (size_t)o * row_bytes, xq8, in);
        });
    } else {
        for (int o = 0; o < out; ++o)
            y[o] = q8_0_weight_dot_q8_row(W + (size_t)o * row_bytes, xq8, in);
    }
}

static void q8_0_weight_row_to_f32(float* out,
                                   const uint8_t* row,
                                   int cols) noexcept {
    const int nb = (cols + 31) / 32;
    for (int b = 0; b < nb; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, row + (size_t)b * 34, 2);
        const float d = f16_to_f32(d_bits);
        const int8_t* q = reinterpret_cast<const int8_t*>(row + (size_t)b * 34 + 2);
        const int base = b * 32;
        const int lim = std::min(32, cols - base);
        for (int i = 0; i < lim; ++i)
            out[base + i] = d * static_cast<float>(q[i]);
    }
}

static inline float q5_0_dot_row(const uint8_t* row,
                                  const float* x,
                                  int in_dim) noexcept {
    float acc = 0.f;
    const int nb = (in_dim + 31) / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* block = row + (size_t)b * 22;
        uint16_t d_bits;
        uint32_t qh;
        std::memcpy(&d_bits, block, 2);
        std::memcpy(&qh, block + 2, 4);
        const float d = f16_to_f32(d_bits);
        const uint8_t* qs = block + 6;
        const int base = b * 32;
        const int lim = std::min(32, in_dim - base);
        for (int i = 0; i < lim; ++i) {
            const uint8_t nib = (i < 16)
                ? static_cast<uint8_t>(qs[i] & 0x0F)
                : static_cast<uint8_t>(qs[i - 16] >> 4);
            const uint8_t hi = static_cast<uint8_t>((qh >> i) & 1u);
            const float w = d * (static_cast<float>(nib | (hi << 4)) - 16.f);
            acc += w * x[base + i];
        }
    }
    return acc;
}

static void q5_0_row_to_f32(float* out,
                            const uint8_t* row,
                            int cols) noexcept {
    const int nb = (cols + 31) / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* block = row + (size_t)b * 22;
        uint16_t d_bits;
        uint32_t qh;
        std::memcpy(&d_bits, block, 2);
        std::memcpy(&qh, block + 2, 4);
        const float d = f16_to_f32(d_bits);
        const uint8_t* qs = block + 6;
        const int base = b * 32;
        const int lim = std::min(32, cols - base);
        for (int i = 0; i < lim; ++i) {
            const uint8_t nib = (i < 16)
                ? static_cast<uint8_t>(qs[i] & 0x0F)
                : static_cast<uint8_t>(qs[i - 16] >> 4);
            const uint8_t hi = static_cast<uint8_t>((qh >> i) & 1u);
            out[base + i] = d * (static_cast<float>(nib | (hi << 4)) - 16.f);
        }
    }
}

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

static void rms_norm(float* __restrict__ x,
                     const float* __restrict__ w,
                     int n, float eps) noexcept {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) x[i] = w[i] * x[i] * ss;
}

// ─── SiLU ────────────────────────────────────────────────────────────────────

static inline float silu(float x) noexcept {
    return x / (1.f + std::exp(-x));
}

// ── AVX2 exp(x) — Cephes single-precision polynomial, max error ~2^-23 ───────
static inline __m256 exp_avx2_(const __m256 x) noexcept {
    const __m256 clamp_hi = _mm256_set1_ps( 88.3762626648f);
    const __m256 clamp_lo = _mm256_set1_ps(-88.3762626648f);
    const __m256 inv_log2e = _mm256_set1_ps(1.44269504089f);
    const __m256 ln2_hi   = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo   = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 half     = _mm256_set1_ps(0.5f);
    const __m256 one      = _mm256_set1_ps(1.f);

    __m256 xx = _mm256_max_ps(clamp_lo, _mm256_min_ps(x, clamp_hi));
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(xx, inv_log2e, half));
    xx = _mm256_fnmadd_ps(fx, ln2_hi, xx);
    xx = _mm256_fnmadd_ps(fx, ln2_lo, xx);

    __m256 z = _mm256_mul_ps(xx, xx);
    __m256 p = _mm256_set1_ps(1.9875691500E-4f);
    p = _mm256_fmadd_ps(p, xx, _mm256_set1_ps(1.3981999507E-3f));
    p = _mm256_fmadd_ps(p, xx, _mm256_set1_ps(8.3334519073E-3f));
    p = _mm256_fmadd_ps(p, xx, _mm256_set1_ps(4.1665795894E-2f));
    p = _mm256_fmadd_ps(p, xx, _mm256_set1_ps(1.6666665459E-1f));
    p = _mm256_fmadd_ps(p, xx, _mm256_set1_ps(5.0000001201E-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_add_ps(xx, one));

    __m256i n = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(n));
}

// ── AVX2 SwiGLU:  gate[i] = silu(gate[i]) * up[i] ───────────────────────────
// sigmoid(g) computed as: exp(-|g|)/(1+exp(-|g|)) or 1/(1+exp(-|g|))
static void swiglu_avx2(float* __restrict__ gate,
                         const float* __restrict__ up, int n) noexcept {
    const __m256 one     = _mm256_set1_ps(1.f);
    const __m256 neg_zero = _mm256_set1_ps(-0.f);
    const __m256 zero    = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 g       = _mm256_loadu_ps(gate + i);
        const __m256 u       = _mm256_loadu_ps(up   + i);
        const __m256 neg_abs = _mm256_or_ps(neg_zero, g);          // -|g|
        const __m256 e       = exp_avx2_(neg_abs);                  // exp(-|g|)
        const __m256 denom   = _mm256_add_ps(one, e);
        const __m256 sig_pos = _mm256_div_ps(one, denom);          // g >= 0
        const __m256 sig_neg = _mm256_div_ps(e,   denom);          // g <  0
        const __m256 mask    = _mm256_cmp_ps(g, zero, _CMP_LT_OQ);
        const __m256 sig     = _mm256_blendv_ps(sig_pos, sig_neg, mask);
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(_mm256_mul_ps(g, sig), u));
    }
    for (; i < n; i++) gate[i] = silu(gate[i]) * up[i];
}

// ── AVX2 RMSNorm ─────────────────────────────────────────────────────────────
static void rms_norm_avx2(float* __restrict__ x,
                            const float* __restrict__ w,
                            int n, float eps) noexcept {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = 0.f;
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    for (int k = 0; k < 8; k++) ss += tmp[k];
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / std::sqrt(ss / n + eps);
    const __m256 scale = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xi = _mm256_loadu_ps(x + i);
        const __m256 wi = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(wi, _mm256_mul_ps(xi, scale)));
    }
    for (; i < n; i++) x[i] = w[i] * x[i] * ss;
}

// ── AVX2 RMSNorm (2-input): reads from src, writes to dst — eliminates copy ──
static void rms_norm_out(float* __restrict__ dst,
                          const float* __restrict__ src,
                          const float* __restrict__ w,
                          int n, float eps) noexcept {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = 0.f;
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    for (int k = 0; k < 8; k++) ss += tmp[k];
    for (; i < n; i++) ss += src[i] * src[i];
    ss = 1.f / std::sqrt(ss / n + eps);
    const __m256 scale = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xi = _mm256_loadu_ps(src + i);
        const __m256 wi = _mm256_loadu_ps(w   + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(wi, _mm256_mul_ps(xi, scale)));
    }
    for (; i < n; i++) dst[i] = w[i] * src[i] * ss;
}

// ─── Softmax in-place ─────────────────────────────────────────────────────────

static void softmax(float* x, int n) noexcept {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ─── Q4_K_M on-the-fly dequantisation ────────────────────────────────────────
// Block layout (144 bytes, QK_K=256 elements):
//   [0:1]   d     (f16) super-block scale for quantised scales
//   [2:3]   dmin  (f16) super-block scale for quantised mins
//   [4:15]  scales[12]  8×(6-bit scale, 6-bit min) packed
//   [16:143] qs[128]    4-bit packed nibbles

static constexpr int QK_K = 256;
static constexpr int Q4K_BYTES = 144;
static constexpr int Q6K_BYTES = 210;

static inline void q4k_get_scale_min(int j, const uint8_t* sc,
                                      uint8_t& d, uint8_t& m) noexcept {
    if (j < 4) {
        d = sc[j]   & 0x3F;
        m = sc[j+4] & 0x3F;
    } else {
        d = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
        m = (sc[j+4] >>   4) | ((sc[j-0] >> 6) << 4);
    }
}

// Dot product of one Q4_K_M row with x[in_dim]
static float q4k_dot_row(const uint8_t* row, const float* x, int in_dim) noexcept {
    float acc = 0.f;
    const int nb = in_dim / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t* block  = row + (size_t)b * Q4K_BYTES;
        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits,    block,     2);
        std::memcpy(&dmin_bits, block + 2, 2);
        const float d    = f16_to_f32(d_bits);
        const float dmin = f16_to_f32(dmin_bits);
        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;
        const float*   xb     = x + (size_t)b * QK_K;
        for (int chunk = 0; chunk < 4; chunk++) {
            uint8_t sc1, m1, sc2, m2;
            q4k_get_scale_min(chunk*2 + 0, scales, sc1, m1);
            q4k_get_scale_min(chunk*2 + 1, scales, sc2, m2);
            const float d1 = d * sc1, mf1 = dmin * m1;
            const float d2 = d * sc2, mf2 = dmin * m2;
            const uint8_t* q = qs + chunk * 32;
            const float*   xi = xb + chunk * 64;
            for (int l = 0; l < 32; l++) {
                acc += (d1 * (q[l] & 0xF) - mf1) * xi[l];
                acc += (d2 * (q[l] >> 4 ) - mf2) * xi[l + 32];
            }
        }
    }
    return acc;
}

// ─── Q4K / Q6K runtime dispatch ───────────────────────────────────────────────
// Selected once at startup based on cpu_features().avx2;
// used by gemv_wt and any future model that calls it.
using Q4kDotRowFn = float(*)(const uint8_t*, const float*, int) noexcept;
static const Q4kDotRowFn kQ4kDotRow =
    cpu_x86::cpu_features().avx2
        ? cpu_x86::q4k_dot_row_avx2
        : q4k_dot_row;

static float q6k_dot_row(const uint8_t* row, const float* x, int in_dim) noexcept;

using Q6kDotRowFn = float(*)(const uint8_t*, const float*, int) noexcept;
static const Q6kDotRowFn kQ6kDotRow =
    cpu_x86::cpu_features().avx2
        ? cpu_x86::q6k_dot_row_avx2
        : q6k_dot_row;

// ─── Q4K / Q6K range-function dispatch (includes 4-row tiling when AVX2) ────
using Q4kRangeFn = void(*)(float*, const uint8_t*, const float*,
                            int, int, int, size_t) noexcept;
static void q4k_scalar_range(float* y, const uint8_t* W, const float* x,
                               int o0, int o1, int in_dim,
                               size_t row_bytes) noexcept {
    for (int o = o0; o < o1; o++)
        y[o] = q4k_dot_row(W + (size_t)o * row_bytes, x, in_dim);
}
static const Q4kRangeFn kQ4kRange =
    cpu_x86::cpu_features().avx2
        ? cpu_x86::gemv_q4k_avx2_range
        : q4k_scalar_range;

using Q6kRangeFn = void(*)(float*, const uint8_t*, const float*,
                            int, int, int, size_t) noexcept;
static void q6k_scalar_range(float* y, const uint8_t* W, const float* x,
                               int o0, int o1, int in_dim,
                               size_t row_bytes) noexcept {
    for (int o = o0; o < o1; o++)
        y[o] = q6k_dot_row(W + (size_t)o * row_bytes, x, in_dim);
}
static const Q6kRangeFn kQ6kRange =
    cpu_x86::cpu_features().avx2
        ? cpu_x86::gemv_q6k_avx2_range
        : q6k_scalar_range;

// ─── Q6_K on-the-fly dequantisation ──────────────────────────────────────────
// Block layout (210 bytes, QK_K=256 elements):
//   [0:127]   ql[128]   lower 4 bits of each 6-bit quant
//   [128:191] qh[64]    upper 2 bits of each 6-bit quant (2 bits × 256 = 64 bytes)
//   [192:207] scales[16] int8 scales for each group of 16
//   [208:209] d (f16)   super-block scale

// Q6_K dequantise one row (or block) using correct is=l/16 scale indexing.
// block_q6_K: ql[128] | qh[64] | scales[16 int8] | d[f16]
// Two outer iterations (n=0,1) each covering 128 elements via 32 inner steps.
static void q6k_row_to_f32(float* out, const uint8_t* row, int in_dim) noexcept {
    const int nb = in_dim / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t* block = row + (size_t)b * Q6K_BYTES;
        uint16_t d_bits;
        std::memcpy(&d_bits, block + 208, 2);
        const float d = f16_to_f32(d_bits);
        float* ob = out + (size_t)b * QK_K;

        const uint8_t* ql_ptr = block;
        const uint8_t* qh_ptr = block + 128;
        const int8_t*  sc_ptr = reinterpret_cast<const int8_t*>(block + 192);
        float* y = ob;

        for (int n = 0; n < 2; n++) {  // two halves of 128 elements
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;  // 0 for l<16, 1 for l>=16
                const int8_t q1 = (int8_t)(((ql_ptr[l     ] & 0x0F) | (((qh_ptr[l] >> 0) & 3) << 4)) - 32);
                const int8_t q2 = (int8_t)(((ql_ptr[l + 32] & 0x0F) | (((qh_ptr[l] >> 2) & 3) << 4)) - 32);
                const int8_t q3 = (int8_t)(((ql_ptr[l     ] >>  4 ) | (((qh_ptr[l] >> 4) & 3) << 4)) - 32);
                const int8_t q4 = (int8_t)(((ql_ptr[l + 32] >>  4 ) | (((qh_ptr[l] >> 6) & 3) << 4)) - 32);
                y[l +  0] = d * sc_ptr[is + 0] * q1;
                y[l + 32] = d * sc_ptr[is + 2] * q2;
                y[l + 64] = d * sc_ptr[is + 4] * q3;
                y[l + 96] = d * sc_ptr[is + 6] * q4;
            }
            ql_ptr += 64;
            qh_ptr += 32;
            sc_ptr += 8;
            y      += 128;
        }
    }
}

// Dot product of one Q6_K row with x[in_dim] (inline dequant)
static float q6k_dot_row(const uint8_t* row, const float* x, int in_dim) noexcept {
    float acc = 0.f;
    const int nb = in_dim / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t* block = row + (size_t)b * Q6K_BYTES;
        uint16_t d_bits;
        std::memcpy(&d_bits, block + 208, 2);
        const float d = f16_to_f32(d_bits);
        const float* xb = x + (size_t)b * QK_K;

        const uint8_t* ql_ptr = block;
        const uint8_t* qh_ptr = block + 128;
        const int8_t*  sc_ptr = reinterpret_cast<const int8_t*>(block + 192);
        const float*   xp     = xb;

        for (int n = 0; n < 2; n++) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)(((ql_ptr[l     ] & 0x0F) | (((qh_ptr[l] >> 0) & 3) << 4)) - 32);
                const int8_t q2 = (int8_t)(((ql_ptr[l + 32] & 0x0F) | (((qh_ptr[l] >> 2) & 3) << 4)) - 32);
                const int8_t q3 = (int8_t)(((ql_ptr[l     ] >>  4 ) | (((qh_ptr[l] >> 4) & 3) << 4)) - 32);
                const int8_t q4 = (int8_t)(((ql_ptr[l + 32] >>  4 ) | (((qh_ptr[l] >> 6) & 3) << 4)) - 32);
                acc += (d * sc_ptr[is + 0] * q1) * xp[l +  0];
                acc += (d * sc_ptr[is + 2] * q2) * xp[l + 32];
                acc += (d * sc_ptr[is + 4] * q3) * xp[l + 64];
                acc += (d * sc_ptr[is + 6] * q4) * xp[l + 96];
            }
            ql_ptr += 64;
            qh_ptr += 32;
            sc_ptr += 8;
            xp     += 128;
        }
    }
    return acc;
}

// ─── Weight pointer table (dtype-aware) ──────────────────────────────────────

static const Tensor* w_req(const Engine& e, const char* name) {
    const Tensor* t = e.weight(name);
    if (!t) throw std::runtime_error(std::string("LLaMA: missing weight ") + name);
    return t;
}
static const float* w_f32(const Engine& e, const char* name) {
    const Tensor* t = w_req(e, name);
    return static_cast<const float*>(t->raw_data());
}

// Dtype-agnostic weight descriptor
struct WT {
    const void* data{nullptr};
    DType       dtype{DType::UNKNOWN};
    bool        q8_fast{true};
    // For Q4K_M: 8-row interleaved repacked copy (same total bytes, sequential layout).
    // Stored as shared_ptr so LlamaWeights can be cheaply copied (O(1) refcount bump)
    // for the weight cache.  Non-null iff dtype==Q4_K_M and nrows%8==0.
    std::shared_ptr<std::vector<uint8_t>> r8;
};
static WT make_wt(const Engine& e, const char* name) {
    const Tensor* t = w_req(e, name);
    WT w;
    w.data = t->raw_data();
    w.dtype = t->dtype();
    return w;
}
static WT make_wt_opt(const Engine& e, const char* name, const WT& fallback) {
    const Tensor* t = e.weight(name);
    if (!t) return fallback;
    WT w;
    w.data = t->raw_data();
    w.dtype = t->dtype();
    return w;
}

// Dispatch GEMV over dtype: F16 uses AVX2 kernel, Q4_K_M uses scalar dequant
static void gemv_wt(float* y, const WT& W, const float* x, int out, int in,
                    ThreadPool* pool) noexcept {
    if (W.dtype == DType::F16) {
        const uint16_t* Wf = static_cast<const uint16_t*>(W.data);
        if (pool && out >= 128) {
            pool->parallel_for(out, [&](int s, int e) {
                kLlamaGemvRange(y, Wf, x, nullptr, s, e, in);
            });
        } else {
            kLlamaGemvRange(y, Wf, x, nullptr, 0, out, in);
        }
    } else if (W.dtype == DType::Q4_K_M) {
        const uint8_t* Wq = static_cast<const uint8_t*>(W.data);
        const size_t row_bytes = (in / QK_K) * Q4K_BYTES;
        if (pool && out >= 32) {
            pool->parallel_for(out, [&](int s, int e) {
                kQ4kRange(y, Wq, x, s, e, in, row_bytes);
            });
        } else {
            kQ4kRange(y, Wq, x, 0, out, in, row_bytes);
        }
    } else if (W.dtype == DType::Q6_K) {
        const uint8_t* Wq = static_cast<const uint8_t*>(W.data);
        const size_t row_bytes = (in / QK_K) * Q6K_BYTES;
        if (pool && out >= 32) {
            pool->parallel_for(out, [&](int s, int e) {
                kQ6kRange(y, Wq, x, s, e, in, row_bytes);
            });
        } else {
            kQ6kRange(y, Wq, x, 0, out, in, row_bytes);
        }
    } else if (W.dtype == DType::Q8_0) {
        const uint8_t* Wq = static_cast<const uint8_t*>(W.data);
        const size_t row_bytes = (size_t)((in + 31) / 32) * 34;
        if (pool && out >= 32) {
            pool->parallel_for(out, [&](int s, int e) {
                for (int o = s; o < e; ++o)
                    y[o] = q8_0_weight_dot_row(Wq + (size_t)o * row_bytes, x, in);
            });
        } else {
            for (int o = 0; o < out; ++o)
                y[o] = q8_0_weight_dot_row(Wq + (size_t)o * row_bytes, x, in);
        }
    } else if (W.dtype == DType::Q5_0) {
        const uint8_t* Wq = static_cast<const uint8_t*>(W.data);
        const size_t row_bytes = (size_t)((in + 31) / 32) * 22;
        if (pool && out >= 32) {
            pool->parallel_for(out, [&](int s, int e) {
                for (int o = s; o < e; ++o)
                    y[o] = q5_0_dot_row(Wq + (size_t)o * row_bytes, x, in);
            });
        } else {
            for (int o = 0; o < out; ++o)
                y[o] = q5_0_dot_row(Wq + (size_t)o * row_bytes, x, in);
        }
    } else {
        // Fallback: treat as F16 (BF16 etc.)
        const uint16_t* Wf = static_cast<const uint16_t*>(W.data);
        kLlamaGemvRange(y, Wf, x, nullptr, 0, out, in);
    }
}

// ─── Phase 4: Q8_0 quantisation dispatch ─────────────────────────────────────

static void kQ8Quant_scalar(block_q8_0* y, const float* x, int k) noexcept {
    for (int i = 0, nb = k / 32; i < nb; i++) {
        float amax = 0.f;
        for (int j = 0; j < 32; j++) {
            float v = x[i*32+j];
            if (v < 0) v = -v;
            if (v > amax) amax = v;
        }
        const float d  = amax / 127.f;
        const float id = (amax > 0.f) ? 127.f / amax : 0.f;
        y[i].d = d;
        for (int j = 0; j < 32; j++) {
            int v = (int)(x[i*32+j] * id + 0.5f);
            y[i].qs[j] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
    }
}

using QuantQ8Fn = void(*)(block_q8_0*, const float*, int) noexcept;
static const QuantQ8Fn kQ8Quant = cpu_x86::cpu_features().avx2
    ? cpu_x86::quantize_q8_avx2
    : kQ8Quant_scalar;

// Runtime-selected Q4K×Q8_0 GEMV range (VNNI > AVX2) — used by fused gate+up lambda
using Q4kQ8RangeFn = void(*)(float*, const uint8_t*, const block_q8_0*,
                              int, int, int, size_t) noexcept;
static const Q4kQ8RangeFn kQ4kQ8Range =
    cpu_x86::cpu_features().avxvnni
        ? cpu_x86::gemv_q4k_q8_avxvnni_range
        : cpu_x86::gemv_q4k_q8_avx2_range;

// ─── GEMV with Q8_0-quantised input (single token, Phase 4) ─────────────────
// Uses maddubs INT8 path for Q4_K/Q6_K; falls back to FP32 gemv_wt for F16.
static void gemv_wt_q8(float* y, const WT& W,
                        const block_q8_0* xq8, const float* xf32,
                        int out, int in, ThreadPool* pool) noexcept {
    if (W.q8_fast && cpu_x86::cpu_features().avx2) {
        if (W.dtype == DType::Q4_K_M) {
            // ── Fast path: 8-row interleaved repacked kernel (AVX2 or AVX-VNNI) ──────
            if (W.r8 && !W.r8->empty() && out % 8 == 0) {
                const uint8_t* R8 = W.r8->data();
                const int n8 = out / 8;  // number of 8-row groups
                const bool vnni = cpu_x86::cpu_features().avxvnni;
                if (pool && n8 >= 2) {
                    pool->parallel_for(n8, [&](int gs, int ge) {
                        if (vnni)
                            cpu_x86::gemv_q4k_r8_avxvnni_range(y, R8, xq8,
                                                                  gs*8, ge*8, in);
                        else
                            cpu_x86::gemv_q4k_r8_avx2_range(y, R8, xq8,
                                                              gs*8, ge*8, in);
                    });
                } else {
                    if (vnni)
                        cpu_x86::gemv_q4k_r8_avxvnni_range(y, R8, xq8, 0, out, in);
                    else
                        cpu_x86::gemv_q4k_r8_avx2_range(y, R8, xq8, 0, out, in);
                }
                return;
            }
            // ── Fallback: 4-row scatter kernel (non-multiple-of-8, or no AVX2) ─
            const uint8_t* Wq        = static_cast<const uint8_t*>(W.data);
            const size_t   row_bytes = (size_t)(in / QK_K) * Q4K_BYTES;
            const bool     vnni      = cpu_x86::cpu_features().avxvnni;
            if (pool && out >= 32) {
                pool->parallel_for(out, [&](int s, int e) {
                    if (vnni)
                        cpu_x86::gemv_q4k_q8_avxvnni_range(y, Wq, xq8, s, e, in, row_bytes);
                    else
                        cpu_x86::gemv_q4k_q8_avx2_range(y, Wq, xq8, s, e, in, row_bytes);
                });
            } else {
                if (vnni)
                    cpu_x86::gemv_q4k_q8_avxvnni_range(y, Wq, xq8, 0, out, in, row_bytes);
                else
                    cpu_x86::gemv_q4k_q8_avx2_range(y, Wq, xq8, 0, out, in, row_bytes);
            }
            return;
        }
        if (W.dtype == DType::Q6_K) {
            const uint8_t* Wq       = static_cast<const uint8_t*>(W.data);
            const size_t   row_bytes = (size_t)(in / QK_K) * Q6K_BYTES;
            if (pool && out >= 32) {
                pool->parallel_for(out, [&](int s, int e) {
                    cpu_x86::gemv_q6k_q8_avx2_range(y, Wq, xq8, s, e, in, row_bytes);
                });
            } else {
                cpu_x86::gemv_q6k_q8_avx2_range(y, Wq, xq8, 0, out, in, row_bytes);
            }
            return;
        }
        if (W.dtype == DType::Q8_0) {
            gemv_q8_0_weight_q8(y, static_cast<const uint8_t*>(W.data), xq8, out, in, pool);
            return;
        }
    }
    gemv_wt(y, W, xf32, out, in, pool);
}

// ─── GEMM with Q8_0-quantised input (T tokens, Phase 4) ─────────────────────
// Y[T×M]:  Y[t*M + m] = W[m,:] · XQ8[t,:]
// XQ8 layout: row t starts at XQ8 + t*(in/32)
// XF32 fallback: row t starts at XF32 + t*in  (for F16 weights)
static void gemm_wt_q8(float* Y, const WT& W,
                        const block_q8_0* XQ8, const float* XF32,
                        int M, int T, int in, ThreadPool* pool) noexcept {
    if (W.q8_fast && cpu_x86::cpu_features().avx2) {
        if (W.dtype == DType::Q4_K_M) {
            const uint8_t* Wq        = static_cast<const uint8_t*>(W.data);
            const size_t   row_bytes = (size_t)(in / QK_K) * Q4K_BYTES;
            const bool     vnni      = cpu_x86::cpu_features().avxvnni;
            if (pool && M >= 32) {
                pool->parallel_for(M, [&](int s, int e) {
                    if (vnni)
                        cpu_x86::gemm_q4k_q8_avxvnni_range(Y, Wq, XQ8, M, T, in, row_bytes, s, e);
                    else
                        cpu_x86::gemm_q4k_q8_avx2_range(Y, Wq, XQ8, M, T, in, row_bytes, s, e);
                });
            } else {
                if (vnni)
                    cpu_x86::gemm_q4k_q8_avxvnni_range(Y, Wq, XQ8, M, T, in, row_bytes, 0, M);
                else
                    cpu_x86::gemm_q4k_q8_avx2_range(Y, Wq, XQ8, M, T, in, row_bytes, 0, M);
            }
            return;
        }
        if (W.dtype == DType::Q6_K) {
            const uint8_t* Wq       = static_cast<const uint8_t*>(W.data);
            const size_t   row_bytes = (size_t)(in / QK_K) * Q6K_BYTES;
            if (pool && M >= 32) {
                pool->parallel_for(M, [&](int s, int e) {
                    for (int t = 0; t < T; t++)
                        cpu_x86::gemv_q6k_q8_avx2_range(
                            Y+(size_t)t*M, Wq, XQ8+(size_t)t*(in/32), s, e, in, row_bytes);
                });
            } else {
                for (int t = 0; t < T; t++)
                    cpu_x86::gemv_q6k_q8_avx2_range(
                        Y+(size_t)t*M, Wq, XQ8+(size_t)t*(in/32), 0, M, in, row_bytes);
            }
            return;
        }
    }
    for (int t = 0; t < T; t++)
        gemv_wt(Y + (size_t)t*M, W, XF32 + (size_t)t*in, M, in, pool);
}

// Embedding lookup: dequantise one row to f32
static void emb_lookup(float* out, const WT& W, int row, int cols) noexcept {
    if (W.dtype == DType::F16) {
        const uint16_t* p = static_cast<const uint16_t*>(W.data) + (size_t)row * cols;
        for (int i = 0; i < cols; i++) out[i] = f16_to_f32(p[i]);
    } else if (W.dtype == DType::Q6_K) {
        const size_t row_bytes = (cols / QK_K) * Q6K_BYTES;
        q6k_row_to_f32(out, static_cast<const uint8_t*>(W.data) + (size_t)row * row_bytes, cols);
    } else if (W.dtype == DType::Q8_0) {
        const size_t row_bytes = (size_t)((cols + 31) / 32) * 34;
        q8_0_weight_row_to_f32(out,
            static_cast<const uint8_t*>(W.data) + (size_t)row * row_bytes,
            cols);
    } else if (W.dtype == DType::Q5_0) {
        const size_t row_bytes = (size_t)((cols + 31) / 32) * 22;
        q5_0_row_to_f32(out,
            static_cast<const uint8_t*>(W.data) + (size_t)row * row_bytes,
            cols);
    } else if (W.dtype == DType::Q4_K_M) {
        const size_t row_bytes = (cols / QK_K) * Q4K_BYTES;
        // Use Q4K dot trick: dot against all-ones → dequant
        // Simpler: manually dequant the row
        const uint8_t* block_row = static_cast<const uint8_t*>(W.data) + (size_t)row * row_bytes;
        const int nb = cols / QK_K;
        for (int b = 0; b < nb; b++) {
            const uint8_t* block  = block_row + b * Q4K_BYTES;
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits,    block,     2);
            std::memcpy(&dmin_bits, block + 2, 2);
            const float d    = f16_to_f32(d_bits);
            const float dmin = f16_to_f32(dmin_bits);
            const uint8_t* scales = block + 4;
            const uint8_t* qs     = block + 16;
            float* ob = out + b * QK_K;
            for (int chunk = 0; chunk < 4; chunk++) {
                uint8_t sc1, m1, sc2, m2;
                q4k_get_scale_min(chunk*2+0, scales, sc1, m1);
                q4k_get_scale_min(chunk*2+1, scales, sc2, m2);
                const float d1 = d*sc1, mf1 = dmin*m1;
                const float d2 = d*sc2, mf2 = dmin*m2;
                const uint8_t* q = qs + chunk * 32;
                float* oc = ob + chunk * 64;
                for (int l = 0; l < 32; l++) {
                    oc[l]      = d1 * (q[l] & 0xF) - mf1;
                    oc[l + 32] = d2 * (q[l] >> 4 ) - mf2;
                }
            }
        }
    } else {
        // Fallback: treat as F16
        const uint16_t* p = static_cast<const uint16_t*>(W.data) + (size_t)row * cols;
        for (int i = 0; i < cols; i++) out[i] = f16_to_f32(p[i]);
    }
}

struct LlamaLayerW {
    const float* attn_norm;   // always F32
    const float* ffn_norm;    // always F32
    WT wq, wk, wv, wo;        // QKV + output projection
    const float* bq{nullptr};
    const float* bk{nullptr};
    const float* bv{nullptr};
    WT wgate, wup, wdown;     // SwiGLU FFN
};

struct LlamaWeights {
    WT           tok_embd;
    const float* output_norm;  // always F32
    WT           output;       // lm head
    std::vector<LlamaLayerW> layers;
};

// Repack a Q4K_M weight matrix to 8-row interleaved format (same total bytes).
// No-op if dtype is not Q4K or nrows is not a multiple of 8.
static void repack_q4k_if_avx2(WT& w, int nrows, int ncols) noexcept {
    if (w.dtype != DType::Q4_K_M || nrows % 8 != 0) return;
    if (!cpu_x86::cpu_features().avx2) return;
    const int nb = ncols / QK_K;
    w.r8 = std::make_shared<std::vector<uint8_t>>((size_t)(nrows / 8) * nb * 1152);
    cpu_x86::repack_q4k_8rows(w.r8->data(),
                               static_cast<const uint8_t*>(w.data),
                               nrows, ncols);
}

static LlamaWeights build_weights(const Engine& e, int n_layer,
                                   int n_embd, int n_heads, int n_kv_heads,
                                   int ffn_dim) {
    LlamaWeights w;
    w.tok_embd   = make_wt(e, "token_embd.weight");
    w.output_norm = w_f32(e, "output_norm.weight");
    w.output     = make_wt_opt(e, "output.weight", w.tok_embd);
    w.layers.resize(n_layer);
    char name[128];
    const int hd   = n_embd / n_heads;      // head_dim
    const int q_out   = n_heads    * hd;    // = n_embd
    const int kv_out  = n_kv_heads * hd;
    const bool enable_r8 = true;
    const bool enable_q8_fast = true;
    w.tok_embd.q8_fast = enable_q8_fast;
    w.output.q8_fast = enable_q8_fast;
    for (int l = 0; l < n_layer; l++) {
        auto& lw = w.layers[l];
        std::snprintf(name, sizeof(name), "blk.%d.attn_norm.weight",  l); lw.attn_norm = w_f32(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight",   l); lw.ffn_norm  = w_f32(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.weight",     l); lw.wq   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.bias",       l); if (const Tensor* t = e.weight(name)) lw.bq = static_cast<const float*>(t->raw_data());
        std::snprintf(name, sizeof(name), "blk.%d.attn_k.weight",     l); lw.wk   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_k.bias",       l); if (const Tensor* t = e.weight(name)) lw.bk = static_cast<const float*>(t->raw_data());
        std::snprintf(name, sizeof(name), "blk.%d.attn_v.weight",     l); lw.wv   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_v.bias",       l); if (const Tensor* t = e.weight(name)) lw.bv = static_cast<const float*>(t->raw_data());
        std::snprintf(name, sizeof(name), "blk.%d.attn_output.weight",l); lw.wo   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight",   l); lw.wgate = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_up.weight",     l); lw.wup   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down.weight",   l); lw.wdown = make_wt(e, name);
        lw.wq.q8_fast = lw.wk.q8_fast = lw.wv.q8_fast = lw.wo.q8_fast = enable_q8_fast;
        lw.wgate.q8_fast = lw.wup.q8_fast = lw.wdown.q8_fast = enable_q8_fast;
        // Repack Q4K matrices to 8-row interleaved for faster decode GEMV
        if (enable_r8) {
            repack_q4k_if_avx2(lw.wq,    q_out,  n_embd);
            repack_q4k_if_avx2(lw.wk,    kv_out, n_embd);
            repack_q4k_if_avx2(lw.wv,    kv_out, n_embd);
            repack_q4k_if_avx2(lw.wo,    n_embd, q_out);
            repack_q4k_if_avx2(lw.wgate, ffn_dim, n_embd);
            repack_q4k_if_avx2(lw.wup,   ffn_dim, n_embd);
            repack_q4k_if_avx2(lw.wdown, n_embd, ffn_dim);
        }
    }
    return w;
}

// ─── RoPE frequency cache ─────────────────────────────────────────────────────
// Precomputed cos/sin for each (position, dim_pair) to avoid runtime division.

struct RopeCache {
    int    head_dim;
    float  theta;
    int    max_pos;
    std::vector<float> cos_cache;  // [max_pos][head_dim/2]
    std::vector<float> sin_cache;

    void build(int hd, float th, int mp) {
        head_dim = hd; theta = th; max_pos = mp;
        cos_cache.resize((size_t)mp * (hd / 2));
        sin_cache.resize((size_t)mp * (hd / 2));
        for (int p = 0; p < mp; p++) {
            for (int i = 0; i < hd / 2; i++) {
                float freq = 1.f / std::pow(theta, (float)(2 * i) / (float)hd);
                float angle = (float)p * freq;
                cos_cache[(size_t)p * (hd / 2) + i] = std::cos(angle);
                sin_cache[(size_t)p * (hd / 2) + i] = std::sin(angle);
            }
        }
    }
};

static void apply_rope(float* x, int pos, int n_heads, int head_dim,
                        const RopeCache& cache) noexcept {
    const int half = head_dim / 2;
    const float* cos_row = cache.cos_cache.data() + (size_t)pos * half;
    const float* sin_row = cache.sin_cache.data() + (size_t)pos * half;
    for (int h = 0; h < n_heads; h++) {
        float* hptr = x + h * head_dim;
        for (int i = 0; i < half; i++) {
            float x0 = hptr[i];
            float x1 = hptr[i + half];
            hptr[i]        = x0 * cos_row[i] - x1 * sin_row[i];
            hptr[i + half] = x0 * sin_row[i] + x1 * cos_row[i];
        }
    }
}

// Fusion E: apply RoPE to K and store directly into KV cache — eliminates k→cache copy
static void rope_and_store_k(float* k_cache,
                               const float* k_src,
                               int pos, int n_heads, int head_dim,
                               const RopeCache& cache) noexcept {
    const int half = head_dim / 2;
    const float* cos_row = cache.cos_cache.data() + (size_t)pos * half;
    const float* sin_row = cache.sin_cache.data() + (size_t)pos * half;
    for (int h = 0; h < n_heads; h++) {
        const float* src = k_src   + h * head_dim;
        float*       dst = k_cache + h * head_dim;
        for (int i = 0; i < half; i++) {
            float x0 = src[i];
            float x1 = src[i + half];
            dst[i]        = x0 * cos_row[i] - x1 * sin_row[i];
            dst[i + half] = x0 * sin_row[i] + x1 * cos_row[i];
        }
    }
}

static inline void add_bias(float* y, const float* b, int n) noexcept {
    if (!b) return;
    for (int i = 0; i < n; ++i) y[i] += b[i];
}

static inline void add_bias_rows(float* y, const float* b, int rows, int n) noexcept {
    if (!b) return;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i)
            y[(size_t)r * n + i] += b[i];
}

// ─── Model state (KV cache + scratch buffers) ─────────────────────────────────

struct LlamaState {
    // Hyperparameters
    int n_layer, n_ctx, n_embd, n_heads, n_kv_heads, head_dim, n_vocab, ffn_dim;
    float rms_eps, rope_theta;

    // KV cache: [n_layer][n_ctx][n_kv_heads * head_dim]
    int kv_embd;
    std::vector<float> kv_k;
    std::vector<float> kv_v;
    int n_past{0};

    // Scratch buffers
    std::vector<float> x, xb;
    std::vector<float> q, k, v;    // current token Q/K/V
    std::vector<float> attn;       // attention scores [n_ctx]
    std::vector<float> attn_out;   // attention output [n_embd]
    std::vector<float> gate, up;   // FFN gate and up projections [ffn_dim]
    std::vector<float> ffn;        // FFN down projection output [n_embd]
    std::vector<float> logits;
    // Q8_0 scratch: holds max(n_embd, ffn_dim)/32 blocks for pre-quantised GEMV input
    std::vector<uint8_t> xq8_scratch;
    block_q8_0* xq8() noexcept { return reinterpret_cast<block_q8_0*>(xq8_scratch.data()); }

    // Pre-built weight pointers
    LlamaWeights weights;

    // RoPE frequency cache
    RopeCache rope_cache;

    // Thread pool — owned_pool_ holds ownership when we create it ourselves;
    // pool is the raw pointer used everywhere (may point to a static external pool).
    std::unique_ptr<ThreadPool> owned_pool_;
    ThreadPool* pool = nullptr;

    LlamaState(int nl, int nc, int ne, int nh, int nkv, int nv, int fd,
               float eps, float rtheta, int n_threads,
               ThreadPool* ext_pool = nullptr)
        : n_layer(nl), n_ctx(nc), n_embd(ne), n_heads(nh), n_kv_heads(nkv)
        , head_dim(ne / nh), n_vocab(nv), ffn_dim(fd)
        , rms_eps(eps), rope_theta(rtheta)
        , kv_embd(nkv * (ne / nh))
    {
        kv_k.resize((size_t)nl * nc * kv_embd, 0.f);
        kv_v.resize((size_t)nl * nc * kv_embd, 0.f);
        x.resize(ne); xb.resize(ne);
        q.resize(nh  * head_dim);
        k.resize(nkv * head_dim);
        v.resize(nkv * head_dim);
        attn.resize(nc);
        attn_out.resize(ne);
        gate.resize(fd); up.resize(fd); ffn.resize(ne);
        logits.resize(nv);
        xq8_scratch.resize(sizeof(block_q8_0) * (std::max(ne, fd) / 32));
        rope_cache.build(head_dim, rope_theta, nc);
        if (ext_pool) {
            pool = ext_pool;
        } else if (n_threads > 1) {
            pool = (owned_pool_ = std::make_unique<ThreadPool>(n_threads - 1)).get();
        }
    }

    float* k_buf(int layer, int pos) {
        return kv_k.data() + ((size_t)layer * n_ctx + pos) * kv_embd;
    }
    float* v_buf(int layer, int pos) {
        return kv_v.data() + ((size_t)layer * n_ctx + pos) * kv_embd;
    }
};

// ─── AVX2 helper: dot product of two F32 vectors (Phase C) ─────────────────
static inline float avx2_dot_f32(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  int n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i  ), _mm256_loadu_ps(b+i  ), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), acc1);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// ─── AVX2 helper: SAXPY  out[i] += alpha * v[i] ──────────────────────────────
static inline void avx2_saxpy(float* __restrict__ out,
                               float alpha,
                               const float* __restrict__ v,
                               int n) noexcept {
    const __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out+i, _mm256_fmadd_ps(va, _mm256_loadu_ps(v+i),
                                                     _mm256_loadu_ps(out+i)));
    for (; i < n; i++) out[i] += alpha * v[i];
}

// ─── Single-token forward pass ────────────────────────────────────────────────

static void llama_forward_token(int token_id, int pos, LlamaState& s) {
    const int E   = s.n_embd;
    const int H   = s.n_heads;
    const int KVH = s.n_kv_heads;
    const int HD  = s.head_dim;
    const int L   = s.n_layer;
    const int FD  = s.ffn_dim;
    const float scale = 1.f / std::sqrt((float)HD);
    const LlamaWeights& w = s.weights;
    ThreadPool* pool = s.pool;

    // ── 1. Token embedding lookup (dtype-aware) ──────────────────────────────
    emb_lookup(s.x.data(), w.tok_embd, token_id, E);

    // ── 2. Transformer blocks ────────────────────────────────────────────────
    for (int l = 0; l < L; l++) {
        const LlamaLayerW& lw = w.layers[l];

        // ---- RMSNorm (attention): Fusion A — read x, write xb without copy ----
        rms_norm_out(s.xb.data(), s.x.data(), lw.attn_norm, E, s.rms_eps);
        kQ8Quant(s.xq8(), s.xb.data(), E);

        // ---- Q, K, V projections (INT8 GEMV via Q8_0 input) ----
        gemv_wt_q8(s.q.data(), lw.wq, s.xq8(), s.xb.data(), H   * HD, E, pool);
        gemv_wt_q8(s.k.data(), lw.wk, s.xq8(), s.xb.data(), KVH * HD, E, pool);
        gemv_wt_q8(s.v.data(), lw.wv, s.xq8(), s.xb.data(), KVH * HD, E, pool);
        add_bias(s.q.data(), lw.bq, H   * HD);
        add_bias(s.k.data(), lw.bk, KVH * HD);
        add_bias(s.v.data(), lw.bv, KVH * HD);

        // ---- RoPE Q in-place; Fusion E: RoPE K + store to cache in one pass ----
        apply_rope(s.q.data(), pos, H, HD, s.rope_cache);
        rope_and_store_k(s.k_buf(l, pos), s.k.data(), pos, KVH, HD, s.rope_cache);
        std::copy(s.v.begin(), s.v.end(), s.v_buf(l, pos));

        // ---- GQA multi-head attention (Phase C: AVX2 dot + SAXPY) ----
        // Note: k_buf(l, pos) already contains RoPE-rotated K (via rope_and_store_k)
        const int seq = pos + 1;
        const int kv_group = H / KVH;
        std::fill(s.attn_out.begin(), s.attn_out.end(), 0.f);

        for (int h = 0; h < H; h++) {
            const int kv_h = h / kv_group;
            const float* Q_h = s.q.data() + h * HD;

            for (int t = 0; t < seq; t++) {
                const float* K_t = s.k_buf(l, t) + kv_h * HD;
                s.attn[t] = avx2_dot_f32(Q_h, K_t, HD) * scale;
            }
            softmax(s.attn.data(), seq);

            float* out_h = s.attn_out.data() + h * HD;
            for (int t = 0; t < seq; t++)
                avx2_saxpy(out_h, s.attn[t], s.v_buf(l, t) + kv_h * HD, HD);
        }

        // ---- Output projection + residual ----
        kQ8Quant(s.xq8(), s.attn_out.data(), E);
        gemv_wt_q8(s.xb.data(), lw.wo, s.xq8(), s.attn_out.data(), E, E, pool);
        for (int i = 0; i < E; i++) s.x[i] += s.xb[i];

        // ---- RMSNorm (FFN): Fusion A — read x, write xb without copy ----
        rms_norm_out(s.xb.data(), s.x.data(), lw.ffn_norm, E, s.rms_eps);
        kQ8Quant(s.xq8(), s.xb.data(), E);

        // ---- SwiGLU FFN: Fusion C — gate+up+swiglu in single parallel_for ----
        if (lw.wgate.q8_fast && lw.wgate.dtype == DType::Q4_K_M && pool && FD >= 32) {
            float*            gate_ptr = s.gate.data();
            float*            up_ptr   = s.up.data();
            const block_q8_0* xq8p    = s.xq8();
            if (lw.wgate.r8 && !lw.wgate.r8->empty() && FD % 8 == 0) {
                // 8-row repacked path: distribute by group of 8 rows
                const uint8_t* R8gate = lw.wgate.r8->data();
                const uint8_t* R8up   = lw.wup.r8->data();
                const int n8 = FD / 8;
                const bool vnni_fc = cpu_x86::cpu_features().avxvnni;
                pool->parallel_for(n8, [=](int gs, int ge) {
                    const int s_r = gs*8, e_r = ge*8;
                    if (vnni_fc) {
                        cpu_x86::gemv_q4k_r8_avxvnni_range(gate_ptr, R8gate, xq8p, s_r, e_r, E);
                        cpu_x86::gemv_q4k_r8_avxvnni_range(up_ptr,   R8up,   xq8p, s_r, e_r, E);
                    } else {
                        cpu_x86::gemv_q4k_r8_avx2_range(gate_ptr, R8gate, xq8p, s_r, e_r, E);
                        cpu_x86::gemv_q4k_r8_avx2_range(up_ptr,   R8up,   xq8p, s_r, e_r, E);
                    }
                    swiglu_avx2(gate_ptr + s_r, up_ptr + s_r, e_r - s_r);
                });
            } else {
                const uint8_t* Wgate  = static_cast<const uint8_t*>(lw.wgate.data);
                const uint8_t* Wup    = static_cast<const uint8_t*>(lw.wup.data);
                const size_t   rbytes = (size_t)(E / QK_K) * Q4K_BYTES;
                pool->parallel_for(FD, [=](int s_r, int e_r) {
                    kQ4kQ8Range(gate_ptr, Wgate, xq8p, s_r, e_r, E, rbytes);
                    kQ4kQ8Range(up_ptr,   Wup,   xq8p, s_r, e_r, E, rbytes);
                    swiglu_avx2(gate_ptr + s_r, up_ptr + s_r, e_r - s_r);
                });
            }
        } else {
            gemv_wt_q8(s.gate.data(), lw.wgate, s.xq8(), s.xb.data(), FD, E, pool);
            gemv_wt_q8(s.up.data(),   lw.wup,   s.xq8(), s.xb.data(), FD, E, pool);
            swiglu_avx2(s.gate.data(), s.up.data(), FD);
        }

        // ---- FFN down projection + residual (INT8 GEMV) ----
        kQ8Quant(s.xq8(), s.gate.data(), FD);
        gemv_wt_q8(s.ffn.data(), lw.wdown, s.xq8(), s.gate.data(), E, FD, pool);
        for (int i = 0; i < E; i++) s.x[i] += s.ffn[i];
    }

    // ── 3. Final RMSNorm ─────────────────────────────────────────────────────
    rms_norm_avx2(s.x.data(), w.output_norm, E, s.rms_eps);

    // ── 4. LM head (INT8 GEMV) ────────────────────────────────────────────────
    kQ8Quant(s.xq8(), s.x.data(), E);
    gemv_wt_q8(s.logits.data(), w.output, s.xq8(), s.x.data(), s.n_vocab, E, pool);
}

// ─── Batched prefill forward pass (Phase B) ─────────────────────────────────
// Processes T prompt tokens in one call, exploiting L3 cache amortization:
// each weight matrix is warm in L3 while all T tokens are projected against it.
// Only the last token's logits are computed (first generated token).
static void llama_forward_batch(const int32_t* token_ids, int T,
                                 int pos_start, LlamaState& s) {
    const int E   = s.n_embd, H = s.n_heads, KVH = s.n_kv_heads;
    const int HD  = s.head_dim, FD = s.ffn_dim;
    const float scale = 1.f / std::sqrt((float)HD);
    const LlamaWeights& w = s.weights;
    ThreadPool* pool = s.pool;

    std::vector<float> X(T * E), XB(T * E);
    std::vector<float> Q(T * H * HD), K(T * KVH * HD), V(T * KVH * HD);
    std::vector<float> AO(T * E);
    std::vector<float> GATE(T * FD), UP(T * FD), FFN_BUF(T * E);

    // Q8_0 scratch for T tokens — two buffers (E-dim and FD-dim inputs)
    const bool have_avx2 = cpu_x86::cpu_features().avx2;
    std::vector<uint8_t> xq8_e_buf, xq8_fd_buf;
    block_q8_0 *XQ8_E = nullptr, *XQ8_FD = nullptr;
    if (have_avx2) {
        xq8_e_buf .resize(sizeof(block_q8_0) * (size_t)T * (E  / 32));
        xq8_fd_buf.resize(sizeof(block_q8_0) * (size_t)T * (FD / 32));
        XQ8_E  = reinterpret_cast<block_q8_0*>(xq8_e_buf .data());
        XQ8_FD = reinterpret_cast<block_q8_0*>(xq8_fd_buf.data());
    }
    // Quantise T F32 rows of dimension `dim` into an XQ8 array
    auto qbatch = [&](block_q8_0* xq8, const float* xf32, int dim) {
        if (have_avx2)
            for (int t = 0; t < T; t++)
                cpu_x86::quantize_q8_avx2(xq8 + (size_t)t*(dim/32), xf32 + (size_t)t*dim, dim);
    };

    // Embedding lookup
    for (int t = 0; t < T; t++)
        emb_lookup(X.data() + (size_t)t*E, w.tok_embd, token_ids[t], E);

    for (int l = 0; l < s.n_layer; l++) {
        const LlamaLayerW& lw = w.layers[l];

        // RMSNorm (attn): Fusion A — rms_norm_out reads X, writes XB without copy
        for (int t = 0; t < T; t++)
            rms_norm_out(XB.data()+(size_t)t*E, X.data()+(size_t)t*E, lw.attn_norm, E, s.rms_eps);
        qbatch(XQ8_E, XB.data(), E);

        // Q, K, V projections — true GEMM (all T tokens × weight matrix at once)
        gemm_wt_q8(Q.data(),    lw.wq, XQ8_E, XB.data(), H*HD,   T, E, pool);
        gemm_wt_q8(K.data(),    lw.wk, XQ8_E, XB.data(), KVH*HD, T, E, pool);
        gemm_wt_q8(V.data(),    lw.wv, XQ8_E, XB.data(), KVH*HD, T, E, pool);
        add_bias_rows(Q.data(), lw.bq, T, H*HD);
        add_bias_rows(K.data(), lw.bk, T, KVH*HD);
        add_bias_rows(V.data(), lw.bv, T, KVH*HD);

        // RoPE + KV cache fill: Fusion E — RoPE K directly into cache
        for (int t = 0; t < T; t++) {
            const int pos = pos_start + t;
            apply_rope(Q.data()+(size_t)t*H*HD, pos, H, HD, s.rope_cache);
            rope_and_store_k(s.k_buf(l,pos), K.data()+(size_t)t*KVH*HD, pos, KVH, HD, s.rope_cache);
            std::copy(V.data()+(size_t)t*KVH*HD, V.data()+(size_t)(t+1)*KVH*HD, s.v_buf(l,pos));
        }

        // Causal attention (each token t attends to positions 0..pos_start+t)
        std::fill(AO.begin(), AO.end(), 0.f);
        const int kv_group = H / KVH;
        for (int t = 0; t < T; t++) {
            const int pos = pos_start + t;
            const int seq = pos + 1;
            for (int h = 0; h < H; h++) {
                const int kv_h = h / kv_group;
                const float* Q_h = Q.data() + (size_t)t*H*HD + h*HD;
                for (int tt = 0; tt < seq; tt++)
                    s.attn[tt] = avx2_dot_f32(Q_h, s.k_buf(l,tt) + kv_h*HD, HD) * scale;
                softmax(s.attn.data(), seq);
                float* ao_h = AO.data() + (size_t)t*E + h*HD;
                for (int tt = 0; tt < seq; tt++)
                    avx2_saxpy(ao_h, s.attn[tt], s.v_buf(l,tt) + kv_h*HD, HD);
            }
        }

        // Output projection — GEMM all T tokens
        qbatch(XQ8_E, AO.data(), E);
        gemm_wt_q8(XB.data(), lw.wo, XQ8_E, AO.data(), E, T, E, pool);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < E; i++) X[(size_t)t*E+i] += XB[(size_t)t*E+i];

        // RMSNorm (FFN): Fusion A — rms_norm_out reads X, writes XB without copy
        for (int t = 0; t < T; t++)
            rms_norm_out(XB.data()+(size_t)t*E, X.data()+(size_t)t*E, lw.ffn_norm, E, s.rms_eps);
        qbatch(XQ8_E, XB.data(), E);

        // FFN gate + up — GEMM all T tokens
        gemm_wt_q8(GATE.data(), lw.wgate, XQ8_E, XB.data(), FD, T, E, pool);
        gemm_wt_q8(UP.data(),   lw.wup,   XQ8_E, XB.data(), FD, T, E, pool);

        // SwiGLU + FFN down — GEMM all T tokens
        for (int t = 0; t < T; t++)
            swiglu_avx2(GATE.data()+(size_t)t*FD, UP.data()+(size_t)t*FD, FD);
        qbatch(XQ8_FD, GATE.data(), FD);
        gemm_wt_q8(FFN_BUF.data(), lw.wdown, XQ8_FD, GATE.data(), E, T, FD, pool);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < E; i++) X[(size_t)t*E+i] += FFN_BUF[(size_t)t*E+i];
    }

    // Final RMSNorm + lm_head for the last token only
    const int last = T - 1;
    rms_norm_avx2(X.data()+(size_t)last*E, w.output_norm, E, s.rms_eps);
    // Reuse xq8_e_buf for single-token quantise (need 1 token × E/32 blocks)
    block_q8_0* xq8_last = XQ8_E;  // first E/32 blocks of the E buffer
    if (have_avx2) cpu_x86::quantize_q8_avx2(xq8_last, X.data()+(size_t)last*E, E);
    gemv_wt_q8(s.logits.data(), w.output, xq8_last, X.data()+(size_t)last*E, s.n_vocab, E, pool);

    // Copy last token residual into s.x so decode loop can continue seamlessly
    std::copy(X.data()+(size_t)last*E, X.data()+(size_t)T*E, s.x.data());
}

// ─── Sampling (identical strategy to GPT-2 side) ─────────────────────────────

static int32_t sample(const float* logits, int n_vocab,
                       int top_k, float top_p, float temperature,
                       std::mt19937& rng) {
    // Greedy if temperature==0
    if (temperature <= 0.f)
        return static_cast<int32_t>(std::max_element(logits, logits + n_vocab) - logits);

    int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    std::vector<std::pair<float, int32_t>> scored(n_vocab);
    for (int i = 0; i < n_vocab; i++) scored[i] = {logits[i] / temperature, i};
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });

    float mx = scored[0].first, sum = 0.f;
    std::vector<float> probs(k);
    for (int i = 0; i < k; i++) { probs[i] = std::exp(scored[i].first - mx); sum += probs[i]; }
    for (int i = 0; i < k; i++) probs[i] /= sum;

    // nucleus (top-p) filtering
    if (top_p < 1.0f) {
        float cum = 0.f; int cut = k;
        for (int i = 0; i < k; i++) { cum += probs[i]; if (cum >= top_p) { cut = i + 1; break; } }
        sum = 0.f;
        for (int i = 0; i < cut; i++) sum += probs[i];
        for (int i = 0; i < cut; i++) probs[i] /= sum;
        k = cut;
    }

    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(rng), cum = 0.f;
    for (int i = 0; i < k; i++) { cum += probs[i]; if (r < cum) return scored[i].second; }
    return scored[k-1].second;
}

// ─── Public API ───────────────────────────────────────────────────────────────

std::vector<int32_t> llama_generate(const Engine& engine,
                                     std::vector<int32_t> prompt_ids,
                                     const LlamaConfig& cfg) {
    if (prompt_ids.empty())
        throw std::invalid_argument("llama_generate: empty prompt");

    const ModelConfig& mc = engine.model_config();

    const int L   = mc.n_layers;
    const int E   = mc.hidden_dim;
    const int H   = mc.n_heads;
    const int KVH = mc.n_kv_heads > 0 ? mc.n_kv_heads : H;
    // Cap context to cfg.max_context_len so LLaMA-3 (128K) doesn't OOM.
    const int NC  = std::min(mc.max_seq_len, cfg.max_context_len);
    const int V   = mc.vocab_size > 0 ? mc.vocab_size : 32000;
    const float eps    = mc.rms_norm_eps > 0.f ? mc.rms_norm_eps : 1e-5f;
    const float rtheta = mc.rope_theta   > 0.f ? mc.rope_theta   : 10000.f;
    const int FD  = mc.ffn_dim > 0 ? mc.ffn_dim : 4 * E;

    const int n_threads = cfg.n_threads > 0
        ? cfg.n_threads
        : std::min(16, (int)std::thread::hardware_concurrency());

    // Constrain all inference threads to CPUs 0..n_threads-1 (P-cores on Alder Lake)
    // so the OS cannot migrate spinwait workers or the calling thread to E-cores.
    // Worker threads inherit this affinity mask at creation inside LlamaState ctor.
#ifdef __linux__
    cpu_set_t orig_aff, infer_aff;
    sched_getaffinity(0, sizeof(orig_aff), &orig_aff);
    CPU_ZERO(&infer_aff);
    for (int i = 0; i < n_threads; i++) CPU_SET(i, &infer_aff);
    sched_setaffinity(0, sizeof(infer_aff), &infer_aff);
#endif

    // Weight cache: repacking ~520 MB of r8 buffers takes ~200 ms.
    // Reuse across calls unless the engine pointer changes.
    static const Engine*              s_wt_engine  = nullptr;
    static LlamaWeights               s_wt_cache;
    static std::unique_ptr<ThreadPool> s_pool;
    static int                        s_pool_nw    = 0;
    if (s_wt_engine != &engine) {
        s_wt_cache  = build_weights(engine, L, E, H, KVH, FD);
        s_wt_engine = &engine;
    }
    const int n_workers = n_threads - 1;
    if (!s_pool || s_pool_nw != n_workers) {
        s_pool    = std::make_unique<ThreadPool>(n_workers);
        s_pool_nw = n_workers;
    }
    // Pin calling thread to CPU n_workers so it has an exclusive P-core
    // (workers[i] are on CPUs 0..n_workers-1; caller gets CPU n_workers).
    // This eliminates CPU sharing between the serial calling thread and any worker.
    ThreadPool::pin_caller(n_workers);

    LlamaState state(L, NC, E, H, KVH, V, FD, eps, rtheta, n_threads, s_pool.get());
    state.weights = s_wt_cache;  // O(1): shared_ptr refcount bumps only
    std::mt19937 rng(42);

    // Prepend BOS token (id=1 for LLaMA-2/3/TinyLlama) if not already present
    const int32_t bos = engine.bos_id();
    if (prompt_ids.empty() || prompt_ids[0] != bos)
        prompt_ids.insert(prompt_ids.begin(), bos);

    // Prefill — batched when prompt > 1 token (Phase B)
    const int nprompt = (int)prompt_ids.size();
    if (nprompt > 1) {
        llama_forward_batch(prompt_ids.data(), nprompt, 0, state);
        state.n_past = nprompt;
    } else {
        llama_forward_token(prompt_ids[0], 0, state);
        state.n_past = 1;
    }
    if (cfg.verbose) {
        const float* lp = state.logits.data();
        std::vector<std::pair<float,int>> ranked(V);
        for (int i = 0; i < V; i++) ranked[i] = {lp[i], i};
        std::partial_sort(ranked.begin(), ranked.begin()+5, ranked.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        std::fprintf(stderr, "[DBG] top-5 prefill:");
        for (int i = 0; i < 5; i++)
            std::fprintf(stderr, " %d(%.2f)", ranked[i].second, ranked[i].first);
        std::fprintf(stderr, "\n");
    }

    // Decode loop
    std::vector<int32_t> generated;
    generated.reserve(cfg.max_new_tokens);

    // Repetition-penalty helper: divides logits of recently-seen tokens.
    std::vector<int32_t> rep_ctx(prompt_ids.begin(), prompt_ids.end());
    auto apply_rep_penalty = [&](float* lp) {
        if (cfg.rep_penalty == 1.0f) return;
        const int win   = std::min(cfg.rep_penalty_last_n, (int)rep_ctx.size());
        const int start = (int)rep_ctx.size() - win;
        for (int i = start; i < (int)rep_ctx.size(); i++) {
            const int32_t id = rep_ctx[i];
            if (id < 0 || id >= V) continue;
            lp[id] = lp[id] > 0.f ? lp[id] / cfg.rep_penalty
                                   : lp[id] * cfg.rep_penalty;
        }
    };

    apply_rep_penalty(state.logits.data());
    int32_t next_id = sample(state.logits.data(), V,
                              cfg.top_k, cfg.top_p, cfg.temperature, rng);

    for (int step = 0; step < cfg.max_new_tokens; step++) {
        if (next_id == engine.eos_id()) break;  // stop cleanly; EOS not emitted
        if (state.n_past >= NC) break;

        generated.push_back(next_id);
        if (cfg.on_token) cfg.on_token(next_id);

        llama_forward_token(next_id, state.n_past, state);
        state.n_past++;
        rep_ctx.push_back(next_id);
        apply_rep_penalty(state.logits.data());
        next_id = sample(state.logits.data(), V,
                          cfg.top_k, cfg.top_p, cfg.temperature, rng);
    }

#ifdef __linux__
    sched_setaffinity(0, sizeof(orig_aff), &orig_aff);
#endif

    return generated;
}

// ─── Tokenization ─────────────────────────────────────────────────────────────
// LLaMA GGUF stores tokens with SentencePiece '▁' (U+2581) prefix for spaces.
// We decode by replacing ▁ with a plain space.

std::string llama_decode_tokens(const Engine& engine,
                                  const std::vector<int32_t>& token_ids) {
    const auto& vocab = engine.vocabulary();
    std::string out;
    auto hex_digit = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (int32_t id : token_ids) {
        if (id < 0 || id >= (int32_t)vocab.size()) continue;
        const std::string& tok = vocab[id];

        // SentencePiece byte-fallback tokens: "<0xHH>" → emit raw byte
        if (tok.size() == 6 &&
            tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
            int hi = hex_digit(tok[3]), lo = hex_digit(tok[4]);
            if (hi >= 0 && lo >= 0) {
                out += static_cast<char>((hi << 4) | lo);
                continue;
            }
        }

        // Replace SentencePiece space marker ▁ (UTF-8: 0xE2 0x96 0x81)
        std::string s;
        s.reserve(tok.size());
        size_t i = 0;
        while (i < tok.size()) {
            if (i + 2 < tok.size() &&
                (uint8_t)tok[i]   == 0xE2 &&
                (uint8_t)tok[i+1] == 0x96 &&
                (uint8_t)tok[i+2] == 0x81) {
                s += ' ';
                i += 3;
            } else {
                s += tok[i++];
            }
        }
        out += s;
    }
    return out;
}

std::vector<int32_t> llama_encode_simple(const Engine& engine,
                                           std::string_view text, bool raw) {
    const auto& vocab = engine.vocabulary();
    if (vocab.empty())
        throw std::runtime_error("llama_encode_simple: vocabulary is empty");

    std::unordered_map<std::string, int32_t> str_to_id;
    str_to_id.reserve(vocab.size());
    for (int32_t i = 0; i < (int32_t)vocab.size(); i++)
        str_to_id.emplace(vocab[i], i);

    // Build byte-fallback map: raw byte value → token ID via "<0xHH>" entries.
    std::array<int32_t, 256> byte_to_id;
    byte_to_id.fill(-1);
    {
        char buf[8];
        for (int b = 0; b < 256; b++) {
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto it = str_to_id.find(buf);
            if (it != str_to_id.end()) byte_to_id[b] = it->second;
        }
    }

    // Normalise a text segment: replace spaces with ▁ (optionally prepend ▁).
    auto normalise = [](std::string_view seg, bool prepend) -> std::string {
        std::string out;
        out.reserve(seg.size() * 2 + 3);
        if (prepend) out += "\xE2\x96\x81";
        for (char c : seg) {
            if (c == ' ') out += "\xE2\x96\x81";
            else          out += c;
        }
        return out;
    };

    // BPE-encode a pre-normalised string segment, appending to `out`.
    auto bpe_encode = [&](const std::string& norm, std::vector<int32_t>& out) {
        size_t pos = 0;
        while (pos < norm.size()) {
            size_t best_len = 0;
            int32_t best_id = -1;
            const size_t max_len = std::min(norm.size() - pos, size_t(64));
            for (size_t len = max_len; len >= 1; len--) {
                auto it = str_to_id.find(norm.substr(pos, len));
                if (it != str_to_id.end()) { best_len = len; best_id = it->second; break; }
            }
            if (best_id >= 0) { out.push_back(best_id); pos += best_len; }
            else {
                const auto b = static_cast<unsigned char>(norm[pos]);
                if (byte_to_id[b] >= 0) out.push_back(byte_to_id[b]);
                pos++;
            }
        }
    };

    std::vector<int32_t> ids;

    if (!raw) {
        // Plain-text mode: prepend leading ▁ and replace spaces.
        bpe_encode(normalise(text, /*prepend=*/true), ids);
        return ids;
    }

    // Raw / chat-template mode (parse_special=true):
    // Pre-split the text at EOS/BOS string boundaries so that e.g. "</s>"
    // is emitted as the dedicated EOS token ID even when immediately
    // preceded by "." (which would otherwise form a merged BPE token ".</").
    struct Special { std::string str; int32_t id; };
    std::vector<Special> specials;
    {
        const int32_t eos = engine.eos_id(), bos = engine.bos_id();
        if (eos >= 0 && eos < (int32_t)vocab.size() && !vocab[eos].empty())
            specials.push_back({vocab[eos], eos});
        if (bos >= 0 && bos < (int32_t)vocab.size() && !vocab[bos].empty())
            specials.push_back({vocab[bos], bos});
        std::sort(specials.begin(), specials.end(),
                  [](const Special& a, const Special& b){ return a.str.size() > b.str.size(); });
    }

    size_t pos = 0;
    while (pos < text.size()) {
        // Check for a special token at current position.
        bool matched = false;
        for (const auto& sp : specials) {
            if (pos + sp.str.size() <= text.size() &&
                text.substr(pos, sp.str.size()) == sp.str) {
                ids.push_back(sp.id);
                pos += sp.str.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Accumulate a regular-text segment up to the next special token.
        std::string seg;
        while (pos < text.size()) {
            bool at_special = false;
            for (const auto& sp : specials) {
                if (pos + sp.str.size() <= text.size() &&
                    text.substr(pos, sp.str.size()) == sp.str) {
                    at_special = true; break;
                }
            }
            if (at_special) break;
            seg += text[pos++];
        }
        if (!seg.empty())
            bpe_encode(normalise(seg, /*prepend=*/false), ids);
    }
    return ids;
}

} // namespace axonforge
