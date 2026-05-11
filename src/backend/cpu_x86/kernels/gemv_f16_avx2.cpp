// ============================================================
// gemv_f16_avx2.cpp — F16-weight × F32-activation GEMV kernel
//
// Compiled with:  -mavx2 -mfma -mf16c -O3
//
// Core instruction sequence per 16 elements (one iteration):
//   _mm256_cvtph_ps  : 8×F16 → 8×F32 (F16C, 1 cycle throughput)
//   _mm256_fmadd_ps  : 8×FMA in one instruction (FMA3, 0.5 cycle)
//   → ~16 elements per 3 cycles = 5.3 elem/cycle (vs scalar ~0.5)
//
// Thread-parallel version (gemv_f16_avx2_range) used by the MT dispatcher.
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace axonforge::cpu_x86 {

// ── Horizontal sum of __m256 → scalar float ──────────────────────────────────
static inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ── Scalar F16→F32 for tail elements (< 8 remaining) ─────────────────────────
static inline float scalar_f16(uint16_t h) noexcept {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s; }
        else {
            uint32_t em = m, ex = 127u - 14u;
            while (!(em & 0x400u)) { em <<= 1; ex--; }
            f = s | (ex << 23) | ((em & 0x3FFu) << 13);
        }
    } else if (e == 31u) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float v; std::memcpy(&v, &f, 4); return v;
}

// ── Single-row dot product: F16 weight row × F32 activation ──────────────────
// Unrolled 2× (two independent accumulators hide cvtph_ps latency ~6 cycles)
static float dot_f16_avx2(const uint16_t* __restrict__ W,
                           const float*    __restrict__ x,
                           int K) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m256 w0 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(W + k)));
        const __m256 w1 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(W + k + 8)));
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k),     acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), acc1);
    }
    for (; k + 8 <= K; k += 8) {
        const __m256 w0 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(W + k)));
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    float sum = hsum256(acc0);

    for (; k < K; k++) sum += scalar_f16(W[k]) * x[k];  // tail (<8 elems)
    return sum;
}

// ── 4-row simultaneous F16 dot product ───────────────────────────────────────
// Rows o, o+1, o+2, o+3 share the same x loads; 8 independent accumulators.
static inline void dot_f16_4rows_(float* y, int o,
                                   const uint16_t* W, int in,
                                   const float* x) noexcept {
    __m256 a0a = _mm256_setzero_ps(), a0b = _mm256_setzero_ps();
    __m256 a1a = _mm256_setzero_ps(), a1b = _mm256_setzero_ps();
    __m256 a2a = _mm256_setzero_ps(), a2b = _mm256_setzero_ps();
    __m256 a3a = _mm256_setzero_ps(), a3b = _mm256_setzero_ps();

    const uint16_t* r0 = W + (size_t)(o + 0) * in;
    const uint16_t* r1 = W + (size_t)(o + 1) * in;
    const uint16_t* r2 = W + (size_t)(o + 2) * in;
    const uint16_t* r3 = W + (size_t)(o + 3) * in;

    int k = 0;
    for (; k + 16 <= in; k += 16) {
        // Load x once (16 floats = 2 AVX2 regs) — shared across 4 rows
        const __m256 xv0 = _mm256_loadu_ps(x + k);
        const __m256 xv1 = _mm256_loadu_ps(x + k + 8);

        a0a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r0+k  ))), xv0, a0a);
        a0b = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r0+k+8))), xv1, a0b);
        a1a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r1+k  ))), xv0, a1a);
        a1b = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r1+k+8))), xv1, a1b);
        a2a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r2+k  ))), xv0, a2a);
        a2b = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r2+k+8))), xv1, a2b);
        a3a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r3+k  ))), xv0, a3a);
        a3b = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r3+k+8))), xv1, a3b);
    }
    // Handle remaining 8-wide tail
    for (; k + 8 <= in; k += 8) {
        const __m256 xv0 = _mm256_loadu_ps(x + k);
        a0a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r0+k))), xv0, a0a);
        a1a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r1+k))), xv0, a1a);
        a2a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r2+k))), xv0, a2a);
        a3a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r3+k))), xv0, a3a);
    }
    float s0 = hsum256(_mm256_add_ps(a0a, a0b));
    float s1 = hsum256(_mm256_add_ps(a1a, a1b));
    float s2 = hsum256(_mm256_add_ps(a2a, a2b));
    float s3 = hsum256(_mm256_add_ps(a3a, a3b));
    // scalar tail (rare, only when in % 8 != 0)
    for (; k < in; k++) {
        float xk = x[k];
        s0 += scalar_f16(r0[k]) * xk;
        s1 += scalar_f16(r1[k]) * xk;
        s2 += scalar_f16(r2[k]) * xk;
        s3 += scalar_f16(r3[k]) * xk;
    }
    y[o + 0] = s0;
    y[o + 1] = s1;
    y[o + 2] = s2;
    y[o + 3] = s3;
}

// ── Full GEMV for a range of output rows [o_start, o_end) ────────────────────
// Called from ThreadPool workers — each thread gets a disjoint row slice.
void gemv_f16_avx2_range(float* __restrict__       y,
                          const uint16_t* __restrict__ W,
                          const float*    __restrict__ x,
                          const float*    __restrict__ b,
                          int o_start, int o_end, int in) noexcept {
    int o = o_start;
    // 4-row tile (b==nullptr is normal for llama; GPT-2 has bias only on attn/ffn)
    if (!b) {
        for (; o + 4 <= o_end; o += 4)
            dot_f16_4rows_(y, o, W, in, x);
    }
    for (; o < o_end; o++) {
        float acc = dot_f16_avx2(W + (size_t)o * in, x, in);
        y[o] = acc + (b ? b[o] : 0.f);
    }
}

// ── Full GEMV (single-threaded entry point) ───────────────────────────────────
void gemv_f16_avx2(float* __restrict__       y,
                    const uint16_t* __restrict__ W,
                    const float*    __restrict__ x,
                    const float*    __restrict__ b,
                    int out, int in) noexcept {
    gemv_f16_avx2_range(y, W, x, b, 0, out, in);
}

} // namespace axonforge::cpu_x86
