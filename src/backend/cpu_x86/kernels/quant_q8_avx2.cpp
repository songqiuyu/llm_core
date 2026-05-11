// ============================================================
// quant_q8_avx2.cpp — F32 → Q8_0 quantisation (AVX2)
//
// Compiled with:  -mavx2 -mfma -mf16c -O3
//
// One Q8_0 block = 32 int8 values + 1 F32 scale d = absmax/127.
// x[i] ≈ d * qs[i].   Reconstruction error ≤ d/2 per element.
//
// AVX2 strategy per 32-element block:
//   1. Load 4×8 F32, compute element-wise |x| → absmax (horizontal max).
//   2. scale = 127 / absmax;  d = absmax / 127.
//   3. Multiply by scale, _mm256_cvtps_epi32 (rounds to nearest).
//   4. Saturating pack: epi32→epi16 (_packs_epi32) → epi8 (_packs_epi16).
//   5. Fix cross-lane order with _mm256_permutevar8x32_epi32.
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include "q8_block.hpp"

namespace axonforge::cpu_x86 {

// ── Horizontal max of __m256 → scalar float ──────────────────────────────────
static inline float hmax256_(const __m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// ── Quantise k floats to k/32 Q8_0 blocks (k must be multiple of 32) ─────────
void quantize_q8_avx2(block_q8_0* __restrict__ y,
                       const float* __restrict__ x,
                       int k) noexcept {
    const __m256 signmask = _mm256_set1_ps(-0.f);
    // Permutation to fix the cross-lane interleaving introduced by packs
    // _mm256_packs operates in two independent 128-bit lanes:
    //   lane0: f0[0..7]→s16, f1[0..7]→s16 → bytes 0..15
    //   lane1: f2[0..7]→s16, f3[0..7]→s16 → bytes 16..31
    // After packs_epi32 + packs_epi16 the natural order (0,1,2,3 × 8 elems) is
    // split across the two 128-bit lanes, so we need to re-interleave.
    // permutevar8x32 treats the __m256i as 8 int32 dwords.
    // Desired dword order to put bytes back in original element order: 0,4,1,5,2,6,3,7
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const float* src = x + (size_t)i * 32;

        // Load 32 floats (4 AVX2 registers)
        __m256 f0 = _mm256_loadu_ps(src);
        __m256 f1 = _mm256_loadu_ps(src +  8);
        __m256 f2 = _mm256_loadu_ps(src + 16);
        __m256 f3 = _mm256_loadu_ps(src + 24);

        // Absolute value
        __m256 a0 = _mm256_andnot_ps(signmask, f0);
        __m256 a1 = _mm256_andnot_ps(signmask, f1);
        __m256 a2 = _mm256_andnot_ps(signmask, f2);
        __m256 a3 = _mm256_andnot_ps(signmask, f3);
        __m256 maxv = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
        const float amax = hmax256_(maxv);

        const float d  = amax / 127.f;
        const float id = (amax > 0.f) ? 127.f / amax : 0.f;
        y[i].d = d;

        // Quantise: round-to-nearest via _mm256_cvtps_epi32 (default rounding mode)
        const __m256 ids = _mm256_set1_ps(id);
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, ids));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, ids));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(f2, ids));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(f3, ids));

        // Saturating pack: int32 → int16 → int8
        __m256i p01 = _mm256_packs_epi32(q0, q1);   // 16 × int16
        __m256i p23 = _mm256_packs_epi32(q2, q3);
        __m256i p8  = _mm256_packs_epi16(p01, p23);  // 32 × int8 (wrong lane order)
        p8 = _mm256_permutevar8x32_epi32(p8, perm);  // fix order

        _mm256_storeu_si256((__m256i*)y[i].qs, p8);
    }
}

} // namespace axonforge::cpu_x86
