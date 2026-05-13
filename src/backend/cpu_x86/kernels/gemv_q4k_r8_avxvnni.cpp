// ============================================================
// gemv_q4k_r8_avxvnni.cpp  —  Q4_K_M × Q8_0 GEMV  (8-row repacked, AVX-VNNI)
//
// Compiled with:  -mavx2 -mfma -mf16c -mavxvnni -O3
//
// Identical to the AVX2 version except the dot product uses dpbusd
// (single vpdpbusd instruction, CPI=0.5) instead of maddubs_epi16 +
// madd_epi16 (two instructions, CPI=0.5+0.5).
//
// Repacked format: same 1152-byte layout as gemv_q4k_r8_avx2.cpp
//   [  0.. 15]  d[8]      fp16 super-block scales
//   [ 16.. 31]  dmin[8]   fp16 min-scales
//   [ 32..127]  scales[96] packed 6-bit sub-block scales, row-sequential
//   [128..1151] qs[1024]  nibbles, 32-byte-chunk interleaved
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include "q8_block.hpp"

namespace axonforge::cpu_x86 {

static constexpr int QK4R8V_  = 256;
static constexpr int Q4K_R8BV = 1152;

// ── hsum __m256 → scalar float ────────────────────────────────────────────────
static inline float hsum256_r8v_(const __m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
}

// ── nib_u8[32] × q8_s8[32] → __m256  (AVX-VNNI dpbusd: 1 instruction) ────────────
static inline __m256 nib_dot_r8v_(const __m256i nib, const __m256i q8v) noexcept {
    return _mm256_cvtepi32_ps(
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), nib, q8v));
}

// ── Decode 6-bit packed Q4K sub-block scales for one row (shared with AVX2) ──────
static inline float f16r8v_(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(unsigned)h)));
}
static inline void decode_scales_r8v_(const uint8_t* sc,
                                       float fd, float fdm,
                                       float* ds, float* ms) noexcept {
    uint8_t dr[8], mr[8];
    dr[0]=sc[0]&63;  mr[0]=sc[4]&63;
    dr[1]=sc[1]&63;  mr[1]=sc[5]&63;
    dr[2]=sc[2]&63;  mr[2]=sc[6]&63;
    dr[3]=sc[3]&63;  mr[3]=sc[7]&63;
    dr[4]=(sc[8] &15)|((sc[0]>>6)<<4); mr[4]=(sc[8] >>4)|((sc[4]>>6)<<4);
    dr[5]=(sc[9] &15)|((sc[1]>>6)<<4); mr[5]=(sc[9] >>4)|((sc[5]>>6)<<4);
    dr[6]=(sc[10]&15)|((sc[2]>>6)<<4); mr[6]=(sc[10]>>4)|((sc[6]>>6)<<4);
    dr[7]=(sc[11]&15)|((sc[3]>>6)<<4); mr[7]=(sc[11]>>4)|((sc[7]>>6)<<4);
    const __m128i di = _mm_loadl_epi64((const __m128i*)dr);
    const __m128i mi = _mm_loadl_epi64((const __m128i*)mr);
    _mm256_storeu_ps(ds, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(di)),
                                        _mm256_set1_ps(fd)));
    _mm256_storeu_ps(ms, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(mi)),
                                        _mm256_set1_ps(fdm)));
}

// ── sum(q8_s8[32]) → __m256 (8 partial int32 sums → float, reused across 8 rows) ────
static inline __m256 q8_sum_f32_r8v_(const __m256i q8v) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m128i lo = _mm256_castsi256_si128(q8v);
    const __m128i hi = _mm256_extracti128_si256(q8v, 1);
    const __m256i e16 = _mm256_add_epi16(_mm256_cvtepi8_epi16(lo),
                                          _mm256_cvtepi8_epi16(hi));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(e16, ones));
}

// ─── 8-row Q4K_M × Q8_0 GEMV over one row-group range (AVX-VNNI) ─────────────
static void q4k_8rows_r8v_(float* __restrict__ y,
                             int g_start, int g_end,
                             const uint8_t* __restrict__ R8,
                             int nb,
                             const block_q8_0* __restrict__ xq8) noexcept {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);

    for (int g = g_start; g < g_end; g++) {
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = R8 + ((size_t)g * nb + b) * Q4K_R8BV;

            if (b + 1 < nb) {
                const char* np = (const char*)(blk + Q4K_R8BV);
                _mm_prefetch(np,        _MM_HINT_T0);
                _mm_prefetch(np +   64, _MM_HINT_T0);
                _mm_prefetch(np +  128, _MM_HINT_T0);
                _mm_prefetch(np +  192, _MM_HINT_T0);
                _mm_prefetch(np +  256, _MM_HINT_T0);
                _mm_prefetch(np +  320, _MM_HINT_T0);
                _mm_prefetch(np +  384, _MM_HINT_T0);
                _mm_prefetch(np +  448, _MM_HINT_T0);
                _mm_prefetch(np +  512, _MM_HINT_T0);
                _mm_prefetch(np +  576, _MM_HINT_T0);
                _mm_prefetch(np +  640, _MM_HINT_T0);
                _mm_prefetch(np +  704, _MM_HINT_T0);
                _mm_prefetch(np +  768, _MM_HINT_T0);
                _mm_prefetch(np +  832, _MM_HINT_T0);
                _mm_prefetch(np +  896, _MM_HINT_T0);
                _mm_prefetch(np +  960, _MM_HINT_T0);
                _mm_prefetch(np + 1024, _MM_HINT_T0);
                _mm_prefetch(np + 1088, _MM_HINT_T0);
            }

            // Load d[8] and dmin[8] (fp16 → f32 via F16C)
            float d_f32[8], dm_f32[8];
            _mm256_storeu_ps(d_f32,
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)blk)));
            _mm256_storeu_ps(dm_f32,
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(blk + 16))));
            // Decode sub-block scales once per block (amortised over 4 chunks)
            const uint8_t* sc = blk + 32;
            float ds[8][8], ms[8][8];
            for (int r = 0; r < 8; r++)
                decode_scales_r8v_(sc + r*12, d_f32[r], dm_f32[r], ds[r], ms[r]);

            const uint8_t* qs_base = blk + 128;

            for (int chunk = 0; chunk < 4; chunk++) {
                const int g0 = chunk*2, g1 = chunk*2+1;
                const block_q8_0* xb = xq8 + (size_t)b * 8;

                const __m256i q8v0 = _mm256_loadu_si256((const __m256i*)xb[g0].qs);
                const __m256i q8v1 = _mm256_loadu_si256((const __m256i*)xb[g1].qs);
                const float d0 = xb[g0].d, d1 = xb[g1].d;
                const __m256 sum0f = q8_sum_f32_r8v_(q8v0);
                const __m256 sum1f = q8_sum_f32_r8v_(q8v1);

                const uint8_t* cqs = qs_base + chunk * 256;

#define ACCUM_R8V_(r_, acc_)  do {                                               \
    const __m256i qv = _mm256_loadu_si256((const __m256i*)(cqs + (r_)*32));    \
    const __m256i lo = _mm256_and_si256(qv, mask4);                              \
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), mask4);      \
    (acc_) = _mm256_fmadd_ps(_mm256_set1_ps(ds[r_][g0]*d0),                     \
                              nib_dot_r8v_(lo, q8v0), (acc_));                   \
    (acc_) = _mm256_fnmadd_ps(_mm256_set1_ps(ms[r_][g0]*d0), sum0f, (acc_));   \
    (acc_) = _mm256_fmadd_ps(_mm256_set1_ps(ds[r_][g1]*d1),                     \
                              nib_dot_r8v_(hi, q8v1), (acc_));                   \
    (acc_) = _mm256_fnmadd_ps(_mm256_set1_ps(ms[r_][g1]*d1), sum1f, (acc_));   \
} while(0)

                ACCUM_R8V_(0, acc0);
                ACCUM_R8V_(1, acc1);
                ACCUM_R8V_(2, acc2);
                ACCUM_R8V_(3, acc3);
                ACCUM_R8V_(4, acc4);
                ACCUM_R8V_(5, acc5);
                ACCUM_R8V_(6, acc6);
                ACCUM_R8V_(7, acc7);
#undef ACCUM_R8V_
            }
        }

        const int base = g * 8;
        y[base+0] = hsum256_r8v_(acc0);
        y[base+1] = hsum256_r8v_(acc1);
        y[base+2] = hsum256_r8v_(acc2);
        y[base+3] = hsum256_r8v_(acc3);
        y[base+4] = hsum256_r8v_(acc4);
        y[base+5] = hsum256_r8v_(acc5);
        y[base+6] = hsum256_r8v_(acc6);
        y[base+7] = hsum256_r8v_(acc7);
    }
}

// ─── Public: GEMV range, repacked format (AVX-VNNI) ──────────────────────────
void gemv_q4k_r8_avxvnni_range(float* __restrict__ y,
                                 const uint8_t* __restrict__ R8,
                                 const block_q8_0* __restrict__ xq8,
                                 int o_start, int o_end, int in_dim) noexcept {
    const int nb = in_dim / QK4R8V_;
    q4k_8rows_r8v_(y, o_start / 8, o_end / 8, R8, nb, xq8);
}

}  // namespace axonforge::cpu_x86
