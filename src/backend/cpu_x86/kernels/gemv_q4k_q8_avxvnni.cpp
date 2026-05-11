// ============================================================
// gemv_q4k_q8_avxvnni.cpp  —  Q4_K_M × Q8_0 GEMV + GEMM  (AVX-VNNI)
//
// Compiled with:  -mavx2 -mfma -mf16c -mavxvnni -O3
//
// Key difference from the AVX2 version:
//   _mm256_dpbusd_epi32(zero, nib_u8, q8_s8)
//   replaces the two-instruction sequence:
//   _mm256_madd_epi16(_mm256_maddubs_epi16(nib, q8), ones)
//   saving one vector operation per 32-element dot product.
//
// GEMV  (single token):   gemv_q4k_q8_avxvnni_range()
// GEMM  (T tokens):       gemm_q4k_q8_avxvnni_range()
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "q8_block.hpp"

namespace axonforge::cpu_x86 {

static constexpr int QK4_   = 256;
static constexpr int Q4K_B_ = 144;

static inline float f16_v_(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(unsigned)h)));
}

static inline void get_scale_min_v_(int j, const uint8_t* sc,
                                     uint8_t& d, uint8_t& m) noexcept {
    if (j < 4) {
        d = sc[j]   & 0x3F;
        m = sc[j+4] & 0x3F;
    } else {
        d = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
        m = (sc[j+4] >>   4) | ((sc[j  ] >> 6) << 4);
    }
}

// ── Horizontal sum: __m256i epi32 → int32 ────────────────────────────────────
static inline int32_t hsum_epi32_v_(const __m256i v) noexcept {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(lo, lo));
}

// ── Horizontal sum: __m256 → float ───────────────────────────────────────────
static inline float hsum256_v_(const __m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
}

// ── AVX-VNNI dot: nib_u8[32] × q8_s8[32] → __m256 (8 int32 → 8 float) ──────
// Replaces maddubs_epi16 + madd_epi16 with a single dpbusd instruction.
static inline __m256 nib_q8_dot_f32_v_(const __m256i nib,
                                        const __m256i q8v) noexcept {
    return _mm256_cvtepi32_ps(
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), nib, q8v));
}

// ── Scalar version (used in GEMM inner loop) ──────────────────────────────────
static inline int32_t dot_nib_q8_v_(const __m256i nib,
                                     const __m256i q8v) noexcept {
    return hsum_epi32_v_(
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), nib, q8v));
}

// ── sum(q8_s8[32]) as __m256 ─────────────────────────────────────────────────
static inline __m256 q8_sum_f32_v_(const __m256i q8v,
                                    const __m256i ones) noexcept {
    const __m128i lo = _mm256_castsi256_si128(q8v);
    const __m128i hi = _mm256_extracti128_si256(q8v, 1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(
        _mm256_add_epi16(_mm256_cvtepi8_epi16(lo), _mm256_cvtepi8_epi16(hi)),
        ones));
}

// ── Scalar sum (used in GEMM inner loop) ─────────────────────────────────────
static inline int32_t sum_q8_v_(const __m256i q8v,
                                  const __m256i ones) noexcept {
    const __m128i lo = _mm256_castsi256_si128(q8v);
    const __m128i hi = _mm256_extracti128_si256(q8v, 1);
    return hsum_epi32_v_(_mm256_madd_epi16(
        _mm256_add_epi16(_mm256_cvtepi8_epi16(lo), _mm256_cvtepi8_epi16(hi)),
        ones));
}

// ── Scale/min precomputation for one Q4_K super-block row ────────────────────
static inline void load_scales_v_(const uint8_t* blk,
                                   float* ds, float* ms) noexcept {
    uint16_t d_bits, dm_bits;
    std::memcpy(&d_bits,  blk,   2);
    std::memcpy(&dm_bits, blk+2, 2);
    const float fd = f16_v_(d_bits), fdm = f16_v_(dm_bits);
    for (int j = 0; j < 8; j++) {
        uint8_t sc, mn;
        get_scale_min_v_(j, blk+4, sc, mn);
        ds[j] = fd  * (float)sc;
        ms[j] = fdm * (float)mn;
    }
}

// ── Single-row Q4_K × Q8_0 dot product (vector acc, 1 hsum/row, AVX-VNNI) ───
static float q4k_dot_q8_row_v_(const uint8_t* row,
                                 const block_q8_0* xq8,
                                 int in_dim) noexcept {
    __m256 acc = _mm256_setzero_ps();
    const int nb = in_dim / QK4_;
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i ones  = _mm256_set1_epi16(1);

    for (int b = 0; b < nb; b++) {
        const uint8_t*    blk = row  + (size_t)b * Q4K_B_;
        const block_q8_0* xb  = xq8 + (size_t)b * 8;

        float ds[8], ms[8];
        load_scales_v_(blk, ds, ms);

        const uint8_t* qs = blk + 16;
        for (int chunk = 0; chunk < 4; chunk++) {
            const int g0 = chunk * 2, g1 = chunk * 2 + 1;

            const __m256i qvec = _mm256_loadu_si256((const __m256i*)(qs + chunk*32));
            const __m256i lo   = _mm256_and_si256(qvec, mask4);
            const __m256i hi   = _mm256_and_si256(_mm256_srli_epi16(qvec, 4), mask4);

            const __m256i q8v0 = _mm256_loadu_si256((const __m256i*)xb[g0].qs);
            const __m256i q8v1 = _mm256_loadu_si256((const __m256i*)xb[g1].qs);
            const float d0 = xb[g0].d, d1 = xb[g1].d;

            acc = _mm256_fmadd_ps(_mm256_set1_ps(ds[g0]*d0), nib_q8_dot_f32_v_(lo,  q8v0), acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(ms[g0]*d0), q8_sum_f32_v_(q8v0, ones), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(ds[g1]*d1), nib_q8_dot_f32_v_(hi,  q8v1), acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(ms[g1]*d1), q8_sum_f32_v_(q8v1, ones), acc);
        }
    }
    return hsum256_v_(acc);
}

// ── 4-row Q4_K × Q8_0 (vector float accumulators, AVX-VNNI dot) ──────────────
static void q4k_4rows_q8_v_(float* y, int o,
                              const uint8_t* W, size_t row_bytes,
                              const block_q8_0* xq8, int in_dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const int nb = in_dim / QK4_;
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i ones  = _mm256_set1_epi16(1);

    const uint8_t* r0 = W + (size_t)(o+0) * row_bytes;
    const uint8_t* r1 = W + (size_t)(o+1) * row_bytes;
    const uint8_t* r2 = W + (size_t)(o+2) * row_bytes;
    const uint8_t* r3 = W + (size_t)(o+3) * row_bytes;

    for (int b = 0; b < nb; b++) {
        const block_q8_0* xb = xq8 + (size_t)b * 8;

        float ds0[8],ms0[8], ds1[8],ms1[8], ds2[8],ms2[8], ds3[8],ms3[8];
        load_scales_v_(r0+(size_t)b*Q4K_B_, ds0, ms0);
        load_scales_v_(r1+(size_t)b*Q4K_B_, ds1, ms1);
        load_scales_v_(r2+(size_t)b*Q4K_B_, ds2, ms2);
        load_scales_v_(r3+(size_t)b*Q4K_B_, ds3, ms3);

        const uint8_t* qs0 = r0+(size_t)b*Q4K_B_+16;
        const uint8_t* qs1 = r1+(size_t)b*Q4K_B_+16;
        const uint8_t* qs2 = r2+(size_t)b*Q4K_B_+16;
        const uint8_t* qs3 = r3+(size_t)b*Q4K_B_+16;

        for (int chunk = 0; chunk < 4; chunk++) {
            const int g0 = chunk*2, g1 = chunk*2+1;

            const __m256i q8v0 = _mm256_loadu_si256((const __m256i*)xb[g0].qs);
            const __m256i q8v1 = _mm256_loadu_si256((const __m256i*)xb[g1].qs);
            const __m256 sum0f = q8_sum_f32_v_(q8v0, ones);
            const __m256 sum1f = q8_sum_f32_v_(q8v1, ones);
            const float d0 = xb[g0].d, d1 = xb[g1].d;

#define ACCUM4V_VNNI_(qs_, ds_, ms_, acc_)  do {                                     \
    const __m256i qv = _mm256_loadu_si256((const __m256i*)(qs_ + chunk*32));        \
    const __m256i lo = _mm256_and_si256(qv, mask4);                                  \
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), mask4);           \
    acc_ = _mm256_fmadd_ps(_mm256_set1_ps(ds_[g0]*d0), nib_q8_dot_f32_v_(lo,q8v0), acc_); \
    acc_ = _mm256_fnmadd_ps(_mm256_set1_ps(ms_[g0]*d0), sum0f, acc_);               \
    acc_ = _mm256_fmadd_ps(_mm256_set1_ps(ds_[g1]*d1), nib_q8_dot_f32_v_(hi,q8v1), acc_); \
    acc_ = _mm256_fnmadd_ps(_mm256_set1_ps(ms_[g1]*d1), sum1f, acc_);               \
} while(0)
            ACCUM4V_VNNI_(qs0, ds0, ms0, acc0);
            ACCUM4V_VNNI_(qs1, ds1, ms1, acc1);
            ACCUM4V_VNNI_(qs2, ds2, ms2, acc2);
            ACCUM4V_VNNI_(qs3, ds3, ms3, acc3);
#undef ACCUM4V_VNNI_
        }
    }
    y[o+0] = hsum256_v_(acc0);  y[o+1] = hsum256_v_(acc1);
    y[o+2] = hsum256_v_(acc2);  y[o+3] = hsum256_v_(acc3);
}

// ── GEMV range (single token, AVX-VNNI) ──────────────────────────────────────
void gemv_q4k_q8_avxvnni_range(float* __restrict__ y,
                                 const uint8_t* __restrict__ W,
                                 const block_q8_0* __restrict__ xq8,
                                 int o_start, int o_end, int in_dim,
                                 size_t row_bytes) noexcept {
    int o = o_start;
    for (; o + 4 <= o_end; o += 4)
        q4k_4rows_q8_v_(y, o, W, row_bytes, xq8, in_dim);
    for (; o < o_end; o++)
        y[o] = q4k_dot_q8_row_v_(W + (size_t)o * row_bytes, xq8, in_dim);
}

// ── GEMM range (T tokens, AVX-VNNI) ──────────────────────────────────────────
void gemm_q4k_q8_avxvnni_range(float* __restrict__ Y,
                                 const uint8_t* __restrict__ W,
                                 const block_q8_0* __restrict__ XQ8,
                                 int M, int T, int K, size_t row_bytes,
                                 int o_start, int o_end) noexcept {
    const int nb               = K / QK4_;
    const int blocks_per_token = K / 32;
    const __m256i mask4        = _mm256_set1_epi8(0x0F);
    const __m256i ones         = _mm256_set1_epi16(1);

    int o = o_start;
    for (; o + 4 <= o_end; o += 4) {
        const uint8_t* r0 = W + (size_t)(o+0)*row_bytes;
        const uint8_t* r1 = W + (size_t)(o+1)*row_bytes;
        const uint8_t* r2 = W + (size_t)(o+2)*row_bytes;
        const uint8_t* r3 = W + (size_t)(o+3)*row_bytes;

        static constexpr int TT = 8;
        for (int t0 = 0; t0 < T; t0 += TT) {
            const int nt = std::min(TT, T - t0);
            float acc[4][TT] = {};

            for (int b = 0; b < nb; b++) {
                float ds0[8],ms0[8], ds1[8],ms1[8], ds2[8],ms2[8], ds3[8],ms3[8];
                load_scales_v_(r0 + (size_t)b*Q4K_B_, ds0, ms0);
                load_scales_v_(r1 + (size_t)b*Q4K_B_, ds1, ms1);
                load_scales_v_(r2 + (size_t)b*Q4K_B_, ds2, ms2);
                load_scales_v_(r3 + (size_t)b*Q4K_B_, ds3, ms3);

                const uint8_t* qs0 = r0 + (size_t)b*Q4K_B_ + 16;
                const uint8_t* qs1 = r1 + (size_t)b*Q4K_B_ + 16;
                const uint8_t* qs2 = r2 + (size_t)b*Q4K_B_ + 16;
                const uint8_t* qs3 = r3 + (size_t)b*Q4K_B_ + 16;

                for (int chunk = 0; chunk < 4; chunk++) {
                    const int g0 = chunk*2, g1 = chunk*2+1;

                    const __m256i qv0 = _mm256_loadu_si256((const __m256i*)(qs0+chunk*32));
                    const __m256i qv1 = _mm256_loadu_si256((const __m256i*)(qs1+chunk*32));
                    const __m256i qv2 = _mm256_loadu_si256((const __m256i*)(qs2+chunk*32));
                    const __m256i qv3 = _mm256_loadu_si256((const __m256i*)(qs3+chunk*32));

                    const __m256i lo0=_mm256_and_si256(qv0,mask4), hi0=_mm256_and_si256(_mm256_srli_epi16(qv0,4),mask4);
                    const __m256i lo1=_mm256_and_si256(qv1,mask4), hi1=_mm256_and_si256(_mm256_srli_epi16(qv1,4),mask4);
                    const __m256i lo2=_mm256_and_si256(qv2,mask4), hi2=_mm256_and_si256(_mm256_srli_epi16(qv2,4),mask4);
                    const __m256i lo3=_mm256_and_si256(qv3,mask4), hi3=_mm256_and_si256(_mm256_srli_epi16(qv3,4),mask4);

                    for (int ti = 0; ti < nt; ti++) {
                        const block_q8_0* xb = XQ8 + (size_t)(t0+ti)*blocks_per_token + (size_t)b*8;
                        const __m256i q8v0 = _mm256_loadu_si256((const __m256i*)xb[g0].qs);
                        const __m256i q8v1 = _mm256_loadu_si256((const __m256i*)xb[g1].qs);
                        const int32_t sum0 = sum_q8_v_(q8v0, ones);
                        const int32_t sum1 = sum_q8_v_(q8v1, ones);
                        const float dq0 = xb[g0].d, dq1 = xb[g1].d;

                        const float contrib0 = dq0 * (float)sum0;
                        const float contrib1 = dq1 * (float)sum1;

#define GACCUM_V_(ri_, lo_, hi_)  do {                                            \
    acc[ri_][ti] +=                                                                \
         ds##ri_[g0]*dq0*(float)dot_nib_q8_v_(lo_,q8v0) - ms##ri_[g0]*contrib0  \
        +ds##ri_[g1]*dq1*(float)dot_nib_q8_v_(hi_,q8v1) - ms##ri_[g1]*contrib1; \
} while(0)
                        GACCUM_V_(0, lo0, hi0);
                        GACCUM_V_(1, lo1, hi1);
                        GACCUM_V_(2, lo2, hi2);
                        GACCUM_V_(3, lo3, hi3);
#undef GACCUM_V_
                    }
                }
            }
            for (int ri = 0; ri < 4; ri++)
                for (int ti = 0; ti < nt; ti++)
                    Y[(size_t)(t0+ti)*M + o+ri] = acc[ri][ti];
        }
    }

    for (; o < o_end; o++) {
        const uint8_t* row = W + (size_t)o * row_bytes;
        for (int t = 0; t < T; t++)
            Y[(size_t)t*M + o] = q4k_dot_q8_row_v_(row, XQ8 + (size_t)t*blocks_per_token, K);
    }
}

} // namespace axonforge::cpu_x86
