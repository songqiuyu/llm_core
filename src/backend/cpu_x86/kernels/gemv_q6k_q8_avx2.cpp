// ============================================================
// gemv_q6k_q8_avx2.cpp  —  Q6_K × Q8_0 GEMV  (AVX2)
//
// Compiled with:  -mavx2 -mfma -mf16c -O3
//
// Q6_K raw quant values are uint6 in [0,63].  Weight formula:
//   w = d * scale * (q6 - 32)
// Equivalently:
//   w * x = d * scale * (q6 * x - 32 * x)
// With Q8_0 input  x ≈ q8.d * q8_s8:
//   dot = d * scale * q8.d * [dot(q6_u8, q8_s8) - 32 * sum(q8_s8)]
//
// This lets us use maddubs_epi16(q6_uint8, q8_int8) for the unsigned×signed
// integer dot product, exactly as in the Q4_K kernel.
//
// Q6_K block layout (210 bytes / 256 elements):
//   [0:127]   ql[128]    lower 4 bits of 6-bit value
//   [128:191] qh[64]     upper 2 bits (4 values packed per byte)
//   [192:207] scales[16] int8, one per 16-element group (16 groups)
//   [208:209] d (f16)    super-block scale
//
// Per 256-element block: 2 halves of 128 elements.
// Per half: 4 streams of 32 elements each, sharing ql/qh pointers.
// Scale grouping: 16 elements per scale → 8 scales per half.
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include "q8_block.hpp"

namespace axonforge::cpu_x86 {

static constexpr int QK6_   = 256;
static constexpr int Q6K_B_ = 210;

static inline float f16_q6_(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(unsigned)h)));
}

static inline float hsum256_q6_(const __m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
}
// Returns 8 floats where:
//   positions 0,1,4,5 = partial sums from q8[0..15]  (use scale sc0)
//   positions 2,3,6,7 = partial sums from q8[16..31] (use scale sc1)
// Must be multiplied by scale_corr = [sc0,sc0,sc1,sc1, sc0,sc0,sc1,sc1]
static inline __m256 q8_sum_hadd_f32_q6_(const __m256i q8v, const __m256i ones) noexcept {
    const __m256i lo_e16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
    const __m256i hi_e16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1));
    const __m256i lo_sum2 = _mm256_madd_epi16(lo_e16, ones);
    const __m256i hi_sum2 = _mm256_madd_epi16(hi_e16, ones);
    return _mm256_cvtepi32_ps(_mm256_hadd_epi32(lo_sum2, hi_sum2));
}

// ── Reconstruct 32 Q6_K raw uint6 values (range [0,63]) into a __m256i ────────
// Reads 32 bytes of ql (4-bit lower) + 8 bytes of qh (2-bit upper).
// stream: 0→ql[l]&0F | qh[l]&3<<4,  1→ql[l+32]&0F | (qh[l]>>2)&3<<4,
//         2→ql[l]>>4  | (qh[l]>>4)&3<<4, 3→ql[l+32]>>4 | (qh[l]>>6)&3<<4
// For l in [0..31] across one half of the block.
// We load 32 ql bytes and 32 qh bytes (though only the lower 8 are needed).
static inline __m256i recon_q6_stream_(
        const uint8_t* ql32,   // pointer to 32 lower-nibble bytes
        const uint8_t* qh8,    // pointer to 8 qh bytes covering 32 elements
        int stream) noexcept {
    // Load
    const __m256i ql = _mm256_loadu_si256((const __m256i*)ql32);
    const __m256i qh = _mm256_loadu_si256((const __m256i*)qh8);  // 32 bytes

    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    const __m256i mask03 = _mm256_set1_epi8(0x03);
    const __m256i x3F   = _mm256_set1_epi8(0x3F);

    __m256i lo4, hi2;
    switch (stream & 3) {
    default:
    case 0: // ql[l] & 0x0F, (qh[l]>>0)&3
        lo4 = _mm256_and_si256(ql, mask0F);
        hi2 = _mm256_and_si256(qh, mask03);
        break;
    case 1: // ql[l+32] & 0x0F, (qh[l]>>2)&3
        // ql+32 lives in the upper 128 bits of ql (since ql was 32 bytes)
        // But the pointer already points to ql[l+32], so ql has those 32 bytes.
        lo4 = _mm256_and_si256(ql, mask0F);
        hi2 = _mm256_and_si256(_mm256_srli_epi16(qh, 2), mask03);
        break;
    case 2: // ql[l] >> 4, (qh[l]>>4)&3
        lo4 = _mm256_srli_epi16(_mm256_and_si256(ql, _mm256_set1_epi8((int8_t)0xF0)), 4);
        hi2 = _mm256_and_si256(_mm256_srli_epi16(qh, 4), mask03);
        break;
    case 3: // ql[l+32] >> 4, (qh[l]>>6)&3
        lo4 = _mm256_srli_epi16(_mm256_and_si256(ql, _mm256_set1_epi8((int8_t)0xF0)), 4);
        hi2 = _mm256_and_si256(_mm256_srli_epi16(qh, 6), mask03);
        break;
    }
    return _mm256_and_si256(_mm256_or_si256(lo4, _mm256_slli_epi16(hi2, 4)), x3F);
}

// ── Single-row Q6_K × Q8_0 dot product ───────────────────────────────────────
// dot = sum_b sum_group [ d * sc_g * q8_g.d * (dot(q6_u8, q8_s8) - 32 * sum(q8_s8)) ]
static float q6k_dot_q8_row_(const uint8_t* row,
                               const block_q8_0* xq8,
                               int in_dim) noexcept {
    __m256 acc = _mm256_setzero_ps();
    const int nb = in_dim / QK6_;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + (size_t)b * Q6K_B_;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk + 208, 2);
        const float fd = f16_q6_(d_bits);
        const int8_t* sc_ptr = reinterpret_cast<const int8_t*>(blk + 192);
        const block_q8_0* xb = xq8 + (size_t)b * 8;
        for (int half = 0; half < 2; half++) {
            const uint8_t* ql_h = blk        + half*64;
            const uint8_t* qh_h = blk + 128  + half*32;
            const int8_t*  sc_h = sc_ptr      + half*8;
            for (int s = 0; s < 4; s++) {
                const __m256i q6v = recon_q6_stream_(ql_h+(s&1)*32, qh_h, s);
                const int qb_idx  = half*4 + s;
                const __m256i q8v = _mm256_loadu_si256((const __m256i*)xb[qb_idx].qs);
                const float q8d   = xb[qb_idx].d;
                const float sc0   = fd * (float)sc_h[s*2];
                const float sc1   = fd * (float)sc_h[s*2+1];
                // dot: pos 0-3 = elems 0-15 (sc0), pos 4-7 = elems 16-31 (sc1)
                const __m256 scale_dot  = _mm256_set_ps(sc1,sc1,sc1,sc1, sc0,sc0,sc0,sc0);
                // sum hadd: pos 0,1,4,5 from lo (sc0); pos 2,3,6,7 from hi (sc1)
                const __m256 scale_corr = _mm256_set_ps(sc1,sc1,sc0,sc0, sc1,sc1,sc0,sc0);
                const __m256 dot_vec  = _mm256_cvtepi32_ps(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(q6v, q8v), ones));
                const __m256 sum_hadd = q8_sum_hadd_f32_q6_(q8v, ones);
                const __m256 q8d_v   = _mm256_set1_ps(q8d);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(scale_dot, q8d_v), dot_vec, acc);
                acc = _mm256_fnmadd_ps(
                    _mm256_mul_ps(_mm256_mul_ps(scale_corr, q8d_v), _mm256_set1_ps(32.f)),
                    sum_hadd, acc);
            }
        }
    }
    return hsum256_q6_(acc);
}

// ── 4-row Q6_K × Q8_0 dot product ────────────────────────────────────────────
static void q6k_4rows_q8_(float* y, int o,
                            const uint8_t* W, size_t row_bytes,
                            const block_q8_0* xq8, int in_dim) noexcept {
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
    __m256 acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
    __m256* accs[4] = {&acc0,&acc1,&acc2,&acc3};
    const int nb = in_dim / QK6_;
    const __m256i ones = _mm256_set1_epi16(1);
    const uint8_t* r[4] = {
        W+(size_t)(o+0)*row_bytes, W+(size_t)(o+1)*row_bytes,
        W+(size_t)(o+2)*row_bytes, W+(size_t)(o+3)*row_bytes
    };
    for (int b = 0; b < nb; b++) {
        const block_q8_0* xb = xq8 + (size_t)b * 8;
        for (int half = 0; half < 2; half++) {
            for (int s = 0; s < 4; s++) {
                const int qb_idx  = half*4 + s;
                const __m256i q8v = _mm256_loadu_si256((const __m256i*)xb[qb_idx].qs);
                const __m256  sum_hadd = q8_sum_hadd_f32_q6_(q8v, ones);
                const float   q8d     = xb[qb_idx].d;
                const __m256  q8d_v   = _mm256_set1_ps(q8d);
                const __m256  c32     = _mm256_set1_ps(32.f);
                for (int ri = 0; ri < 4; ri++) {
                    const uint8_t* blk = r[ri] + (size_t)b * Q6K_B_;
                    const uint8_t* ql_h = blk       + half*64;
                    const uint8_t* qh_h = blk + 128 + half*32;
                    const int8_t*  sc_h = reinterpret_cast<const int8_t*>(blk+192) + half*8;
                    uint16_t d_bits; std::memcpy(&d_bits, blk+208, 2);
                    const float fd  = f16_q6_(d_bits);
                    const float sc0 = fd*(float)sc_h[s*2];
                    const float sc1 = fd*(float)sc_h[s*2+1];
                    const __m256i q6v = recon_q6_stream_(ql_h+(s&1)*32, qh_h, s);
                    const __m256 dot_vec    = _mm256_cvtepi32_ps(
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6v, q8v), ones));
                    const __m256 scale_dot  = _mm256_set_ps(sc1,sc1,sc1,sc1, sc0,sc0,sc0,sc0);
                    const __m256 scale_corr = _mm256_set_ps(sc1,sc1,sc0,sc0, sc1,sc1,sc0,sc0);
                    *accs[ri] = _mm256_fmadd_ps(_mm256_mul_ps(scale_dot, q8d_v), dot_vec, *accs[ri]);
                    *accs[ri] = _mm256_fnmadd_ps(
                        _mm256_mul_ps(_mm256_mul_ps(scale_corr, q8d_v), c32), sum_hadd, *accs[ri]);
                }
            }
        }
    }
    y[o+0]=hsum256_q6_(acc0); y[o+1]=hsum256_q6_(acc1);
    y[o+2]=hsum256_q6_(acc2); y[o+3]=hsum256_q6_(acc3);
}

// ── GEMV range (single token, called by ThreadPool workers) ──────────────────
void gemv_q6k_q8_avx2_range(float* __restrict__ y,
                              const uint8_t* __restrict__ W,
                              const block_q8_0* __restrict__ xq8,
                              int o_start, int o_end, int in_dim,
                              size_t row_bytes) noexcept {
    int o = o_start;
    for (; o + 4 <= o_end; o += 4)
        q6k_4rows_q8_(y, o, W, row_bytes, xq8, in_dim);
    for (; o < o_end; o++)
        y[o] = q6k_dot_q8_row_(W + (size_t)o * row_bytes, xq8, in_dim);
}

} // namespace axonforge::cpu_x86
