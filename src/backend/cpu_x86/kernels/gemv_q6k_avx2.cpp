// ============================================================
// gemv_q6k_avx2.cpp — Q6_K-weight × F32-activation GEMV
//
// Compiled with:  -mavx2 -mfma -mf16c -O3
//
// Q6_K block layout (210 bytes / 256 elements):
//   [0:127]   ql[128]    lower 4 bits of each 6-bit value
//   [128:191] qh[64]     upper 2 bits (packed 4-per-byte)
//   [192:207] scales[16] int8, one per 16-element group
//   [208:209] d (f16)    super-block scale
//
// Per 256-element block: 2 halves of 128 elements.
// Per half: ql[0..63], qh[0..31], sc[0..7] (int8).
//   Four streams per half (32 elements each → x offsets 0, 32, 64, 96):
//     q1[l] = (ql[l]    & 0x0F) | ((qh[l] & 0x03)        << 4)  → x[l+ 0]
//     q2[l] = (ql[l+32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)  → x[l+32]
//     q3[l] = (ql[l]    >>   4) | (((qh[l] >> 4) & 0x03) << 4)  → x[l+64]
//     q4[l] = (ql[l+32] >>   4) | (((qh[l] >> 6) & 0x03) << 4)  → x[l+96]
//   For l in [0..31]; scale changes at l=16:
//     l in [0..15]  → scale = d * sc[0/2/4/6]
//     l in [16..31] → scale = d * sc[1/3/5/7]
//
// AVX2 strategy:
//   • Load 32 bytes of ql_lo/ql_hi/qh in one _mm256_loadu_si256 each.
//   • Reconstruct each stream with AND + srli_epi16 + slli_epi16.
//   • Split each 32-element stream into two __m128i halves (16 elems each),
//     applying the two different scales via fmsub_ps + fmadd_ps.
//   • Two independent accumulators hide 4-cycle FMA latency.
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace axonforge::cpu_x86 {

static constexpr int QK_K_Q6_  = 256;
static constexpr int Q6K_BYTES_ = 210;

// ── F16→F32 via F16C ─────────────────────────────────────────────────────────
static inline float f16c_q6_(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(unsigned)h)));
}

// ── Horizontal sum of __m256 ─────────────────────────────────────────────────
static inline float hsum256_q6_(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ── Process 32 reconstructed uint8 quant values × 32 f32 activations ─────────
// q: 32 raw uint8 values (range 0..63).
// x: pointer to 32 f32 activation values.
// scale0: effective scale for elements 0..15;  scale1: for elements 16..31.
// weight formula: d*sc*(q_raw - 32) = scale*q_raw - scale*32
static inline void accum_stream_q6_(const __m256i q, const float* x,
                                     float scale0, float scale1,
                                     __m256& acc0, __m256& acc1) noexcept {
    const __m128i q_lo = _mm256_castsi256_si128(q);          // elements 0..15
    const __m128i q_hi = _mm256_extracti128_si256(q, 1);     // elements 16..31

    const __m256 s0 = _mm256_set1_ps(scale0);
    const __m256 b0 = _mm256_set1_ps(scale0 * 32.f);
    const __m256 s1 = _mm256_set1_ps(scale1);
    const __m256 b1 = _mm256_set1_ps(scale1 * 32.f);

    // Elements 0..7: scale0
    const __m256 qf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_lo));
    acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(s0, qf0, b0), _mm256_loadu_ps(x +  0), acc0);
    // Elements 8..15: scale0
    const __m256 qf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(q_lo, 8)));
    acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(s0, qf1, b0), _mm256_loadu_ps(x +  8), acc1);
    // Elements 16..23: scale1
    const __m256 qf2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_hi));
    acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(s1, qf2, b1), _mm256_loadu_ps(x + 16), acc0);
    // Elements 24..31: scale1
    const __m256 qf3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(q_hi, 8)));
    acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(s1, qf3, b1), _mm256_loadu_ps(x + 24), acc1);
}

// ── Single-row Q6K dot product ────────────────────────────────────────────────
float q6k_dot_row_avx2(const uint8_t* row, const float* x, int in_dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const int nb = in_dim / QK_K_Q6_;
    const __m256i mask_0F = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block   = row + (size_t)b * Q6K_BYTES_;
        const uint8_t* ql_base = block;           // 128 bytes
        const uint8_t* qh_base = block + 128;     // 64 bytes
        const int8_t*  sc_base = reinterpret_cast<const int8_t*>(block + 192); // 16 int8
        uint16_t d_bits;
        std::memcpy(&d_bits, block + 208, 2);
        const float d    = f16c_q6_(d_bits);
        const float* xb  = x + (size_t)b * QK_K_Q6_;

        for (int n = 0; n < 2; n++) {
            const uint8_t* ql_ptr = ql_base + (size_t)n * 64;
            const uint8_t* qh_ptr = qh_base + (size_t)n * 32;
            const int8_t*  sc_ptr = sc_base  + n * 8;
            const float*   xp     = xb       + n * 128;

            // Load 32 bytes of ql_lo (indices 0..31) and ql_hi (32..63) and qh
            const __m256i vql_lo = _mm256_loadu_si256((const __m256i*)ql_ptr);
            const __m256i vql_hi = _mm256_loadu_si256((const __m256i*)(ql_ptr + 32));
            const __m256i vqh    = _mm256_loadu_si256((const __m256i*)qh_ptr);

            // ── Reconstruct 4 streams (each byte = 6-bit raw value 0..63) ────
            // All srli_epi16 shifts are followed by an AND mask to discard bits
            // that leaked across byte boundaries within the 16-bit lane.
            //
            // q1: bits[3:0] of ql_lo  |  bits[1:0] of qh  << 4
            const __m256i q1 = _mm256_or_si256(
                _mm256_and_si256(vql_lo, mask_0F),
                _mm256_slli_epi16(_mm256_and_si256(vqh, mask_03), 4)
            );
            // q2: bits[3:0] of ql_hi  |  bits[3:2] of qh  << 4
            const __m256i q2 = _mm256_or_si256(
                _mm256_and_si256(vql_hi, mask_0F),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(vqh, 2), mask_03), 4)
            );
            // q3: bits[7:4] of ql_lo  |  bits[5:4] of qh  << 4
            const __m256i q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(vql_lo, 4), mask_0F),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(vqh, 4), mask_03), 4)
            );
            // q4: bits[7:4] of ql_hi  |  bits[7:6] of qh  << 4
            const __m256i q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(vql_hi, 4), mask_0F),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(vqh, 6), mask_03), 4)
            );

            // ── Accumulate 4 streams × 32 elements against x ─────────────────
            // Scale indices per GGML spec (is = l/16, streams at sc[0..7]):
            //   q1 → sc[0..1]  q2 → sc[2..3]  q3 → sc[4..5]  q4 → sc[6..7]
            accum_stream_q6_(q1, xp +   0, d * (float)sc_ptr[0], d * (float)sc_ptr[1], acc0, acc1);
            accum_stream_q6_(q2, xp +  32, d * (float)sc_ptr[2], d * (float)sc_ptr[3], acc0, acc1);
            accum_stream_q6_(q3, xp +  64, d * (float)sc_ptr[4], d * (float)sc_ptr[5], acc0, acc1);
            accum_stream_q6_(q4, xp +  96, d * (float)sc_ptr[6], d * (float)sc_ptr[7], acc0, acc1);
        }
    }
    return hsum256_q6_(_mm256_add_ps(acc0, acc1));
}

// ── Range GEMV — called by ThreadPool workers ─────────────────────────────────
void gemv_q6k_avx2_range(float* y, const uint8_t* W, const float* x,
                          int o_start, int o_end, int in_dim,
                          size_t row_bytes) noexcept {
    for (int o = o_start; o < o_end; o++)
        y[o] = q6k_dot_row_avx2(W + (size_t)o * row_bytes, x, in_dim);
}

} // namespace axonforge::cpu_x86
