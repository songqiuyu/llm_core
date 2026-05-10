// ============================================================
// gemv_q4k_avx2.cpp — Q4_K_M-weight × F32-activation GEMV
//
// Compiled with:  -mavx2 -mfma -mf16c -O3
//
// Q4_K_M block layout (144 bytes / 256 elements):
//   [0:1]    d     (f16) super-scale for quantised scales
//   [2:3]    dmin  (f16) super-scale for quantised mins
//   [4:15]   scales[12]  8×(6-bit scale, 6-bit min) packed
//   [16:143] qs[128]     256 nibbles, 2 per byte
//
// Per 256-elem block: 4 chunks of 64 elements.
// Each chunk: 32 bytes → 64 nibbles.
//   Low  nibble of qs[l] → element l    (scale d1, min m1)
//   High nibble of qs[l] → element l+32 (scale d2, min m2)
//
// AVX2 inner loop:
//   • Load 32 bytes → 64 nibbles with two AND operations.
//   • Convert 8 uint8 nibbles → 8 f32 via cvtepu8_epi32 + cvtepi32_ps.
//   • Apply (d*sc) * nibble - (dmin*m) with fmsub_ps.
//   • Accumulate with loadu_ps + fmadd_ps.
//   Two independent accumulators (acc0/acc1) hide 4-cycle FMA latency.
// ============================================================
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace axonforge::cpu_x86 {

static constexpr int QK_K_      = 256;
static constexpr int Q4K_BYTES_ = 144;

// ── Scale/min extraction — must match GGML get_scale_min_k4 ──────────────────
static inline void get_scale_min_k4_(int j, const uint8_t* sc,
                                      uint8_t& d, uint8_t& m) noexcept {
    if (j < 4) {
        d = sc[j]   & 0x3F;
        m = sc[j+4] & 0x3F;
    } else {
        d = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
        m = (sc[j+4] >>   4) | ((sc[j  ] >> 6) << 4);
    }
}

// ── F16→F32 via F16C ─────────────────────────────────────────────────────────
static inline float f16c_(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(unsigned)h)));
}

// ── Horizontal sum of __m256 → scalar float ───────────────────────────────────
static inline float hsum256_(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ── Extract 8 uint8 values at byte_offset (0/8/16/24) from __m256i → 8×f32 ──
// Relies on _mm256_cvtepu8_epi32 reading the lowest 8 bytes of its __m128i arg.
static inline __m256 u8x8_to_f32_(const __m256i v, int byte_offset) noexcept {
    __m128i half = (byte_offset < 16)
        ? _mm256_castsi256_si128(v)
        : _mm256_extracti128_si256(v, 1);
    if (byte_offset & 8)
        half = _mm_srli_si128(half, 8);
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(half));
}

// ── Single-row Q4K dot product ────────────────────────────────────────────────
float q4k_dot_row_avx2(const uint8_t* row, const float* x, int in_dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const int nb = in_dim / QK_K_;
    const __m256i mask = _mm256_set1_epi8(0x0F);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = row + (size_t)b * Q4K_BYTES_;
        uint16_t d_bits, dm_bits;
        std::memcpy(&d_bits,  block,   2);
        std::memcpy(&dm_bits, block+2, 2);
        const float d    = f16c_(d_bits);
        const float dmin = f16c_(dm_bits);
        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;
        const float*   xb     = x + (size_t)b * QK_K_;

        // Precompute 8 effective (d*scale, dmin*min) pairs once per block,
        // hoisting 16 multiplications outside the 4-chunk × 8-group loop.
        float ds[8], ms[8];
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min_k4_(j, scales, sc, mn);
            ds[j] = d    * (float)sc;
            ms[j] = dmin * (float)mn;
        }

        for (int chunk = 0; chunk < 4; chunk++) {
            const __m256 d1  = _mm256_set1_ps(ds[chunk*2    ]);
            const __m256 mf1 = _mm256_set1_ps(ms[chunk*2    ]);
            const __m256 d2  = _mm256_set1_ps(ds[chunk*2 + 1]);
            const __m256 mf2 = _mm256_set1_ps(ms[chunk*2 + 1]);

            // Load 32 bytes (= 64 nibbles) for this chunk
            const __m256i qvec = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(qs + chunk * 32));
            // lo[i] = low  nibble of qs[i], for i in [0..31]
            const __m256i lo = _mm256_and_si256(qvec, mask);
            // hi[i] = high nibble of qs[i], for i in [0..31]
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qvec, 4), mask);

            const float* xi = xb + chunk * 64;

            // Low nibbles → x[0..31]  weight = d1 * nibble - mf1
            // Alternate acc0/acc1 for 2-accumulator pipelining (hides FMA latency)
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(d1, u8x8_to_f32_(lo,  0), mf1), _mm256_loadu_ps(xi +  0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(d1, u8x8_to_f32_(lo,  8), mf1), _mm256_loadu_ps(xi +  8), acc1);
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(d1, u8x8_to_f32_(lo, 16), mf1), _mm256_loadu_ps(xi + 16), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(d1, u8x8_to_f32_(lo, 24), mf1), _mm256_loadu_ps(xi + 24), acc1);

            // High nibbles → x[32..63]  weight = d2 * nibble - mf2
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(d2, u8x8_to_f32_(hi,  0), mf2), _mm256_loadu_ps(xi + 32), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(d2, u8x8_to_f32_(hi,  8), mf2), _mm256_loadu_ps(xi + 40), acc1);
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(d2, u8x8_to_f32_(hi, 16), mf2), _mm256_loadu_ps(xi + 48), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(d2, u8x8_to_f32_(hi, 24), mf2), _mm256_loadu_ps(xi + 56), acc1);
        }
    }
    return hsum256_(_mm256_add_ps(acc0, acc1));
}

// ── Range GEMV — called by ThreadPool workers ─────────────────────────────────
void gemv_q4k_avx2_range(float* y, const uint8_t* W, const float* x,
                          int o_start, int o_end, int in_dim,
                          size_t row_bytes) noexcept {
    for (int o = o_start; o < o_end; o++)
        y[o] = q4k_dot_row_avx2(W + (size_t)o * row_bytes, x, in_dim);
}

} // namespace axonforge::cpu_x86
