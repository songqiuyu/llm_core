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

// ── 4-row simultaneous dot product ───────────────────────────────────────────
// Processes rows o, o+1, o+2, o+3 sharing a single x load per chunk.
// Uses 8 independent accumulators (4 rows × 2) to saturate FMA pipelines.
static inline void q4k_dot_4rows_(float* y, int o,
                                   const uint8_t* W, size_t row_bytes,
                                   const float* x, int in_dim) noexcept {
    __m256 a0a = _mm256_setzero_ps(), a0b = _mm256_setzero_ps();
    __m256 a1a = _mm256_setzero_ps(), a1b = _mm256_setzero_ps();
    __m256 a2a = _mm256_setzero_ps(), a2b = _mm256_setzero_ps();
    __m256 a3a = _mm256_setzero_ps(), a3b = _mm256_setzero_ps();

    const int nb = in_dim / QK_K_;
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const uint8_t* r0 = W + (size_t)(o + 0) * row_bytes;
    const uint8_t* r1 = W + (size_t)(o + 1) * row_bytes;
    const uint8_t* r2 = W + (size_t)(o + 2) * row_bytes;
    const uint8_t* r3 = W + (size_t)(o + 3) * row_bytes;

    for (int b = 0; b < nb; b++) {
        const uint8_t* b0 = r0 + (size_t)b * Q4K_BYTES_;
        const uint8_t* b1 = r1 + (size_t)b * Q4K_BYTES_;
        const uint8_t* b2 = r2 + (size_t)b * Q4K_BYTES_;
        const uint8_t* b3 = r3 + (size_t)b * Q4K_BYTES_;
        const float* xb = x + (size_t)b * QK_K_;

        // Precompute effective scale/min for all 4 rows × 8 groups
        float ds0[8], ms0[8], ds1[8], ms1[8], ds2[8], ms2[8], ds3[8], ms3[8];
        {
            uint16_t d, dm;
            float fsd, fsdm;
            uint8_t sc, mn;

            memcpy(&d, b0, 2); memcpy(&dm, b0+2, 2); fsd=f16c_(d); fsdm=f16c_(dm);
            for (int j=0;j<8;j++) { get_scale_min_k4_(j,b0+4,sc,mn); ds0[j]=fsd*(float)sc; ms0[j]=fsdm*(float)mn; }

            memcpy(&d, b1, 2); memcpy(&dm, b1+2, 2); fsd=f16c_(d); fsdm=f16c_(dm);
            for (int j=0;j<8;j++) { get_scale_min_k4_(j,b1+4,sc,mn); ds1[j]=fsd*(float)sc; ms1[j]=fsdm*(float)mn; }

            memcpy(&d, b2, 2); memcpy(&dm, b2+2, 2); fsd=f16c_(d); fsdm=f16c_(dm);
            for (int j=0;j<8;j++) { get_scale_min_k4_(j,b2+4,sc,mn); ds2[j]=fsd*(float)sc; ms2[j]=fsdm*(float)mn; }

            memcpy(&d, b3, 2); memcpy(&dm, b3+2, 2); fsd=f16c_(d); fsdm=f16c_(dm);
            for (int j=0;j<8;j++) { get_scale_min_k4_(j,b3+4,sc,mn); ds3[j]=fsd*(float)sc; ms3[j]=fsdm*(float)mn; }
        }

        for (int chunk = 0; chunk < 4; chunk++) {
            const float* xi = xb + chunk * 64;
            const int ci = chunk * 2;

            // Load x once for this chunk (shared across all 4 rows)
            const __m256 x0 = _mm256_loadu_ps(xi +  0);
            const __m256 x1 = _mm256_loadu_ps(xi +  8);
            const __m256 x2 = _mm256_loadu_ps(xi + 16);
            const __m256 x3 = _mm256_loadu_ps(xi + 24);
            const __m256 x4 = _mm256_loadu_ps(xi + 32);
            const __m256 x5 = _mm256_loadu_ps(xi + 40);
            const __m256 x6 = _mm256_loadu_ps(xi + 48);
            const __m256 x7 = _mm256_loadu_ps(xi + 56);

#define Q4K_ACCUM_ROW_(qvec_, d1_, mf1_, d2_, mf2_, aa_, ab_)                       \
    do {                                                                              \
        const __m256i lo_ = _mm256_and_si256(qvec_, mask);                           \
        const __m256i hi_ = _mm256_and_si256(_mm256_srli_epi16(qvec_, 4), mask);     \
        aa_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d1_,u8x8_to_f32_(lo_, 0),mf1_),x0,aa_);\
        ab_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d1_,u8x8_to_f32_(lo_, 8),mf1_),x1,ab_);\
        aa_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d1_,u8x8_to_f32_(lo_,16),mf1_),x2,aa_);\
        ab_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d1_,u8x8_to_f32_(lo_,24),mf1_),x3,ab_);\
        aa_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d2_,u8x8_to_f32_(hi_, 0),mf2_),x4,aa_);\
        ab_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d2_,u8x8_to_f32_(hi_, 8),mf2_),x5,ab_);\
        aa_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d2_,u8x8_to_f32_(hi_,16),mf2_),x6,aa_);\
        ab_ = _mm256_fmadd_ps(_mm256_fmsub_ps(d2_,u8x8_to_f32_(hi_,24),mf2_),x7,ab_);\
    } while(0)

            const __m256i qv0 = _mm256_loadu_si256((const __m256i*)(b0+16+chunk*32));
            Q4K_ACCUM_ROW_(qv0, _mm256_set1_ps(ds0[ci]),_mm256_set1_ps(ms0[ci]),
                                _mm256_set1_ps(ds0[ci+1]),_mm256_set1_ps(ms0[ci+1]), a0a,a0b);

            const __m256i qv1 = _mm256_loadu_si256((const __m256i*)(b1+16+chunk*32));
            Q4K_ACCUM_ROW_(qv1, _mm256_set1_ps(ds1[ci]),_mm256_set1_ps(ms1[ci]),
                                _mm256_set1_ps(ds1[ci+1]),_mm256_set1_ps(ms1[ci+1]), a1a,a1b);

            const __m256i qv2 = _mm256_loadu_si256((const __m256i*)(b2+16+chunk*32));
            Q4K_ACCUM_ROW_(qv2, _mm256_set1_ps(ds2[ci]),_mm256_set1_ps(ms2[ci]),
                                _mm256_set1_ps(ds2[ci+1]),_mm256_set1_ps(ms2[ci+1]), a2a,a2b);

            const __m256i qv3 = _mm256_loadu_si256((const __m256i*)(b3+16+chunk*32));
            Q4K_ACCUM_ROW_(qv3, _mm256_set1_ps(ds3[ci]),_mm256_set1_ps(ms3[ci]),
                                _mm256_set1_ps(ds3[ci+1]),_mm256_set1_ps(ms3[ci+1]), a3a,a3b);
#undef Q4K_ACCUM_ROW_
        }
    }
    y[o + 0] = hsum256_(_mm256_add_ps(a0a, a0b));
    y[o + 1] = hsum256_(_mm256_add_ps(a1a, a1b));
    y[o + 2] = hsum256_(_mm256_add_ps(a2a, a2b));
    y[o + 3] = hsum256_(_mm256_add_ps(a3a, a3b));
}

// ── Range GEMV — called by ThreadPool workers ─────────────────────────────────
void gemv_q4k_avx2_range(float* y, const uint8_t* W, const float* x,
                          int o_start, int o_end, int in_dim,
                          size_t row_bytes) noexcept {
    int o = o_start;
    for (; o + 4 <= o_end; o += 4)
        q4k_dot_4rows_(y, o, W, row_bytes, x, in_dim);
    for (; o < o_end; o++)
        y[o] = q4k_dot_row_avx2(W + (size_t)o * row_bytes, x, in_dim);
}

} // namespace axonforge::cpu_x86
