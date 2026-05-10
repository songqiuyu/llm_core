// ============================================================
// AxonForge — LLaMA-family inference (F16 / Q4_K_M / Q6_K weights)
//
// Covers: LLaMA-2, LLaMA-3, LLaMA-3.2, TinyLlama, Mistral
// Architecture: RMSNorm + RoPE + SwiGLU FFN + GQA attention
//
// Key optimisations:
//   • gemv_f16_dispatch  — AVX2+F16C GEMV via persistent ThreadPool
//   • q4k_dot_row        — scalar Q4_K_M row dot (on-the-fly dequant)
//   • q6k_emb_row        — Q6_K embedding row dequant
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
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Persistent thread pool (header-only)
#include "../core/thread_pool.hpp"
// CPU feature detection
#include "../backend/cpu_x86/cpuid.hpp"

// ─── AVX2+F16C GEMV forward declaration ──────────────────────────────────────
namespace axonforge::cpu_x86 {
    void gemv_f16_avx2_range(float* y, const uint16_t* W, const float* x,
                             const float* b,
                             int row_start, int row_end, int in_dim) noexcept;
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
        d = (sc[j+4] >> 4) | ((sc[j-4] & 0xC0) >> 2);
        m = (sc[j+0] >> 4) | ((sc[j-4] & 0xC0) >> 2);
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
};
static WT make_wt(const Engine& e, const char* name) {
    const Tensor* t = w_req(e, name);
    return {t->raw_data(), t->dtype()};
}
static WT make_wt_opt(const Engine& e, const char* name, const WT& fallback) {
    const Tensor* t = e.weight(name);
    if (!t) return fallback;
    return {t->raw_data(), t->dtype()};
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
                for (int o = s; o < e; o++)
                    y[o] = q4k_dot_row(Wq + (size_t)o * row_bytes, x, in);
            });
        } else {
            for (int o = 0; o < out; o++)
                y[o] = q4k_dot_row(Wq + (size_t)o * row_bytes, x, in);
        }
    } else if (W.dtype == DType::Q6_K) {
        const uint8_t* Wq = static_cast<const uint8_t*>(W.data);
        const size_t row_bytes = (in / QK_K) * Q6K_BYTES;
        if (pool && out >= 32) {
            pool->parallel_for(out, [&](int s, int e) {
                for (int o = s; o < e; o++)
                    y[o] = q6k_dot_row(Wq + (size_t)o * row_bytes, x, in);
            });
        } else {
            for (int o = 0; o < out; o++)
                y[o] = q6k_dot_row(Wq + (size_t)o * row_bytes, x, in);
        }
    } else {
        // Fallback: treat as F16 (BF16 etc.)
        const uint16_t* Wf = static_cast<const uint16_t*>(W.data);
        kLlamaGemvRange(y, Wf, x, nullptr, 0, out, in);
    }
}

// Embedding lookup: dequantise one row to f32
static void emb_lookup(float* out, const WT& W, int row, int cols) noexcept {
    if (W.dtype == DType::F16) {
        const uint16_t* p = static_cast<const uint16_t*>(W.data) + (size_t)row * cols;
        for (int i = 0; i < cols; i++) out[i] = f16_to_f32(p[i]);
    } else if (W.dtype == DType::Q6_K) {
        const size_t row_bytes = (cols / QK_K) * Q6K_BYTES;
        q6k_row_to_f32(out, static_cast<const uint8_t*>(W.data) + (size_t)row * row_bytes, cols);
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
    WT wgate, wup, wdown;     // SwiGLU FFN
};

struct LlamaWeights {
    WT           tok_embd;
    const float* output_norm;  // always F32
    WT           output;       // lm head
    std::vector<LlamaLayerW> layers;
};

static LlamaWeights build_weights(const Engine& e, int n_layer) {
    LlamaWeights w;
    w.tok_embd   = make_wt(e, "token_embd.weight");
    w.output_norm = w_f32(e, "output_norm.weight");
    w.output     = make_wt_opt(e, "output.weight", w.tok_embd);
    w.layers.resize(n_layer);
    char name[128];
    for (int l = 0; l < n_layer; l++) {
        auto& lw = w.layers[l];
        std::snprintf(name, sizeof(name), "blk.%d.attn_norm.weight",  l); lw.attn_norm = w_f32(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight",   l); lw.ffn_norm  = w_f32(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.weight",     l); lw.wq   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_k.weight",     l); lw.wk   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_v.weight",     l); lw.wv   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.attn_output.weight",l); lw.wo   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight",   l); lw.wgate = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_up.weight",     l); lw.wup   = make_wt(e, name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down.weight",   l); lw.wdown = make_wt(e, name);
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

    // Pre-built weight pointers
    LlamaWeights weights;

    // RoPE frequency cache
    RopeCache rope_cache;

    // Thread pool
    std::unique_ptr<ThreadPool> pool;

    LlamaState(int nl, int nc, int ne, int nh, int nkv, int nv, int fd,
               float eps, float rtheta, int n_threads)
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
        rope_cache.build(head_dim, rope_theta, nc);
        if (n_threads > 1)
            pool = std::make_unique<ThreadPool>(n_threads);
    }

    float* k_buf(int layer, int pos) {
        return kv_k.data() + ((size_t)layer * n_ctx + pos) * kv_embd;
    }
    float* v_buf(int layer, int pos) {
        return kv_v.data() + ((size_t)layer * n_ctx + pos) * kv_embd;
    }
};

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
    ThreadPool* pool = s.pool.get();

    // ── 1. Token embedding lookup (dtype-aware) ──────────────────────────────
    emb_lookup(s.x.data(), w.tok_embd, token_id, E);

    // ── 2. Transformer blocks ────────────────────────────────────────────────
    for (int l = 0; l < L; l++) {
        const LlamaLayerW& lw = w.layers[l];

        // ---- RMSNorm (attention) ----
        std::copy(s.x.begin(), s.x.end(), s.xb.begin());
        rms_norm(s.xb.data(), lw.attn_norm, E, s.rms_eps);

        // ---- Q, K, V projections (dtype-aware GEMV) ----
        gemv_wt(s.q.data(), lw.wq, s.xb.data(), H   * HD, E, pool);
        gemv_wt(s.k.data(), lw.wk, s.xb.data(), KVH * HD, E, pool);
        gemv_wt(s.v.data(), lw.wv, s.xb.data(), KVH * HD, E, pool);

        // ---- Apply RoPE to Q and K ----
        apply_rope(s.q.data(), pos, H,   HD, s.rope_cache);
        apply_rope(s.k.data(), pos, KVH, HD, s.rope_cache);

        // ---- Store K, V in cache ----
        std::copy(s.k.begin(), s.k.end(), s.k_buf(l, pos));
        std::copy(s.v.begin(), s.v.end(), s.v_buf(l, pos));

        // ---- GQA multi-head attention ----
        const int seq = pos + 1;
        const int kv_group = H / KVH;
        std::fill(s.attn_out.begin(), s.attn_out.end(), 0.f);

        for (int h = 0; h < H; h++) {
            const int kv_h = h / kv_group;
            const float* Q_h = s.q.data() + h * HD;

            for (int t = 0; t < seq; t++) {
                const float* K_t = s.k_buf(l, t) + kv_h * HD;
                float score = 0.f;
                for (int d = 0; d < HD; d++) score += Q_h[d] * K_t[d];
                s.attn[t] = score * scale;
            }
            softmax(s.attn.data(), seq);

            float* out_h = s.attn_out.data() + h * HD;
            for (int t = 0; t < seq; t++) {
                const float* V_t = s.v_buf(l, t) + kv_h * HD;
                for (int d = 0; d < HD; d++) out_h[d] += s.attn[t] * V_t[d];
            }
        }

        // ---- Output projection + residual ----
        gemv_wt(s.xb.data(), lw.wo, s.attn_out.data(), E, E, pool);
        for (int i = 0; i < E; i++) s.x[i] += s.xb[i];

        // ---- RMSNorm (FFN) ----
        std::copy(s.x.begin(), s.x.end(), s.xb.begin());
        rms_norm(s.xb.data(), lw.ffn_norm, E, s.rms_eps);

        // ---- SwiGLU FFN: gate and up projections (dtype-aware) ----
        gemv_wt(s.gate.data(), lw.wgate, s.xb.data(), FD, E, pool);
        gemv_wt(s.up.data(),   lw.wup,   s.xb.data(), FD, E, pool);
        for (int i = 0; i < FD; i++) s.gate[i] = silu(s.gate[i]) * s.up[i];

        // ---- FFN down projection + residual ----
        gemv_wt(s.ffn.data(), lw.wdown, s.gate.data(), E, FD, pool);
        for (int i = 0; i < E; i++) s.x[i] += s.ffn[i];
    }

    // ── 3. Final RMSNorm ─────────────────────────────────────────────────────
    rms_norm(s.x.data(), w.output_norm, E, s.rms_eps);

    // ── 4. LM head: logits[V] = output_w[V×E] × x (dtype-aware) ─────────────
    gemv_wt(s.logits.data(), w.output, s.x.data(), s.n_vocab, E, pool);
}

// ─── Sampling (identical strategy to GPT-2 side) ─────────────────────────────

static int sample_top_k(const float* logits, int n_vocab,
                         int top_k, float temperature,
                         std::mt19937& rng) {
    std::vector<std::pair<float, int>> scored(n_vocab);
    for (int i = 0; i < n_vocab; i++) scored[i] = {logits[i] / temperature, i};
    if (top_k <= 1)
        return std::max_element(scored.begin(), scored.end())->second;
    int k = std::min(top_k, n_vocab);
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });
    float mx = scored[0].first, sum = 0.f;
    std::vector<float> probs(k);
    for (int i = 0; i < k; i++) { probs[i] = std::exp(scored[i].first - mx); sum += probs[i]; }
    for (int i = 0; i < k; i++) probs[i] /= sum;
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
        : std::min(8, (int)std::thread::hardware_concurrency() / 2);

    LlamaState state(L, NC, E, H, KVH, V, FD, eps, rtheta, n_threads);
    state.weights = build_weights(engine, L);
    std::mt19937 rng(42);

    // Prepend BOS token (id=1 for LLaMA-2/3/TinyLlama) if not already present
    const int32_t bos = engine.bos_id();
    if (prompt_ids.empty() || prompt_ids[0] != bos)
        prompt_ids.insert(prompt_ids.begin(), bos);

    // Prefill
    for (int i = 0; i < (int)prompt_ids.size(); i++) {
        llama_forward_token(prompt_ids[i], i, state);
        state.n_past++;
        if (cfg.verbose)
            std::fprintf(stderr, "\r[LLaMA] prefill %d/%d", i+1, (int)prompt_ids.size());
    }
    if (cfg.verbose) std::fprintf(stderr, "\n");

    // Decode loop
    std::vector<int32_t> generated;
    generated.reserve(cfg.max_new_tokens);

    int32_t next_id = sample_top_k(state.logits.data(), V,
                                    cfg.top_k, cfg.temperature, rng);

    for (int step = 0; step < cfg.max_new_tokens; step++) {
        generated.push_back(next_id);
        if (cfg.on_token) cfg.on_token(next_id);
        if (next_id == engine.eos_id()) break;
        if (state.n_past >= NC) break;

        llama_forward_token(next_id, state.n_past, state);
        state.n_past++;
        next_id = sample_top_k(state.logits.data(), V,
                                cfg.top_k, cfg.temperature, rng);
    }
    return generated;
}

// ─── Tokenization ─────────────────────────────────────────────────────────────
// LLaMA GGUF stores tokens with SentencePiece '▁' (U+2581) prefix for spaces.
// We decode by replacing ▁ with a plain space.

std::string llama_decode_tokens(const Engine& engine,
                                  const std::vector<int32_t>& token_ids) {
    const auto& vocab = engine.vocabulary();
    std::string out;
    for (int32_t id : token_ids) {
        if (id < 0 || id >= (int32_t)vocab.size()) continue;
        const std::string& tok = vocab[id];
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
                                           std::string_view text) {
    const auto& vocab = engine.vocabulary();
    if (vocab.empty())
        throw std::runtime_error("llama_encode_simple: vocabulary is empty");

    // Build reverse map. LLaMA vocab tokens use ▁ for spaces; we normalise
    // input spaces to ▁ so the lookup matches.
    std::unordered_map<std::string, int32_t> str_to_id;
    str_to_id.reserve(vocab.size());
    for (int32_t i = 0; i < (int32_t)vocab.size(); i++)
        str_to_id.emplace(vocab[i], i);

    // SentencePiece convention: prepend ▁ to whole input so first word maps to
    // e.g. "▁The" just like llama.cpp's add_space_prefix=true behaviour.
    // Also replace every internal space with ▁.
    std::string normalised = "\xE2\x96\x81";  // leading ▁
    normalised.reserve(text.size() * 2 + 3);
    for (char c : text) {
        if (c == ' ')
            normalised += "\xE2\x96\x81";  // ▁
        else
            normalised += c;
    }

    // Greedy longest-match
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < normalised.size()) {
        size_t best_len = 0;
        int32_t best_id  = -1;
        const size_t max_len = std::min(normalised.size() - pos, size_t(64));
        for (size_t len = max_len; len >= 1; len--) {
            auto it = str_to_id.find(normalised.substr(pos, len));
            if (it != str_to_id.end()) { best_len = len; best_id = it->second; break; }
        }
        if (best_id < 0) pos++;
        else { ids.push_back(best_id); pos += best_len; }
    }
    return ids;
}

} // namespace axonforge
