// ============================================================
// GPT-2 117M autoregressive inference
//   — AVX2+F16C GEMV, parallel ThreadPool, zero-snprintf weights
//
// Architecture (12-layer, h=12, d=768):
//   Embedding (wte F16 + wpe F32)
//   × 12 TransformerBlock: LayerNorm → QKV(MT) → MHA+KVcache → proj(MT)
//                           LayerNorm → fc(MT) → GELU → proj(MT)
//   LayerNorm → LM head wte.T (MT)
//   → top-k sample
//
// Optimisations vs Phase-0 scalar baseline:
//   1. gemv_f16_dispatch(): runtime-select AVX2+F16C or scalar
//   2. ThreadPool parallel_for() on output rows (n_threads, default 8)
//   3. Gpt2Weights: pre-built raw-pointer table — zero snprintf/hashmap per token
// ============================================================

#include "axonforge/models/gpt2.hpp"
#include "axonforge/tensor.hpp"
#include "backend/cpu_x86/cpuid.hpp"    // cpu_features()
#include "core/thread_pool.hpp"          // ThreadPool
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward-declare AVX2+F16C kernels (compiled with -mavx2 -mfma -mf16c in a
// separate TU so gpt2_model.cpp needs no SIMD flags itself).
namespace axonforge::cpu_x86 {
    void gemv_f16_avx2_range(float* y, const uint16_t* W, const float* x,
                              const float* b, int o_start, int o_end, int in) noexcept;
    void gemv_f16_avx2(float* y, const uint16_t* W, const float* x,
                        const float* b, int out, int in) noexcept;
}

namespace axonforge {

// ─── F16 ↔ F32 conversion ────────────────────────────────────────────────────

static inline float f16_to_f32(uint16_t h) noexcept {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x03FF;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s; }
        else {
            uint32_t em = m, ex = 127 - 14;
            while (!(em & 0x400)) { em <<= 1; ex--; }
            f = s | (ex << 23) | ((em & 0x3FF) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000 | (m << 13);
    } else {
        f = s | ((e + 112) << 23) | (m << 13);
    }
    float v; std::memcpy(&v, &f, 4); return v;
}

// ─── Math primitives ─────────────────────────────────────────────────────────

// In-place layer norm: y = (x - mean) / sqrt(var + eps) * g + b
static void layer_norm(float* x, const float* g, const float* b,
                       int n, float eps = 1e-5f) noexcept {
    float mean = 0.f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d*d; }
    var /= n;
    const float inv_std = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < n; i++)
        x[i] = (x[i] - mean) * inv_std * g[i] + b[i];
}

// ─── GEMV dispatch layer ──────────────────────────────────────────────────────

namespace cpu_x86 = axonforge::cpu_x86;

// Scalar fallback (no SIMD requirement on the caller TU)
static void gemv_f16_scalar_range(float* __restrict__ y,
                                   const uint16_t* __restrict__ W,
                                   const float*    __restrict__ x,
                                   const float*    __restrict__ b,
                                   int o_start, int o_end, int in) noexcept {
    for (int o = o_start; o < o_end; o++) {
        float acc = b ? b[o] : 0.f;
        const uint16_t* row = W + (size_t)o * in;
        for (int k = 0; k < in; k++)
            acc += f16_to_f32(row[k]) * x[k];
        y[o] = acc;
    }
}

// ─── Runtime-selected implementation ─────────────────────────────────────────
// Detect once at startup; choose AVX2 path when f16c is available.
using GemvRangeFn = void(*)(float*, const uint16_t*, const float*,
                              const float*, int, int, int);

static GemvRangeFn select_gemv_range() noexcept {
    return cpu_x86::cpu_features().avx2
        ? cpu_x86::gemv_f16_avx2_range
        : gemv_f16_scalar_range;
}
static const GemvRangeFn kGemvRange = select_gemv_range();

// Threshold: below this output size, single-thread is faster (avoids pool overhead)
static constexpr int kMtThreshold = 128;

// Dispatch: multithreaded when pool != nullptr && out >= threshold
static void gemv_f16_dispatch(float* y, const uint16_t* W, const float* x,
                               const float* b, int out, int in,
                               ThreadPool* pool) noexcept {
    if (pool && out >= kMtThreshold) {
        pool->parallel_for(out, [&](int s, int e) {
            kGemvRange(y, W, x, b, s, e, in);
        });
    } else {
        kGemvRange(y, W, x, b, 0, out, in);
    }
}

// Approximate GELU (same as GPT-2 TF implementation)
static inline float gelu(float x) noexcept {
    constexpr float c = 0.7978845608f; // sqrt(2/pi)
    constexpr float a = 0.044715f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.f + std::tanh(c * (x + a * x3)));
}

// Softmax in-place
static void softmax(float* x, int n) noexcept {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ─── Weight pointer helpers ───────────────────────────────────────────────────

static const uint16_t* w_f16(const Engine& e, const char* name) {
    const Tensor* t = e.weight(name);
    if (!t) throw std::runtime_error(std::string("GPT-2: missing weight ") + name);
    return static_cast<const uint16_t*>(t->raw_data());
}
static const float* w_f32(const Engine& e, const char* name) {
    const Tensor* t = e.weight(name);
    if (!t) throw std::runtime_error(std::string("GPT-2: missing weight ") + name);
    return static_cast<const float*>(t->raw_data());
}

// ─── Pre-built weight pointer table ──────────────────────────────────────────
// Built once before the generate loop; eliminates snprintf + hashmap lookups.

struct Gpt2LayerW {
    const float*    ln1g,  *ln1b,  *ln2g,  *ln2b;
    const uint16_t* cattn_w, *cproj_w, *fc_w, *proj_w;
    const float*    cattn_b, *cproj_b, *fc_b, *proj_b;
};

struct Gpt2Weights {
    const uint16_t* wte;
    const float*    wpe;
    const float*    lnf_g, *lnf_b;
    std::vector<Gpt2LayerW> layers;
};

static Gpt2Weights build_weights(const Engine& e, int n_layer) {
    Gpt2Weights w;
    w.wte   = w_f16(e, "model/wte");
    w.wpe   = w_f32(e, "model/wpe");
    w.lnf_g = w_f32(e, "model/ln_f/g");
    w.lnf_b = w_f32(e, "model/ln_f/b");
    w.layers.resize(n_layer);
    char name[128];
    for (int l = 0; l < n_layer; l++) {
        auto& lw = w.layers[l];
        std::snprintf(name, sizeof(name), "model/h%d/ln_1/g",        l); lw.ln1g    = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/ln_1/b",        l); lw.ln1b    = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/ln_2/g",        l); lw.ln2g    = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/ln_2/b",        l); lw.ln2b    = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/attn/c_attn/w", l); lw.cattn_w = w_f16(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/attn/c_attn/b", l); lw.cattn_b = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/attn/c_proj/w", l); lw.cproj_w = w_f16(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/attn/c_proj/b", l); lw.cproj_b = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/mlp/c_fc/w",    l); lw.fc_w    = w_f16(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/mlp/c_fc/b",    l); lw.fc_b    = w_f32(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/mlp/c_proj/w",  l); lw.proj_w  = w_f16(e, name);
        std::snprintf(name, sizeof(name), "model/h%d/mlp/c_proj/b",  l); lw.proj_b  = w_f32(e, name);
    }
    return w;
}

// ─── GPT-2 model state (KV cache + activations + thread pool) ────────────────

struct Gpt2State {
    int n_layer, n_ctx, n_embd, n_head, head_dim, n_vocab;

    // KV cache: [n_layer][n_ctx][n_embd]
    std::vector<float> kv_k;
    std::vector<float> kv_v;
    int n_past{0};

    // Scratch buffers
    std::vector<float> x, xb, qkv, attn, attn_out, ffn, logits;

    // Pre-built weight pointers (set by gpt2_generate before the loop)
    Gpt2Weights weights;

    // Thread pool (null = single-threaded)
    std::unique_ptr<ThreadPool> pool;

    explicit Gpt2State(int nl, int nc, int ne, int nh, int nv, int n_threads)
        : n_layer(nl), n_ctx(nc), n_embd(ne), n_head(nh)
        , head_dim(ne / nh), n_vocab(nv)
    {
        kv_k.resize((size_t)nl * nc * ne, 0.f);
        kv_v.resize((size_t)nl * nc * ne, 0.f);
        x.resize(ne); xb.resize(ne);
        qkv.resize(3*ne);
        attn.resize(nc);
        attn_out.resize(ne);
        ffn.resize(4*ne);
        logits.resize(nv);
        if (n_threads > 1)
            pool = std::make_unique<ThreadPool>(n_threads);
    }

    float* k_buf(int layer, int pos) {
        return kv_k.data() + ((size_t)layer * n_ctx + pos) * n_embd;
    }
    float* v_buf(int layer, int pos) {
        return kv_v.data() + ((size_t)layer * n_ctx + pos) * n_embd;
    }
};

// ─── Single-token forward pass ────────────────────────────────────────────────
// Uses pre-built s.weights (no snprintf) and dispatches GEMV via pool.

static void gpt2_forward_token(int token_id, int pos, Gpt2State& s) {
    const int E  = s.n_embd;
    const int H  = s.n_head;
    const int HD = s.head_dim;
    const int L  = s.n_layer;
    const float scale = 1.f / std::sqrt((float)HD);
    const Gpt2Weights& w = s.weights;
    ThreadPool* pool = s.pool.get();

    // ── 1. Embedding: x = wte[token_id] + wpe[pos] ──────────────────────────
    {
        const uint16_t* tok_row = w.wte + (size_t)token_id * E;
        const float*    pos_row = w.wpe + (size_t)pos       * E;
        for (int i = 0; i < E; i++)
            s.x[i] = f16_to_f32(tok_row[i]) + pos_row[i];
    }

    // ── 2. Transformer blocks ────────────────────────────────────────────────
    for (int l = 0; l < L; l++) {
        const Gpt2LayerW& lw = w.layers[l];

        // ---- ln_1 ----
        std::copy(s.x.begin(), s.x.end(), s.xb.begin());
        layer_norm(s.xb.data(), lw.ln1g, lw.ln1b, E);

        // ---- QKV projection [3*E] = c_attn_w[3E×E] × xb + c_attn_b --------
        gemv_f16_dispatch(s.qkv.data(), lw.cattn_w, s.xb.data(),
                          lw.cattn_b, 3*E, E, pool);

        // ---- Store K, V in cache for this position ----
        std::copy(s.qkv.data() + E,   s.qkv.data() + 2*E, s.k_buf(l, pos));
        std::copy(s.qkv.data() + 2*E, s.qkv.data() + 3*E, s.v_buf(l, pos));

        // ---- Multi-head causal self-attention --------------------------------
        const int seq = pos + 1;
        std::fill(s.attn_out.begin(), s.attn_out.end(), 0.f);

        for (int h = 0; h < H; h++) {
            const float* Q_h = s.qkv.data() + h * HD;
            for (int t = 0; t < seq; t++) {
                const float* K_t_h = s.k_buf(l, t) + h * HD;
                float score = 0.f;
                for (int d = 0; d < HD; d++) score += Q_h[d] * K_t_h[d];
                s.attn[t] = score * scale;
            }
            softmax(s.attn.data(), seq);
            float* out_h = s.attn_out.data() + h * HD;
            for (int t = 0; t < seq; t++) {
                const float* V_t_h = s.v_buf(l, t) + h * HD;
                for (int d = 0; d < HD; d++)
                    out_h[d] += s.attn[t] * V_t_h[d];
            }
        }

        // ---- Attention output projection [E] = c_proj_w[E×E] × attn_out ----
        gemv_f16_dispatch(s.xb.data(), lw.cproj_w, s.attn_out.data(),
                          lw.cproj_b, E, E, pool);
        for (int i = 0; i < E; i++) s.x[i] += s.xb[i];   // residual

        // ---- ln_2 ----
        std::copy(s.x.begin(), s.x.end(), s.xb.begin());
        layer_norm(s.xb.data(), lw.ln2g, lw.ln2b, E);

        // ---- FFN up-proj [4E] = c_fc_w[4E×E] × xb -------------------------
        gemv_f16_dispatch(s.ffn.data(), lw.fc_w, s.xb.data(),
                          lw.fc_b, 4*E, E, pool);
        for (int i = 0; i < 4*E; i++) s.ffn[i] = gelu(s.ffn[i]);

        // ---- FFN down-proj [E] = c_proj_w[E×4E] × ffn ----------------------
        gemv_f16_dispatch(s.xb.data(), lw.proj_w, s.ffn.data(),
                          lw.proj_b, E, 4*E, pool);
        for (int i = 0; i < E; i++) s.x[i] += s.xb[i];   // residual
    }

    // ── 3. Final LayerNorm ───────────────────────────────────────────────────
    layer_norm(s.x.data(), w.lnf_g, w.lnf_b, E);

    // ── 4. LM head: logits[V] = wte[V×E] × x  (weight tying) ───────────────
    gemv_f16_dispatch(s.logits.data(), w.wte, s.x.data(),
                      nullptr, s.n_vocab, E, pool);
}

// ─── Sampling ─────────────────────────────────────────────────────────────────

static int sample_top_k(const float* logits, int n_vocab,
                         int top_k, float temperature,
                         std::mt19937& rng) {
    // Temperature scaling + find top-k indices
    std::vector<std::pair<float, int>> scored(n_vocab);
    for (int i = 0; i < n_vocab; i++)
        scored[i] = {logits[i] / temperature, i};

    if (top_k <= 1) {
        return std::max_element(scored.begin(), scored.end())->second;
    }

    // Partial sort to get top-k
    int k = std::min(top_k, n_vocab);
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });

    // Softmax over top-k
    float mx = scored[0].first;
    float sum = 0.f;
    std::vector<float> probs(k);
    for (int i = 0; i < k; i++) {
        probs[i] = std::exp(scored[i].first - mx);
        sum += probs[i];
    }
    for (int i = 0; i < k; i++) probs[i] /= sum;

    // Categorical sample
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(rng);
    float cum = 0.f;
    for (int i = 0; i < k; i++) {
        cum += probs[i];
        if (r < cum) return scored[i].second;
    }
    return scored[k-1].second;
}

// ─── Public API ───────────────────────────────────────────────────────────────

std::vector<int32_t> gpt2_generate(const Engine& engine,
                                    const std::vector<int32_t>& prompt_ids,
                                    const Gpt2Config& cfg) {
    if (prompt_ids.empty())
        throw std::invalid_argument("gpt2_generate: empty prompt");

    const ModelConfig& mc = engine.model_config();
    if (mc.arch != "gpt2")
        throw std::runtime_error("gpt2_generate: model arch is '" +
                                  mc.arch + "', expected 'gpt2'");

    const int L  = mc.n_layers;
    const int E  = mc.hidden_dim;
    const int H  = mc.n_heads;
    const int NC = mc.max_seq_len;
    const int V  = mc.vocab_size > 0 ? mc.vocab_size : 50257;

    // Use at most nproc/2 threads; default 8 if user didn't specify
    const int n_threads = cfg.n_threads > 0
        ? cfg.n_threads
        : std::min(8, (int)std::thread::hardware_concurrency() / 2);

    Gpt2State state(L, NC, E, H, V, n_threads);
    // Pre-build weight pointer table (eliminates per-token snprintf)
    state.weights = build_weights(engine, L);
    std::mt19937 rng(42);

    // Prefill: process all prompt tokens
    for (int i = 0; i < (int)prompt_ids.size(); i++) {
        gpt2_forward_token(prompt_ids[i], i, state);
        state.n_past++;
        if (cfg.verbose)
            std::fprintf(stderr, "\r[GPT-2] prefill %d/%d",
                         i+1, (int)prompt_ids.size());
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
        if (state.n_past >= mc.max_seq_len) break;

        gpt2_forward_token(next_id, state.n_past, state);
        state.n_past++;

        next_id = sample_top_k(state.logits.data(), V,
                                cfg.top_k, cfg.temperature, rng);
    }

    return generated;
}

// ─── GPT-2 byte-level BPE detokenization ─────────────────────────────────────
//
// Each GPT-2 token string is a sequence of unicode characters where each
// character encodes exactly one byte using the bytes_to_unicode mapping.
//
// Direct range: codepoints 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF → byte = codepoint
// Extended:     codepoints 0x100-0x143 → bytes 0x00-0x43 (those not in direct range)

static uint8_t cp_to_byte(uint32_t cp) noexcept {
    // Direct-range codepoints map to themselves
    if ((cp >= 0x21  && cp <= 0x7E) ||
        (cp >= 0xA1  && cp <= 0xAC) ||
        (cp >= 0xAE  && cp <= 0xFF))
        return static_cast<uint8_t>(cp);

    // Extended: codepoints >= 0x100 encode bytes that were NOT in direct range.
    // Bytes not in direct range (in order): 0x00-0x20 (33), 0x7F-0xA0 (34), 0xAD (1) = 68
    static constexpr uint8_t extra[] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
        0x20,
        0x7F,0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,
        0x8F,0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,
        0x9F,0xA0,0xAD
    };
    if (cp >= 0x100) {
        int idx = static_cast<int>(cp - 0x100);
        if (idx < static_cast<int>(sizeof(extra))) return extra[idx];
    }
    return '?';
}

// Decode UTF-8 string to list of unicode codepoints
static std::vector<uint32_t> utf8_to_codepoints(std::string_view s) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < s.size(); ) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        uint32_t cp;
        int bytes;
        if      (c < 0x80) { cp = c; bytes = 1; }
        else if (c < 0xC0) { ++i; continue; }  // invalid continuation byte
        else if (c < 0xE0) { cp = c & 0x1F; bytes = 2; }
        else if (c < 0xF0) { cp = c & 0x0F; bytes = 3; }
        else               { cp = c & 0x07; bytes = 4; }
        for (int b = 1; b < bytes && i+b < s.size(); b++)
            cp = (cp << 6) | (static_cast<uint8_t>(s[i+b]) & 0x3F);
        cps.push_back(cp);
        i += bytes;
    }
    return cps;
}

static std::string token_to_bytes(std::string_view token) {
    std::string out;
    out.reserve(token.size());
    for (uint32_t cp : utf8_to_codepoints(token))
        out += static_cast<char>(cp_to_byte(cp));
    return out;
}

std::string gpt2_decode_tokens(const Engine& engine,
                                 const std::vector<int32_t>& token_ids) {
    const auto& vocab = engine.vocabulary();
    std::string out;
    for (int32_t id : token_ids) {
        if (id >= 0 && id < static_cast<int32_t>(vocab.size()))
            out += vocab[id];
    }
    return out;
}

// ─── Simple greedy tokenizer (raw-byte vocabulary) ───────────────────────────
// The vocabulary in the GGML-converted GGUF stores token strings as raw bytes
// (not the GPT-2 unicode encoding), so we can match directly against the input.

std::vector<int32_t> gpt2_encode_simple(const Engine& engine,
                                          std::string_view text) {
    const auto& vocab = engine.vocabulary();
    if (vocab.empty())
        throw std::runtime_error("gpt2_encode_simple: vocabulary is empty");

    // Build string_view → id map (zero-copy keys)
    std::unordered_map<std::string_view, int32_t> str_to_id;
    str_to_id.reserve(vocab.size());
    for (int32_t i = 0; i < (int32_t)vocab.size(); i++)
        str_to_id.emplace(std::string_view(vocab[i]), i);

    // Greedy longest-match on the raw input
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_len = 0;
        int32_t best_id = -1;
        const size_t max_len = std::min(text.size() - pos, size_t(50));
        for (size_t len = max_len; len >= 1; len--) {
            auto it = str_to_id.find(text.substr(pos, len));
            if (it != str_to_id.end()) { best_len = len; best_id = it->second; break; }
        }
        if (best_id < 0) { pos++; }  // skip unknown byte
        else { ids.push_back(best_id); pos += best_len; }
    }
    return ids;
}

} // namespace axonforge
