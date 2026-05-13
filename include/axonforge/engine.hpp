#pragma once
#include "axonforge/common.hpp"
#include "axonforge/tensor.hpp"
#include "axonforge/sampler.hpp"
#include "axonforge/kv_cache.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace axonforge {

// Forward declarations
class IBackend;
class ComputeGraph;
class IKvCacheManager;

// ============================================================
// ModelConfig — hyperparameters extracted from the model file.
// ============================================================
struct ModelConfig {
    std::string arch;           // e.g. "llama", "qwen2", "gemma2"
    int  n_layers{0};
    int  n_heads{0};
    int  n_kv_heads{0};
    int  head_dim{0};
    int  hidden_dim{0};
    int  ffn_dim{0};
    int  vocab_size{0};
    int  max_seq_len{4096};
    DType weight_dtype{DType::F16};
    float rope_theta{10000.0f};
    float rms_norm_eps{1e-5f};
};

// ============================================================
// EngineConfig — user-facing configuration for Engine creation.
// ============================================================
struct EngineConfig {
    std::string backend     = "cpu_x86"; // must match a registered backend id
    std::string kv_cache    = "paged";   // "continuous" or "paged"
    int  max_batch_tokens   = 2048;
    int  max_kv_pages       = 512;       // used by PagedKvCache
    int  num_threads        = 0;         // 0 = auto (physical core count)
    bool verbose            = false;
};

// ============================================================
// KvState — per-session KV cache handle, passed to Session::forward().
// ============================================================
struct KvState {
    std::unique_ptr<IKvCacheManager> cache;
    int  position{0};   // number of tokens processed so far

    explicit KvState(std::unique_ptr<IKvCacheManager> cache_impl)
        : cache(std::move(cache_impl)) {}
};

// ============================================================
// Session — stateless inference handle (shares model weights
// with the Engine). Each Session has its own KvState.
// ============================================================
class Session {
public:
    ~Session();

    // Run one forward pass; returns logits tensor [vocab_size].
    [[nodiscard]] Tensor forward(std::span<const int32_t> token_ids,
                                 KvState&                 kv);

    // Run prefill + decode loop; returns generated token ids.
    [[nodiscard]] std::vector<int32_t>
        generate(std::span<const int32_t> prompt_ids,
                 KvState&                 kv,
                 const SamplerConfig&     sampler_cfg,
                 int                      max_new_tokens = 256,
                 int                      eos_id         = -1);

    // Create a fresh KvState for a new conversation.
    [[nodiscard]] KvState new_kv_state() const;

private:
    friend class Engine;

    struct Impl;
    explicit Session(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// ============================================================
// Engine — top-level inference engine.
// Owns the loaded model weights and the compiled compute graph.
// Sessions share the engine (and thus the weights).
// ============================================================
class Engine {
public:
    ~Engine();

    // ---- Factory ----
    // Load from a GGUF file. Model weights are mmap'd (lazy page faults).
    [[nodiscard]] static Engine from_gguf(std::string_view path,
                                          EngineConfig     config = {});

    // ---- Session management ----
    [[nodiscard]] Session new_session() const;

    // ---- Tokenizer ----
    [[nodiscard]] std::vector<int32_t>
        encode(std::string_view text, bool add_bos = true, bool raw = false) const;
    [[nodiscard]] std::string
        decode(std::span<const int32_t> token_ids) const;
    [[nodiscard]] int32_t bos_id()  const noexcept;
    [[nodiscard]] int32_t eos_id()  const noexcept;
    [[nodiscard]] int32_t vocab_size() const noexcept;

    // ---- Model info ----
    [[nodiscard]] const ModelConfig&  model_config()  const noexcept;
    [[nodiscard]] const EngineConfig& engine_config() const noexcept;
    [[nodiscard]] const std::string&  chat_template() const noexcept;

    // ---- Weight access (zero-copy views into mmap region) ----
    // Returns nullptr if name not found.
    [[nodiscard]] const Tensor* weight(std::string_view name) const noexcept;

    // ---- Vocabulary ----
    // Returns the token vocabulary loaded from GGUF tokenizer.ggml.tokens.
    [[nodiscard]] const std::vector<std::string>& vocabulary() const noexcept;
    [[nodiscard]] const std::vector<std::string>& tokenizer_merges() const noexcept;
    [[nodiscard]] const std::vector<int32_t>&     tokenizer_token_types() const noexcept;
    [[nodiscard]] const std::string&              tokenizer_pre() const noexcept;

private:
    struct Impl;
    explicit Engine(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace axonforge
