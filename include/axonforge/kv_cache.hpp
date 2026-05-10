#pragma once
#include "axonforge/tensor.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace axonforge {

// ============================================================
// IKvCacheManager — abstract KV cache interface.
// Concrete: ContinuousKvCache (single session, simple),
//           PagedKvCache (multi-session, prefix sharing).
// ============================================================
class IKvCacheManager {
public:
    virtual ~IKvCacheManager() = default;

    // Initialise buffers for a given model configuration.
    virtual void init(int num_layers,
                      int num_kv_heads,
                      int head_dim,
                      DType dtype,
                      int max_seq_len) = 0;

    virtual void clear()    = 0;

    [[nodiscard]] virtual int used_tokens() const noexcept = 0;
    [[nodiscard]] virtual int max_tokens()  const noexcept = 0;

    // Returns (key_view, value_view) for layer `layer_idx`,
    // covering positions [0, current_len).
    virtual std::pair<Tensor, Tensor>
        get_kv_view(int layer_idx, int current_len) = 0;

    // Called after a decode step to advance the write pointer.
    virtual void advance(int n_new_tokens) = 0;
};

// ============================================================
// ContinuousKvCache — contiguous pre-allocated buffers.
// Suitable for single-session, short-sequence inference.
// Shape per layer: [max_seq_len, num_kv_heads, head_dim]
// ============================================================
class ContinuousKvCache final : public IKvCacheManager {
public:
    explicit ContinuousKvCache() = default;

    void init(int num_layers, int num_kv_heads, int head_dim,
              DType dtype, int max_seq_len) override;
    void clear() override;
    [[nodiscard]] int used_tokens() const noexcept override;
    [[nodiscard]] int max_tokens()  const noexcept override;
    std::pair<Tensor, Tensor> get_kv_view(int layer_idx, int current_len) override;
    void advance(int n_new_tokens) override;

private:
    struct LayerCache { Tensor k; Tensor v; };
    std::vector<LayerCache> layers_;
    int used_tokens_{0};
    int max_tokens_{0};
};

// ============================================================
// PagedKvCache — block-table KV cache (vLLM-style).
// Memory is divided into fixed-size pages.
// Supports multiple sessions, prefix sharing, and defragmentation.
//
// Phase 1 TODO: implement full block table allocation.
// ============================================================
class PagedKvCache final : public IKvCacheManager {
public:
    static constexpr int DEFAULT_PAGE_SIZE = 16;  // tokens per page

    explicit PagedKvCache(int max_pages, int page_size = DEFAULT_PAGE_SIZE);

    void init(int num_layers, int num_kv_heads, int head_dim,
              DType dtype, int max_seq_len) override;
    void clear() override;
    [[nodiscard]] int used_tokens() const noexcept override;
    [[nodiscard]] int max_tokens()  const noexcept override;
    std::pair<Tensor, Tensor> get_kv_view(int layer_idx, int current_len) override;
    void advance(int n_new_tokens) override;

private:
    int page_size_;
    int max_pages_;
    int used_tokens_{0};
    // TODO(Phase1): block_table, free_page_list, physical page pool
};

} // namespace axonforge
