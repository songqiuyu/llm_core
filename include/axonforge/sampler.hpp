#pragma once
#include "axonforge/tensor.hpp"
#include <memory>
#include <vector>

namespace axonforge {

// ============================================================
// ISampler — a single step in the sampling pipeline.
// Each implementation transforms logits in-place.
// ============================================================
class ISampler {
public:
    virtual ~ISampler() = default;
    // Apply transformation to `logits` (shape: [vocab_size] or [1, vocab_size]).
    // Returns modified logits (may be the same tensor or a new one).
    virtual Tensor apply(Tensor logits) = 0;
};

// ---- Concrete samplers ----

// Scale logits by 1/temperature. temperature = 0 ⟹ greedy (argmax).
class TemperatureSampler final : public ISampler {
public:
    explicit TemperatureSampler(float temperature = 1.0f);
    Tensor apply(Tensor logits) override;
private:
    float temperature_;
};

// Keep only the top-k logits, set others to -∞.
class TopKSampler final : public ISampler {
public:
    explicit TopKSampler(int k = 40);
    Tensor apply(Tensor logits) override;
private:
    int k_;
};

// Nucleus (top-p) sampling: keep the smallest set of tokens
// whose cumulative probability ≥ p.
class TopPSampler final : public ISampler {
public:
    explicit TopPSampler(float p = 0.95f);
    Tensor apply(Tensor logits) override;
private:
    float p_;
};

// Penalise tokens that appear in the recent context.
class RepetitionPenaltySampler final : public ISampler {
public:
    explicit RepetitionPenaltySampler(float penalty = 1.1f, int context_window = 64);
    void set_context(std::vector<int32_t> context);
    Tensor apply(Tensor logits) override;
private:
    float penalty_;
    int   context_window_;
    std::vector<int32_t> context_;
};

// ============================================================
// SamplerPipeline — composable chain of ISampler steps.
//
// Usage:
//   SamplerPipeline pipeline;
//   pipeline.add(std::make_unique<TemperatureSampler>(0.8f));
//   pipeline.add(std::make_unique<TopPSampler>(0.95f));
//   int32_t token = pipeline.sample(logits);
// ============================================================
class SamplerPipeline {
public:
    void add(std::unique_ptr<ISampler> sampler);

    // Run the pipeline and draw one token id.
    // If temperature == 0 after TemperatureSampler, returns argmax.
    [[nodiscard]] int32_t sample(Tensor logits);

    void clear();

private:
    std::vector<std::unique_ptr<ISampler>> samplers_;
};

// ---- Factory helpers ----

// Build a default pipeline: temperature → top_k → top_p → sample.
struct SamplerConfig {
    float temperature    = 0.8f;
    int   top_k          = 40;
    float top_p          = 0.95f;
    float rep_penalty    = 1.0f;
    int   rep_context_window = 64;
};

[[nodiscard]] SamplerPipeline make_sampler_pipeline(const SamplerConfig& cfg);

} // namespace axonforge
