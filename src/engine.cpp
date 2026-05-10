#include "axonforge/engine.hpp"
#include "axonforge/backend.hpp"
#include "axonforge/graph.hpp"
#include "axonforge/kv_cache.hpp"
#include "axonforge/sampler.hpp"
#include <stdexcept>
#include <memory>

namespace axonforge {

// ============================================================
// Engine::Impl — holds backend, compiled graph, weights, tokenizer.
// ============================================================
struct Engine::Impl {
    EngineConfig              engine_cfg;
    ModelConfig               model_cfg;
    std::unique_ptr<IBackend> backend;
    ComputeGraph              graph;
    // TODO(Phase0): weight tensors (mmap'd from GGUF)
    // TODO(Phase0): BpeTokenizer tokenizer
};

// ============================================================
// Session::Impl — per-session state (shares engine read-only).
// Defined before Engine methods that create Sessions.
// ============================================================
struct Session::Impl {
    const EngineConfig* engine_cfg{nullptr};
    const ModelConfig*  model_cfg{nullptr};
    // TODO(Phase0): shared weight tensors pointer
};

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Engine::~Engine() = default;

const ModelConfig&  Engine::model_config()   const noexcept { return impl_->model_cfg; }
const EngineConfig& Engine::engine_config()  const noexcept { return impl_->engine_cfg; }
int32_t Engine::bos_id()     const noexcept { return 1; }
int32_t Engine::eos_id()     const noexcept { return 2; }
int32_t Engine::vocab_size() const noexcept { return impl_->model_cfg.vocab_size; }

std::vector<int32_t> Engine::encode(std::string_view, bool) const {
    throw std::runtime_error("Engine::encode: tokenizer not yet implemented");
}
std::string Engine::decode(std::span<const int32_t>) const {
    throw std::runtime_error("Engine::decode: tokenizer not yet implemented");
}

Session Engine::new_session() const {
    auto s_impl = std::make_unique<Session::Impl>();
    s_impl->engine_cfg  = &impl_->engine_cfg;
    s_impl->model_cfg   = &impl_->model_cfg;
    return Session(std::move(s_impl));
}

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() = default;

KvState Session::new_kv_state() const {
    return KvState(std::make_unique<ContinuousKvCache>());
}

Tensor Session::forward(std::span<const int32_t>, KvState&) {
    throw std::runtime_error("Session::forward: not yet implemented");
}

std::vector<int32_t> Session::generate(
    std::span<const int32_t>, KvState&, const SamplerConfig&, int, int)
{
    throw std::runtime_error("Session::generate: not yet implemented");
}

// ============================================================
// KvCache stubs
// ============================================================
void ContinuousKvCache::init(int nl, int nh, int hd, DType dtype, int max_seq) {
    max_tokens_  = max_seq;
    used_tokens_ = 0;
    layers_.clear();
    layers_.reserve(nl);
    for (int l = 0; l < nl; ++l) {
        Tensor k = Tensor::zeros({max_seq, nh, hd}, dtype);
        Tensor v = Tensor::zeros({max_seq, nh, hd}, dtype);
        layers_.push_back({std::move(k), std::move(v)});
    }
}
void ContinuousKvCache::clear()       { used_tokens_ = 0; }
int  ContinuousKvCache::used_tokens() const noexcept { return used_tokens_; }
int  ContinuousKvCache::max_tokens()  const noexcept { return max_tokens_; }
void ContinuousKvCache::advance(int n) { used_tokens_ += n; }

std::pair<Tensor, Tensor>
ContinuousKvCache::get_kv_view(int layer_idx, int current_len) {
    auto& l = layers_.at(layer_idx);
    return {l.k.slice(0, 0, current_len),
            l.v.slice(0, 0, current_len)};
}

void PagedKvCache::init(int /*nl*/, int /*nh*/, int /*hd*/, DType /*dtype*/, int /*max_seq*/) {
    // TODO(Phase1): allocate physical page pool
}
void PagedKvCache::clear()         { used_tokens_ = 0; }
int  PagedKvCache::used_tokens()   const noexcept { return used_tokens_; }
int  PagedKvCache::max_tokens()    const noexcept { return max_pages_ * page_size_; }
void PagedKvCache::advance(int n)  { used_tokens_ += n; }
PagedKvCache::PagedKvCache(int max_pages, int page_size)
    : page_size_(page_size), max_pages_(max_pages) {}

std::pair<Tensor, Tensor>
PagedKvCache::get_kv_view(int /*layer_idx*/, int /*current_len*/) {
    throw std::runtime_error("PagedKvCache::get_kv_view: not yet implemented");
}

// ============================================================
// Sampler implementations (scalar; vectorise in Phase1)
// ============================================================

TemperatureSampler::TemperatureSampler(float t) : temperature_(t) {}
Tensor TemperatureSampler::apply(Tensor logits) {
    if (temperature_ <= 0.f || temperature_ == 1.f) return logits;
    float inv_t = 1.f / temperature_;
    auto  data  = logits.data_as<float>();
    for (float& v : data) v *= inv_t;
    return logits;
}

TopKSampler::TopKSampler(int k) : k_(k) {}
Tensor TopKSampler::apply(Tensor logits) {
    (void)logits;
    // TODO(Phase0): partial sort, mask tokens below k-th
    return logits;
}

TopPSampler::TopPSampler(float p) : p_(p) {}
Tensor TopPSampler::apply(Tensor logits) {
    (void)logits;
    // TODO(Phase0): softmax → cumsum → mask
    return logits;
}

RepetitionPenaltySampler::RepetitionPenaltySampler(float penalty, int ctx)
    : penalty_(penalty), context_window_(ctx) {}
void RepetitionPenaltySampler::set_context(std::vector<int32_t> ctx) {
    context_ = std::move(ctx);
}
Tensor RepetitionPenaltySampler::apply(Tensor logits) {
    auto data = logits.data_as<float>();
    for (int32_t tok : context_) {
        if (tok >= 0 && tok < static_cast<int32_t>(data.size())) {
            data[tok] = data[tok] > 0.f
                      ? data[tok] / penalty_
                      : data[tok] * penalty_;
        }
    }
    return logits;
}

// ---- SamplerPipeline ----
void SamplerPipeline::add(std::unique_ptr<ISampler> s) {
    samplers_.push_back(std::move(s));
}
void SamplerPipeline::clear() { samplers_.clear(); }

int32_t SamplerPipeline::sample(Tensor logits) {
    for (auto& s : samplers_) logits = s->apply(std::move(logits));
    // Greedy argmax
    auto data = logits.data_as<float>();
    int32_t best = 0;
    for (int32_t i = 1; i < static_cast<int32_t>(data.size()); ++i) {
        if (data[i] > data[best]) best = i;
    }
    return best;
}

SamplerPipeline make_sampler_pipeline(const SamplerConfig& cfg) {
    SamplerPipeline p;
    p.add(std::make_unique<TemperatureSampler>(cfg.temperature));
    if (cfg.top_k > 0)      p.add(std::make_unique<TopKSampler>(cfg.top_k));
    if (cfg.top_p < 1.f)    p.add(std::make_unique<TopPSampler>(cfg.top_p));
    if (cfg.rep_penalty > 1.f)
        p.add(std::make_unique<RepetitionPenaltySampler>(
            cfg.rep_penalty, cfg.rep_context_window));
    return p;
}

} // namespace axonforge
