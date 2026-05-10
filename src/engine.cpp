#include "axonforge/engine.hpp"
#include "axonforge/backend.hpp"
#include "axonforge/graph.hpp"
#include "axonforge/kv_cache.hpp"
#include "axonforge/sampler.hpp"
#include "loader/gguf_reader.hpp"
#include "axonforge/models/gpt2.hpp"
#include "axonforge/models/llama.hpp"
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <unordered_map>

namespace axonforge {

// ============================================================
// Engine::Impl — holds backend, compiled graph, weights, tokenizer.
// ============================================================
struct Engine::Impl {
    EngineConfig                           engine_cfg;
    ModelConfig                            model_cfg;
    std::unique_ptr<IBackend>              backend;
    ComputeGraph                           graph;
    std::unique_ptr<GgufReader>            gguf;       // keeps mmap alive
    std::unordered_map<std::string,Tensor> weights;    // name → zero-copy mmap view
    std::vector<std::string>               vocabulary;   // token_id → token string
    int32_t                                bos_token_id{1};
    int32_t                                eos_token_id{2};
    // TODO(Phase0): BpeTokenizer tokenizer
};

// ============================================================
// Session::Impl — per-session state (shares engine read-only).
// Defined before Engine methods that create Sessions.
// ============================================================
struct Session::Impl {
    const EngineConfig* engine_cfg{nullptr};
    const ModelConfig*  model_cfg{nullptr};
    const Engine*       owner{nullptr};    // back-pointer for dispatch
};    // TODO(Phase0): shared weight tensors pointer

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Engine::~Engine() = default;

const ModelConfig&  Engine::model_config()   const noexcept { return impl_->model_cfg; }
const EngineConfig& Engine::engine_config()  const noexcept { return impl_->engine_cfg; }
int32_t Engine::bos_id()     const noexcept { return impl_->bos_token_id; }
int32_t Engine::eos_id()     const noexcept { return impl_->eos_token_id; }
int32_t Engine::vocab_size() const noexcept { return impl_->model_cfg.vocab_size; }

const Tensor* Engine::weight(std::string_view name) const noexcept {
    auto it = impl_->weights.find(std::string(name));
    return it != impl_->weights.end() ? &it->second : nullptr;
}

const std::vector<std::string>& Engine::vocabulary() const noexcept {
    return impl_->vocabulary;
}

Engine Engine::from_gguf(std::string_view path, EngineConfig cfg) {
    // ---- 1. Parse the GGUF file (mmap'd) ----
    auto gguf = std::make_unique<GgufReader>();
    gguf->open(path);  // throws on any format error

    // ---- 2. Extract ModelConfig from KV metadata ----
    ModelConfig mcfg;
    mcfg.arch = gguf->get_str("general.architecture", "unknown");

    const std::string& arch = mcfg.arch;
    mcfg.n_layers    = static_cast<int>(gguf->get_u64(arch + ".block_count",       0));
    mcfg.hidden_dim  = static_cast<int>(gguf->get_u64(arch + ".embedding_length",  0));
    mcfg.ffn_dim     = static_cast<int>(gguf->get_u64(arch + ".feed_forward_length",0));
    mcfg.n_heads     = static_cast<int>(gguf->get_u64(arch + ".attention.head_count", 0));
    mcfg.n_kv_heads  = static_cast<int>(gguf->get_u64(arch + ".attention.head_count_kv",
                                        static_cast<uint64_t>(mcfg.n_heads)));
    mcfg.max_seq_len = static_cast<int>(gguf->get_u64(arch + ".context_length", 4096));
    mcfg.rms_norm_eps= gguf->get_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    mcfg.rope_theta  = gguf->get_f32(arch + ".rope.freq_base", 10000.0f);

    if (mcfg.n_heads > 0 && mcfg.hidden_dim > 0)
        mcfg.head_dim = mcfg.hidden_dim / mcfg.n_heads;

    // vocab_size = length of tokenizer.ggml.tokens array
    const size_t vocab_count = gguf->get_array_count("tokenizer.ggml.tokens");
    mcfg.vocab_size = static_cast<int>(vocab_count > 0 ? vocab_count : 0);

    // Primary weight dtype: inspect the first tensor
    if (gguf->n_tensors() > 0) {
        mcfg.weight_dtype = gguf->tensor_info(0).dtype;
    }

    // ---- 3. BOS / EOS token ids ----
    const int32_t bos_id = gguf->get_i32("tokenizer.ggml.bos_token_id", 1);
    const int32_t eos_id = gguf->get_i32("tokenizer.ggml.eos_token_id", 2);

    // ---- 4. Create and initialise backend ----
    auto backend = BackendRegistry::instance().create(cfg.backend);
    if (!backend->initialize()) {
        throw std::runtime_error(
            "Engine::from_gguf: backend '" + cfg.backend + "' failed to initialise");
    }

    // ---- 5. Build weight map (zero-copy views into mmap region) ----
    std::unordered_map<std::string, Tensor> weights;
    weights.reserve(gguf->n_tensors());
    for (size_t i = 0; i < gguf->n_tensors(); ++i) {
        const auto& ti = gguf->tensor_info(i);
        weights.emplace(ti.name, gguf->weight_tensor(i));
    }

    // ---- 6. Assemble Engine::Impl ----
    auto impl              = std::make_unique<Engine::Impl>();
    impl->engine_cfg       = std::move(cfg);
    impl->model_cfg        = std::move(mcfg);
    impl->backend          = std::move(backend);
    impl->gguf             = std::move(gguf);
    impl->weights          = std::move(weights);
    impl->bos_token_id     = bos_id;
    impl->eos_token_id     = eos_id;

    // ---- Extract vocabulary from tokenizer.ggml.tokens ----
    if (const auto* v = impl->gguf->find_value("tokenizer.ggml.tokens")) {
        if (const auto* arr = std::get_if<GgufArray>(v)) {
            impl->vocabulary = arr->strings;
        }
    }

    if (impl->engine_cfg.verbose) {
        const auto& mc = impl->model_cfg;
        std::fprintf(stderr,
            "[AxonForge] Loaded '%.*s'  arch=%s  layers=%d  hidden=%d"
            "  heads=%d/%d  vocab=%d  tensors=%zu\n",
            static_cast<int>(path.size()), path.data(),
            mc.arch.c_str(), mc.n_layers, mc.hidden_dim,
            mc.n_heads, mc.n_kv_heads, mc.vocab_size,
            impl->weights.size());
    }

    return Engine(std::move(impl));
}

// Helper: is this a LLaMA-family arch string?
static bool is_llama_arch(const std::string& arch) {
    return arch == "llama" || arch == "llama2" || arch == "llama3"
        || arch == "mistral" || arch == "tinyllama";
}

std::vector<int32_t> Engine::encode(std::string_view text, bool /*add_bos*/) const {
    if (impl_->model_cfg.arch == "gpt2")
        return gpt2_encode_simple(*this, text);
    if (is_llama_arch(impl_->model_cfg.arch))
        return llama_encode_simple(*this, text);
    throw std::runtime_error(
        "Engine::encode: no tokenizer for arch '" + impl_->model_cfg.arch + "'");
}
std::string Engine::decode(std::span<const int32_t> ids) const {
    if (impl_->model_cfg.arch == "gpt2") {
        std::vector<int32_t> v(ids.begin(), ids.end());
        return gpt2_decode_tokens(*this, v);
    }
    if (is_llama_arch(impl_->model_cfg.arch)) {
        std::vector<int32_t> v(ids.begin(), ids.end());
        return llama_decode_tokens(*this, v);
    }
    throw std::runtime_error(
        "Engine::decode: no tokenizer for arch '" + impl_->model_cfg.arch + "'");
}

Session Engine::new_session() const {
    auto s_impl = std::make_unique<Session::Impl>();
    s_impl->engine_cfg  = &impl_->engine_cfg;
    s_impl->model_cfg   = &impl_->model_cfg;
    s_impl->owner       = this;
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
    std::span<const int32_t> prompt_ids,
    KvState&                 /*kv*/,
    const SamplerConfig&     smp,
    int                      max_new_tokens,
    int                      /*eos_id*/)
{
    if (!impl_->owner)
        throw std::runtime_error("Session::generate: no owner engine");
    const Engine& eng = *impl_->owner;
    if (eng.model_config().arch == "gpt2") {
        Gpt2Config cfg;
        cfg.max_new_tokens = max_new_tokens;
        cfg.temperature    = smp.temperature;
        cfg.top_k          = smp.top_k > 0 ? smp.top_k : 40;
        cfg.verbose        = eng.engine_config().verbose;
        std::vector<int32_t> prompt(prompt_ids.begin(), prompt_ids.end());
        return gpt2_generate(eng, prompt, cfg);
    }
    if (is_llama_arch(eng.model_config().arch)) {
        LlamaConfig cfg;
        cfg.max_new_tokens  = max_new_tokens;
        cfg.temperature     = smp.temperature;
        cfg.top_k           = smp.top_k > 0 ? smp.top_k : 40;
        cfg.verbose         = eng.engine_config().verbose;
        cfg.max_context_len = 4096;
        std::vector<int32_t> prompt(prompt_ids.begin(), prompt_ids.end());
        return llama_generate(eng, prompt, cfg);
    }
    throw std::runtime_error(
        "Session::generate: not implemented for arch '" +
        eng.model_config().arch + "'");
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
