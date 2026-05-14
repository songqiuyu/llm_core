#include "axonforge/models/llama_cuda.hpp"
#include <cstdio>
#include <utility>

namespace axonforge {

std::vector<int32_t> llama_cuda_generate(
    const Engine& engine,
    std::vector<int32_t> prompt_ids,
    const LlamaConfig& cfg) {
    if (cfg.verbose) {
        std::fprintf(stderr,
            "[AxonForge] CUDA generation path is experimental: backend/kernels are enabled; "
            "falling back to CPU forward until full GPU-resident weights/KV cache are complete.\n");
    }
    return llama_generate_cpu_impl(engine, std::move(prompt_ids), cfg);
}

} // namespace axonforge
