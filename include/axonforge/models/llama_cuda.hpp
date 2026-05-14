#pragma once
#include "axonforge/models/llama.hpp"

namespace axonforge {

[[nodiscard]] std::vector<int32_t> llama_cuda_generate(
    const Engine&        engine,
    std::vector<int32_t> prompt_ids,
    const LlamaConfig&   cfg);

[[nodiscard]] std::vector<int32_t> llama_generate_cpu_impl(
    const Engine&        engine,
    std::vector<int32_t> prompt_ids,
    const LlamaConfig&   cfg);

} // namespace axonforge

