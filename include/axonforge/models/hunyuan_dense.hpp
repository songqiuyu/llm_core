#pragma once
#include "axonforge/engine.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace axonforge {

struct HunyuanDenseConfig {
    int   max_new_tokens  = 200;
    float temperature     = 1.0f;
    int   top_k           = 40;
    bool  verbose         = false;
    int   n_threads       = 0;
    int   max_context_len = 4096;

    float top_p             = 1.0f;
    float rep_penalty       = 1.0f;
    int   rep_penalty_last_n= 64;

    std::function<void(int32_t)> on_token;
};

[[nodiscard]] std::vector<int32_t> hunyuan_dense_generate(
    const Engine&              engine,
    std::vector<int32_t>       prompt_ids,
    const HunyuanDenseConfig&  cfg = {});

std::vector<int32_t> hunyuan_dense_encode(
    const Engine&    engine,
    std::string_view text,
    bool             raw = false);

std::string hunyuan_dense_decode(
    const Engine&               engine,
    const std::vector<int32_t>& token_ids);

} // namespace axonforge
