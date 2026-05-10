#pragma once
#include "axonforge/engine.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace axonforge {

// ============================================================
// GPT-2 generation configuration
// ============================================================
struct Gpt2Config {
    int   max_new_tokens = 200;
    float temperature    = 1.0f;
    int   top_k          = 40;
    bool  verbose        = false;
    int   n_threads      = 0;   // 0 = auto (min(8, nproc/2))

    // Called immediately after each new token is sampled.
    // Receives the raw token id; call gpt2_decode_tokens(engine, {id})
    // inside the callback to get the text fragment.
    // Leave null to disable streaming (tokens returned as a batch).
    std::function<void(int32_t /*token_id*/)> on_token;
};

// ============================================================
// GPT-2 autoregressive text generation.
//
// engine    : loaded via Engine::from_gguf() from a GPT-2 GGUF file
// prompt_ids: input token IDs (use gpt2_encode_simple or raw IDs)
// cfg       : generation parameters
//
// Returns all generated token IDs (NOT including the prompt).
// Uses greedy (top-1) or top-k sampling depending on cfg.
//
// The Engine MUST outlive the return value.
// ============================================================
std::vector<int32_t> gpt2_generate(
    const Engine&              engine,
    const std::vector<int32_t>& prompt_ids,
    const Gpt2Config&           cfg = {});

// ============================================================
// Decode GPT-2 token IDs → UTF-8 string.
// Applies the byte-level BPE inverse mapping so output is
// human-readable (e.g. 'Ġhello' → ' hello').
// ============================================================
std::string gpt2_decode_tokens(
    const Engine&               engine,
    const std::vector<int32_t>& token_ids);

// ============================================================
// Very simple word-piece tokenizer for ASCII prompts.
// Splits on whitespace/punctuation and does longest-match
// against the vocabulary.  For proper BPE, supply token IDs
// directly (e.g. using tiktoken / the Python side).
// ============================================================
std::vector<int32_t> gpt2_encode_simple(
    const Engine&    engine,
    std::string_view text);

} // namespace axonforge
