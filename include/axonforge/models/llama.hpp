#pragma once
#include "axonforge/engine.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace axonforge {

// ============================================================
// LLaMA generation configuration
//
// Covers LLaMA-2, LLaMA-3, LLaMA-3.2, TinyLlama, Mistral, etc.
// All models sharing the RMSNorm + RoPE + SwiGLU + GQA architecture.
// ============================================================
struct LlamaConfig {
    int   max_new_tokens  = 200;
    float temperature     = 1.0f;
    int   top_k           = 40;
    bool  verbose         = false;
    int   n_threads       = 0;    // 0 = auto (min(8, nproc/2))

    // Hard cap on the KV cache size (context window).
    // LLaMA-3 declares 128K context; allocating that naively costs ~32 GB.
    // Keep at 4096 unless you explicitly need longer sequences.
    int   max_context_len = 4096;

    // Streaming callback: called immediately after each new token is sampled.
    // Receives raw token id; call llama_decode_tokens(engine, {id}) for text.
    std::function<void(int32_t /*token_id*/)> on_token;
};

// ============================================================
// LLaMA autoregressive text generation.
//
// engine    : loaded via Engine::from_gguf() from a LLaMA-family GGUF
// prompt_ids: input token IDs (use llama_encode_simple or raw IDs)
// cfg       : generation parameters
//
// Returns all generated token IDs (NOT including the prompt).
// ============================================================
[[nodiscard]] std::vector<int32_t> llama_generate(
    const Engine&               engine,
    std::vector<int32_t>        prompt_ids,
    const LlamaConfig&          cfg = {});

// ============================================================
// Decode LLaMA token IDs → UTF-8 string.
// LLaMA GGUF tokens store raw UTF-8 bytes directly (or with
// the SentencePiece ▁ prefix for spaces), so this joins them.
// ============================================================
std::string llama_decode_tokens(
    const Engine&               engine,
    const std::vector<int32_t>& token_ids);

// ============================================================
// Simple greedy tokenizer — longest-match against vocabulary.
// Sufficient for throughput benchmarking and simple prompts.
// ============================================================
std::vector<int32_t> llama_encode_simple(
    const Engine&    engine,
    std::string_view text);

} // namespace axonforge
