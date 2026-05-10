#include "axonforge/engine.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace axonforge {
// ============================================================
// BPE Tokenizer
//
// Vocabulary and merge rules are loaded from GGUF metadata
// (key: "tokenizer.ggml.tokens", "tokenizer.ggml.merges").
// Implements the same byte-level BPE as llama.cpp / GPT-2.
//
// TODO(Phase0):
//   1. Load vocab / merges from GgufReader
//   2. Implement encode() using priority-queue merge loop
//   3. Implement decode() using vocab lookup + UTF-8 byte heal
// ============================================================

struct BpeTokenizer {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string>                 id_to_token;
    int32_t bos_id_{1};
    int32_t eos_id_{2};

    std::vector<int32_t> encode(std::string_view text, bool add_bos) const {
        (void)text; (void)add_bos;
        throw std::runtime_error("BpeTokenizer::encode: not yet implemented");
    }

    std::string decode(std::span<const int32_t> ids) const {
        (void)ids;
        throw std::runtime_error("BpeTokenizer::decode: not yet implemented");
    }
};

} // namespace axonforge
