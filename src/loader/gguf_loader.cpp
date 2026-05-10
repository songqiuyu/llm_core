#include "axonforge/engine.hpp"
#include <cstdio>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace axonforge {
// ============================================================
// GGUF Model Loader
//
// GGUF format (llama.cpp specification):
//   - Magic: 0x46554747 ("GGUF")
//   - Header: version, n_tensors, n_kv
//   - Metadata KV pairs (arch, tokenizer vocab, hyperparams, ...)
//   - Tensor info array (name, shape, dtype, offset)
//   - Tensor data (mmap'd, lazy page faults)
//
// TODO(Phase0): implement full GGUF parser.
//   Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// ============================================================

static constexpr uint32_t GGUF_MAGIC   = 0x46554747;
static constexpr uint32_t GGUF_VERSION = 3;

struct GgufHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

// TODO(Phase0): GgufReader class
// struct GgufReader {
//     int      fd    = -1;
//     void*    mmap_ptr = nullptr;
//     size_t   file_size = 0;
//
//     bool open(std::string_view path);
//     ModelConfig read_config();
//     Tensor      read_tensor(std::string_view name);
//     void        close();
// };

// Placeholder implementation — throws until Phase0 is complete.
Engine Engine::from_gguf(std::string_view path, EngineConfig config) {
    (void)path; (void)config;
    throw std::runtime_error("Engine::from_gguf: not yet implemented (Phase0 TODO)");
}

} // namespace axonforge
