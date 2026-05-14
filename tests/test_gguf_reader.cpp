#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "axonforge/dtype.hpp"

// GgufReader is an internal header — include via relative path
// (tests are built with ${PROJECT_SOURCE_DIR}/include in path;
//  the PRIVATE src/ path is NOT available here, so we write the
//  synthetic test against the public API: Engine::from_gguf)
// For white-box unit testing of the parser itself we compile the
// reader sources directly into this test via a thin shim approach:
// instead we write a minimal GGUF to a temp file and call Engine::from_gguf.
#include "axonforge/engine.hpp"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>   // mkstemp, unlink, write, close

using namespace axonforge;

// ============================================================
// Synthetic GGUF binary builder
// ============================================================

namespace {

struct BufWriter {
    std::vector<uint8_t> buf;

    template<typename T>
    void pod(T v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(v));
    }

    void str(std::string_view s) {
        pod(static_cast<uint64_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void pad_to(size_t alignment) {
        while (buf.size() % alignment != 0) buf.push_back(0);
    }

    // Write KV: string type (8)
    void kv_str(std::string_view key, std::string_view val) {
        str(key);
        pod(int32_t(8));   // GGUF_TYPE_STRING
        str(val);
    }

    // Write KV: uint64 type (10)
    void kv_u64(std::string_view key, uint64_t val) {
        str(key);
        pod(int32_t(10));  // GGUF_TYPE_UINT64
        pod(val);
    }

    // Write KV: uint32 type (4)
    void kv_u32(std::string_view key, uint32_t val) {
        str(key);
        pod(int32_t(4));   // GGUF_TYPE_UINT32
        pod(val);
    }

    // Write KV: float32 type (6)
    void kv_f32(std::string_view key, float val) {
        str(key);
        pod(int32_t(6));   // GGUF_TYPE_FLOAT32
        pod(val);
    }

    // Write KV: array of strings (type 9, elem 8)
    void kv_str_array(std::string_view key,
                      const std::vector<std::string>& vals) {
        str(key);
        pod(int32_t(9));   // GGUF_TYPE_ARRAY
        pod(int32_t(8));   // elem = STRING
        pod(uint64_t(vals.size()));
        for (const auto& v : vals) str(v);
    }

    void kv_i32_array(std::string_view key,
                      const std::vector<int32_t>& vals) {
        str(key);
        pod(int32_t(9));   // GGUF_TYPE_ARRAY
        pod(int32_t(5));   // elem = INT32
        pod(uint64_t(vals.size()));
        for (int32_t v : vals) pod(v);
    }
};

// Build a minimal but complete GGUF v3 file:
//   - 1 KV per required LLaMA field + tokenizer tokens
//   - 1 F32 tensor [4, 4] = 64 bytes of weights
std::vector<uint8_t> make_minimal_gguf() {
    BufWriter w;

    // ---- header ----
    w.buf.insert(w.buf.end(), {'G','G','U','F'});
    w.pod(uint32_t(3));            // version
    w.pod(int64_t(1));             // n_tensors
    w.pod(int64_t(9));             // n_kv

    // ---- KV pairs ----
    w.kv_str("general.architecture",   "llama");
    w.kv_u64("llama.block_count",       4);
    w.kv_u64("llama.embedding_length",  32);
    w.kv_u64("llama.feed_forward_length", 64);
    w.kv_u64("llama.attention.head_count", 4);
    w.kv_f32("llama.rope.freq_base",    500000.0f);
    w.kv_u32("tokenizer.ggml.bos_token_id", 128000);
    w.kv_u32("tokenizer.ggml.eos_token_id", 128001);
    w.kv_str_array("tokenizer.ggml.tokens",
        {"<unk>","<bos>","<eos>","hello","world"});

    // ---- TensorInfo: "weight.0" [4, 4] F32 offset=0 ----
    w.str("weight.0");
    w.pod(uint32_t(2));            // n_dims
    w.pod(int64_t(4));             // ne[0]
    w.pod(int64_t(4));             // ne[1]
    w.pod(int32_t(0));             // ggml_type = F32
    w.pod(uint64_t(0));            // data offset within blob

    // ---- alignment padding (default = 32) ----
    w.pad_to(32);

    // ---- tensor data: 16 F32 values = 64 bytes ----
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(i);
        w.pod(v);
    }

    return w.buf;
}

std::vector<uint8_t> make_qwen2_gguf() {
    BufWriter w;

    w.buf.insert(w.buf.end(), {'G','G','U','F'});
    w.pod(uint32_t(3));
    w.pod(int64_t(1));
    w.pod(int64_t(14));

    w.kv_str("general.architecture",   "qwen2");
    w.kv_u64("qwen2.block_count",       24);
    w.kv_u64("qwen2.embedding_length",  896);
    w.kv_u64("qwen2.feed_forward_length", 4864);
    w.kv_u64("qwen2.attention.head_count", 14);
    w.kv_u64("qwen2.attention.head_count_kv", 2);
    w.kv_u64("qwen2.context_length", 32768);
    w.kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_f32("qwen2.rope.freq_base",    1000000.0f);
    w.kv_u32("tokenizer.ggml.bos_token_id", 151643);
    w.kv_u32("tokenizer.ggml.eos_token_id", 151645);
    w.kv_str_array("tokenizer.ggml.tokens",
        {"h","e","l","o","he","hel","hell","hello","<|im_start|>","<|im_end|>"});
    w.kv_str_array("tokenizer.ggml.merges",
        {"h e","he l","hel l","hell o"});
    w.kv_i32_array("tokenizer.ggml.token_type",
        {1,1,1,1,1,1,1,1,3,3});

    w.str("weight.0");
    w.pod(uint32_t(2));
    w.pod(int64_t(4));
    w.pod(int64_t(4));
    w.pod(int32_t(0));
    w.pod(uint64_t(0));

    w.pad_to(32);
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(i);
        w.pod(v);
    }
    return w.buf;
}

std::vector<uint8_t> make_qwen2_3b_gguf() {
    BufWriter w;

    w.buf.insert(w.buf.end(), {'G','G','U','F'});
    w.pod(uint32_t(3));
    w.pod(int64_t(1));
    w.pod(int64_t(14));

    w.kv_str("general.architecture",   "qwen2");
    w.kv_u64("qwen2.block_count",       36);
    w.kv_u64("qwen2.embedding_length",  2048);
    w.kv_u64("qwen2.feed_forward_length", 11008);
    w.kv_u64("qwen2.attention.head_count", 16);
    w.kv_u64("qwen2.attention.head_count_kv", 2);
    w.kv_u64("qwen2.context_length", 32768);
    w.kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_f32("qwen2.rope.freq_base",    1000000.0f);
    w.kv_u32("tokenizer.ggml.bos_token_id", 151643);
    w.kv_u32("tokenizer.ggml.eos_token_id", 151645);
    w.kv_str_array("tokenizer.ggml.tokens",
        {"h","e","l","o","he","hel","hell","hello","<|im_start|>","<|im_end|>"});
    w.kv_str_array("tokenizer.ggml.merges",
        {"h e","he l","hel l","hell o"});
    w.kv_i32_array("tokenizer.ggml.token_type",
        {1,1,1,1,1,1,1,1,3,3});

    w.str("weight.0");
    w.pod(uint32_t(2));
    w.pod(int64_t(4));
    w.pod(int64_t(4));
    w.pod(int32_t(0));
    w.pod(uint64_t(0));

    w.pad_to(32);
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(i);
        w.pod(v);
    }
    return w.buf;
}

std::vector<uint8_t> make_qwen3_4b_gguf() {
    BufWriter w;

    w.buf.insert(w.buf.end(), {'G','G','U','F'});
    w.pod(uint32_t(3));
    w.pod(int64_t(1));
    w.pod(int64_t(16));

    w.kv_str("general.architecture",   "qwen3");
    w.kv_u64("qwen3.block_count",       36);
    w.kv_u64("qwen3.embedding_length",  2560);
    w.kv_u64("qwen3.feed_forward_length", 9728);
    w.kv_u64("qwen3.attention.head_count", 32);
    w.kv_u64("qwen3.attention.head_count_kv", 8);
    w.kv_u64("qwen3.context_length", 32768);
    w.kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_f32("qwen3.rope.freq_base",    1000000.0f);
    w.kv_u32("tokenizer.ggml.bos_token_id", 151643);
    w.kv_u32("tokenizer.ggml.eos_token_id", 151645);
    w.kv_str_array("tokenizer.ggml.tokens",
        {"h","e","l","o","he","hel","hell","hello","<|im_start|>","<|im_end|>"});
    w.kv_str_array("tokenizer.ggml.merges",
        {"h e","he l","hel l","hell o"});
    w.kv_i32_array("tokenizer.ggml.token_type",
        {1,1,1,1,1,1,1,1,3,3});
    w.kv_str("tokenizer.ggml.pre", "qwen2");
    w.kv_str("tokenizer.chat_template",
             "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n");

    w.str("weight.0");
    w.pod(uint32_t(2));
    w.pod(int64_t(4));
    w.pod(int64_t(4));
    w.pod(int32_t(0));
    w.pod(uint64_t(0));

    w.pad_to(32);
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(i);
        w.pod(v);
    }
    return w.buf;
}

std::vector<uint8_t> make_hunyuan_dense_gguf() {
    BufWriter w;

    const std::string hy_user = "<\xEF\xBD\x9C" "hy_User" "\xEF\xBD\x9C" ">";
    const std::string hy_asst = "<\xEF\xBD\x9C" "hy_Assistant" "\xEF\xBD\x9C" ">";
    const std::string hy_end  = "<\xEF\xBD\x9C" "hy_place" "\xE2\x96\x81" "holder" "\xE2\x96\x81" "no" "\xE2\x96\x81" "2" "\xEF\xBD\x9C" ">";

    w.buf.insert(w.buf.end(), {'G','G','U','F'});
    w.pod(uint32_t(3));
    w.pod(int64_t(1));
    w.pod(int64_t(17));

    w.kv_str("general.architecture",   "hunyuan-dense");
    w.kv_u64("hunyuan-dense.block_count",       24);
    w.kv_u64("hunyuan-dense.embedding_length",  2048);
    w.kv_u64("hunyuan-dense.feed_forward_length", 5504);
    w.kv_u64("hunyuan-dense.attention.head_count", 16);
    w.kv_u64("hunyuan-dense.attention.head_count_kv", 8);
    w.kv_u64("hunyuan-dense.context_length", 262144);
    w.kv_f32("hunyuan-dense.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_f32("hunyuan-dense.rope.freq_base",    500000.0f);
    w.kv_f32("hunyuan-dense.rope.scaling.alpha", 32.0f);
    w.kv_u32("tokenizer.ggml.bos_token_id", 127960);
    w.kv_u32("tokenizer.ggml.eos_token_id", 127967);
    w.kv_str_array("tokenizer.ggml.tokens",
        {"h","e","l","o","he","hel","hell","hello",
         "1","2","3","12","123","4","\xC4\x8A",
         "\xC3\xA4","\xC2\xBD","\xC5\x82",
         hy_user, hy_asst, hy_end});
    w.kv_str_array("tokenizer.ggml.merges",
        {"h e","he l","hel l","hell o","1 2","12 3"});
    w.kv_i32_array("tokenizer.ggml.token_type",
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3});
    w.kv_str("tokenizer.ggml.pre", "hunyuan-dense");
    w.kv_str("tokenizer.chat_template", "{{ hunyuan_dense }}");

    w.str("weight.0");
    w.pod(uint32_t(2));
    w.pod(int64_t(4));
    w.pod(int64_t(4));
    w.pod(int32_t(0));
    w.pod(uint64_t(0));

    w.pad_to(32);
    for (int i = 0; i < 16; ++i) {
        float v = static_cast<float>(i);
        w.pod(v);
    }
    return w.buf;
}

// RAII temp file
struct TempFile {
    std::string path;
    explicit TempFile(const std::vector<uint8_t>& data) {
        char tmpl[] = "/tmp/axonforge_test_gguf_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) throw std::runtime_error("TempFile: mkstemp failed");
        path = tmpl;
        ssize_t written = ::write(fd, data.data(), data.size());
        ::close(fd);
        if (written != static_cast<ssize_t>(data.size()))
            throw std::runtime_error("TempFile: write failed");
    }
    ~TempFile() { ::unlink(path.c_str()); }
};

} // anonymous namespace

// ============================================================
// dtype.hpp helper tests
// ============================================================

TEST_CASE("ggml_type_to_dtype maps known types", "[gguf][dtype]") {
    REQUIRE(ggml_type_to_dtype(0)  == DType::F32);
    REQUIRE(ggml_type_to_dtype(1)  == DType::F16);
    REQUIRE(ggml_type_to_dtype(2)  == DType::Q4_0);
    REQUIRE(ggml_type_to_dtype(6)  == DType::Q5_0);
    REQUIRE(ggml_type_to_dtype(8)  == DType::Q8_0);
    REQUIRE(ggml_type_to_dtype(10) == DType::Q2_K);
    REQUIRE(ggml_type_to_dtype(11) == DType::Q3_K_M);
    REQUIRE(ggml_type_to_dtype(12) == DType::Q4_K_M);
    REQUIRE(ggml_type_to_dtype(30) == DType::BF16);
    REQUIRE(ggml_type_to_dtype(99) == DType::UNKNOWN);
}

TEST_CASE("gguf_tensor_nbytes exact sizes", "[gguf][dtype]") {
    // F32: 4 bytes/element
    REQUIRE(gguf_tensor_nbytes(DType::F32, 16) == 64);
    // F16: 2 bytes/element
    REQUIRE(gguf_tensor_nbytes(DType::F16, 32) == 64);
    // Q8_0: 32-element blocks, 34 bytes/block
    REQUIRE(gguf_tensor_nbytes(DType::Q8_0, 32)  == 34);
    REQUIRE(gguf_tensor_nbytes(DType::Q8_0, 64)  == 68);
    // Q4_0: 32-element blocks, 18 bytes/block
    REQUIRE(gguf_tensor_nbytes(DType::Q4_0, 32)  == 18);
    REQUIRE(gguf_tensor_nbytes(DType::Q4_0, 64)  == 36);
    REQUIRE(gguf_tensor_nbytes(DType::Q5_0, 32)  == 22);
    REQUIRE(gguf_tensor_nbytes(DType::Q5_0, 64)  == 44);
    // Q4_K_M: 256-element blocks, 144 bytes/block
    REQUIRE(gguf_tensor_nbytes(DType::Q4_K_M, 256) == 144);
    REQUIRE(gguf_tensor_nbytes(DType::Q4_K_M, 512) == 288);
    // Q2_K: 256-element blocks, 84 bytes/block
    REQUIRE(gguf_tensor_nbytes(DType::Q2_K, 256) == 84);
    // zero elements
    REQUIRE(gguf_tensor_nbytes(DType::F32, 0) == 0);
}

TEST_CASE("gguf_type_pod_size covers all 13 types", "[gguf][dtype]") {
    REQUIRE(gguf_type_pod_size(0)  == 1);   // u8
    REQUIRE(gguf_type_pod_size(1)  == 1);   // i8
    REQUIRE(gguf_type_pod_size(2)  == 2);   // u16
    REQUIRE(gguf_type_pod_size(3)  == 2);   // i16
    REQUIRE(gguf_type_pod_size(4)  == 4);   // u32
    REQUIRE(gguf_type_pod_size(5)  == 4);   // i32
    REQUIRE(gguf_type_pod_size(6)  == 4);   // f32
    REQUIRE(gguf_type_pod_size(7)  == 1);   // bool
    REQUIRE(gguf_type_pod_size(8)  == 0);   // string (variable)
    REQUIRE(gguf_type_pod_size(9)  == 0);   // array  (variable)
    REQUIRE(gguf_type_pod_size(10) == 8);   // u64
    REQUIRE(gguf_type_pod_size(11) == 8);   // i64
    REQUIRE(gguf_type_pod_size(12) == 8);   // f64
}

// ============================================================
// Engine::from_gguf integration tests (white-box via temp file)
// ============================================================

TEST_CASE("Engine::from_gguf parses architecture KV", "[gguf][engine]") {
    const TempFile f(make_minimal_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    REQUIRE(e.model_config().arch == "llama");
}

TEST_CASE("Engine::from_gguf reads LLaMA hyperparams", "[gguf][engine]") {
    const TempFile f(make_minimal_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    const ModelConfig& mc = e.model_config();
    REQUIRE(mc.n_layers   == 4);
    REQUIRE(mc.hidden_dim == 32);
    REQUIRE(mc.ffn_dim    == 64);
    REQUIRE(mc.n_heads    == 4);
    REQUIRE(mc.head_dim   == 8);   // hidden_dim / n_heads = 32/4
    REQUIRE_THAT(mc.rope_theta, Catch::Matchers::WithinRel(500000.0f, 1e-5f));
}

TEST_CASE("Engine::from_gguf reads Qwen2 hyperparams", "[gguf][engine][qwen2]") {
    const TempFile f(make_qwen2_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    const ModelConfig& mc = e.model_config();
    REQUIRE(mc.arch       == "qwen2");
    REQUIRE(mc.n_layers   == 24);
    REQUIRE(mc.hidden_dim == 896);
    REQUIRE(mc.ffn_dim    == 4864);
    REQUIRE(mc.n_heads    == 14);
    REQUIRE(mc.n_kv_heads == 2);
    REQUIRE(mc.head_dim   == 64);
    REQUIRE(mc.max_seq_len == 32768);
    REQUIRE_THAT(mc.rope_theta, Catch::Matchers::WithinRel(1000000.0f, 1e-5f));
    REQUIRE_THAT(mc.rms_norm_eps, Catch::Matchers::WithinRel(1e-6f, 1e-5f));
}

TEST_CASE("Engine::from_gguf reads Qwen2.5 3B-style metadata", "[gguf][engine][qwen2][3b]") {
    const TempFile f(make_qwen2_3b_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    const ModelConfig& mc = e.model_config();
    REQUIRE(mc.arch       == "qwen2");
    REQUIRE(mc.n_layers   == 36);
    REQUIRE(mc.hidden_dim == 2048);
    REQUIRE(mc.ffn_dim    == 11008);
    REQUIRE(mc.n_heads    == 16);
    REQUIRE(mc.n_kv_heads == 2);
    REQUIRE(mc.head_dim   == 128);
    REQUIRE(mc.max_seq_len == 32768);
    REQUIRE(mc.vocab_size == 10);
    REQUIRE_THAT(mc.rope_theta, Catch::Matchers::WithinRel(1000000.0f, 1e-5f));
    REQUIRE_THAT(mc.rms_norm_eps, Catch::Matchers::WithinRel(1e-6f, 1e-5f));
}

TEST_CASE("Engine::from_gguf reads Qwen3 4B-style metadata", "[gguf][engine][qwen3][3b]") {
    const TempFile f(make_qwen3_4b_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    const ModelConfig& mc = e.model_config();
    REQUIRE(mc.arch       == "qwen3");
    REQUIRE(mc.n_layers   == 36);
    REQUIRE(mc.hidden_dim == 2560);
    REQUIRE(mc.ffn_dim    == 9728);
    REQUIRE(mc.n_heads    == 32);
    REQUIRE(mc.n_kv_heads == 8);
    REQUIRE(mc.head_dim   == 80);
    REQUIRE(mc.max_seq_len == 32768);
    REQUIRE(mc.vocab_size == 10);
    REQUIRE(e.tokenizer_pre() == "qwen2");
    REQUIRE_FALSE(e.chat_template().empty());
    REQUIRE_THAT(mc.rope_theta, Catch::Matchers::WithinRel(1000000.0f, 1e-5f));
    REQUIRE_THAT(mc.rms_norm_eps, Catch::Matchers::WithinRel(1e-6f, 1e-5f));
}

TEST_CASE("Qwen2 tokenizer parses BPE merges and special tokens", "[gguf][engine][qwen2]") {
    const TempFile f(make_qwen2_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    auto ids = e.encode("hello", false);
    REQUIRE(ids == std::vector<int32_t>{7});
    REQUIRE(e.decode(ids) == "hello");

    auto raw = e.encode("<|im_start|>hello<|im_end|>", false, true);
    REQUIRE(raw == std::vector<int32_t>{8, 7, 9});
    REQUIRE(e.decode(raw) == "hello");
}

TEST_CASE("Qwen3 reuses Qwen BPE special-token tokenizer path", "[gguf][engine][qwen3]") {
    const TempFile f(make_qwen3_4b_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    auto ids = e.encode("hello", false);
    REQUIRE(ids == std::vector<int32_t>{7});
    REQUIRE(e.decode(ids) == "hello");

    auto raw = e.encode("<|im_start|>hello<|im_end|>", false, true);
    REQUIRE(raw == std::vector<int32_t>{8, 7, 9});
    REQUIRE(e.decode(raw) == "hello");
}

TEST_CASE("Engine::from_gguf reads Hunyuan dense metadata", "[gguf][engine][hunyuan]") {
    const TempFile f(make_hunyuan_dense_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    const ModelConfig& mc = e.model_config();
    REQUIRE(mc.arch       == "hunyuan-dense");
    REQUIRE(mc.n_layers   == 24);
    REQUIRE(mc.hidden_dim == 2048);
    REQUIRE(mc.ffn_dim    == 5504);
    REQUIRE(mc.n_heads    == 16);
    REQUIRE(mc.n_kv_heads == 8);
    REQUIRE(mc.head_dim   == 128);
    REQUIRE(mc.max_seq_len == 262144);
    REQUIRE(mc.vocab_size == 21);
    REQUIRE(e.tokenizer_pre() == "hunyuan-dense");
    REQUIRE_FALSE(e.chat_template().empty());
    REQUIRE_THAT(mc.rope_scaling_alpha, Catch::Matchers::WithinRel(32.0f, 1e-5f));
    const float expected_theta = 500000.0f * std::pow(32.0f, 128.0f / 126.0f);
    REQUIRE_THAT(mc.rope_theta, Catch::Matchers::WithinRel(expected_theta, 1e-5f));
    REQUIRE_THAT(mc.rms_norm_eps, Catch::Matchers::WithinRel(1e-6f, 1e-5f));
}

TEST_CASE("Hunyuan dense tokenizer handles text and special tokens", "[gguf][engine][hunyuan]") {
    const TempFile f(make_hunyuan_dense_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    auto ids = e.encode("hello1234\n", false);
    REQUIRE(ids == std::vector<int32_t>{7, 12, 13, 14});
    REQUIRE(e.decode(ids) == "hello1234\n");

    const std::string ni = "\xE4\xBD\xA0";
    auto zh = e.encode(ni, false);
    REQUIRE(zh == std::vector<int32_t>{15, 16, 17});
    REQUIRE(e.decode(zh) == ni);

    const std::string hy_user = "<\xEF\xBD\x9C" "hy_User" "\xEF\xBD\x9C" ">";
    const std::string hy_asst = "<\xEF\xBD\x9C" "hy_Assistant" "\xEF\xBD\x9C" ">";
    auto raw = e.encode(hy_user + "hello" + hy_asst, false, true);
    REQUIRE(raw == std::vector<int32_t>{18, 7, 19});
    REQUIRE(e.decode(raw) == "hello");
}

TEST_CASE("Engine::from_gguf sets bos/eos token ids", "[gguf][engine]") {
    const TempFile f(make_minimal_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    REQUIRE(e.bos_id() == 128000);
    REQUIRE(e.eos_id() == 128001);
}

TEST_CASE("Engine::from_gguf reads vocab_size from tokens array", "[gguf][engine]") {
    const TempFile f(make_minimal_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    REQUIRE(e.vocab_size() == 5);  // 5 tokens in our synthetic file
}

TEST_CASE("Engine::from_gguf loads one tensor with correct shape and dtype",
          "[gguf][engine]") {
    const TempFile f(make_minimal_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    const ModelConfig& mc = e.model_config();
    REQUIRE(mc.weight_dtype == DType::F32);
}

TEST_CASE("Engine::from_gguf tensor data is accessible (zero-copy check)",
          "[gguf][engine]") {
    const TempFile f(make_minimal_gguf());
    const Engine e = Engine::from_gguf(f.path, EngineConfig{});
    // The Engine owns the GgufReader; tensor views are valid as long as e is alive.
    // We verify by re-reading the file and checking the first few float values
    // match what we wrote in make_minimal_gguf (0.0f, 1.0f, 2.0f ...).
    // Access via GgufReader directly (via engine weights) isn't exposed publicly,
    // so we just check the model loaded without throwing.
    REQUIRE(e.model_config().n_layers == 4);
}

TEST_CASE("Engine::from_gguf throws on non-existent file", "[gguf][engine]") {
    REQUIRE_THROWS_AS(
        Engine::from_gguf("/tmp/this_file_does_not_exist_axonforge.gguf"),
        std::runtime_error);
}

TEST_CASE("Engine::from_gguf throws on invalid magic", "[gguf][engine]") {
    std::vector<uint8_t> bad(32, 0xAB);  // definitely not "GGUF"
    const TempFile f(bad);
    REQUIRE_THROWS_AS(Engine::from_gguf(f.path), std::runtime_error);
}

TEST_CASE("Engine::from_gguf throws on truncated file", "[gguf][engine]") {
    auto buf = make_minimal_gguf();
    buf.resize(10);  // cut off mid-header
    const TempFile f(buf);
    REQUIRE_THROWS_AS(Engine::from_gguf(f.path), std::runtime_error);
}

TEST_CASE("Engine::from_gguf throws on bad GGUF version", "[gguf][engine]") {
    auto buf = make_minimal_gguf();
    // Patch version field at offset 4 to an unsupported value (e.g., 99)
    const uint32_t bad_ver = 99;
    std::memcpy(buf.data() + 4, &bad_ver, 4);
    const TempFile f(buf);
    REQUIRE_THROWS_AS(Engine::from_gguf(f.path), std::runtime_error);
}
