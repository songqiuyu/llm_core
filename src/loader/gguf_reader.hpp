#pragma once
#include "axonforge/dtype.hpp"
#include "axonforge/tensor.hpp"
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <variant>

namespace axonforge {

// ============================================================
// GgufArray — typed array KV value (POD or string elements).
// ============================================================
struct GgufArray {
    int32_t                  elem_type{0};   // gguf_type of elements (int32 in file)
    size_t                   count{0};
    std::vector<uint8_t>     data;           // raw bytes for POD elements
    std::vector<std::string> strings;        // used when elem_type == 8 (STRING)
};

// ============================================================
// GgufValue — all possible GGUF KV value types.
//
// Index mapping (mirrors gguf_type enum):
//   0 u8 | 1 i8 | 2 u16 | 3 i16 | 4 u32 | 5 i32 | 6 f32 | 7 bool
//   8 string | 9 array (GgufArray) | 10 u64 | 11 i64 | 12 f64
// ============================================================
using GgufValue = std::variant<
    uint8_t,    // 0
    int8_t,     // 1
    uint16_t,   // 2
    int16_t,    // 3
    uint32_t,   // 4
    int32_t,    // 5
    float,      // 6
    bool,       // 7
    std::string,// 8
    GgufArray,  // 9  (ARRAY)
    uint64_t,   // 10
    int64_t,    // 11
    double      // 12
>;

// ============================================================
// GgufTensorInfo — metadata for one weight tensor.
// ============================================================
struct GgufTensorInfo {
    std::string            name;
    std::vector<int64_t>   shape;      // GGML dimension order (innermost first)
    int32_t                ggml_type{0};
    DType                  dtype{DType::UNKNOWN};
    uint64_t               offset{0};  // byte offset within tensor data blob
    size_t                 nbytes{0};  // exact storage size in bytes
};

// ============================================================
// GgufReader — mmap-based GGUF v2/v3 parser.
//
// Usage:
//   auto r = std::make_unique<GgufReader>();
//   r->open("model.gguf");                   // throws on error
//   auto arch = r->get_str("general.architecture");
//   Tensor w  = r->weight_tensor("blk.0.attn_q.weight");
//
// The GgufReader MUST outlive any Tensor returned by weight_tensor()
// because those Tensors are zero-copy views into the mmap region.
// ============================================================
class GgufReader {
public:
    GgufReader()  = default;
    ~GgufReader();

    GgufReader(const GgufReader&)            = delete;
    GgufReader& operator=(const GgufReader&) = delete;
    GgufReader(GgufReader&&)                 = delete;

    // Open and fully parse the GGUF file.  Throws std::runtime_error on failure.
    void open(std::string_view path);

    // ---- Header fields ----
    uint32_t version()   const noexcept { return version_; }
    size_t   alignment() const noexcept { return alignment_; }

    // ---- KV metadata ----
    bool             has_key(std::string_view key) const;
    const GgufValue* find_value(std::string_view key) const;

    // Typed accessors — return default_val when the key is absent or the
    // stored type does not widen/narrow to T without loss.
    std::string get_str (std::string_view key, std::string_view def = "")   const;
    uint32_t    get_u32 (std::string_view key, uint32_t  def = 0)           const;
    uint64_t    get_u64 (std::string_view key, uint64_t  def = 0)           const;
    int32_t     get_i32 (std::string_view key, int32_t   def = 0)           const;
    float       get_f32 (std::string_view key, float     def = 0.f)         const;
    bool        get_bool(std::string_view key, bool      def = false)       const;

    // Number of elements in an ARRAY KV entry (0 if absent or not array).
    size_t get_array_count(std::string_view key) const;

    // ---- Tensor metadata ----
    size_t                             n_tensors() const noexcept;
    const GgufTensorInfo&              tensor_info(size_t i) const;
    const GgufTensorInfo*              find_tensor(std::string_view name) const;
    const std::vector<GgufTensorInfo>& tensors()   const noexcept;

    // Zero-copy Tensor view into the mmap'd weight data.
    // MUST NOT outlive this GgufReader.
    Tensor weight_tensor(size_t idx) const;
    Tensor weight_tensor(std::string_view name) const;

    // ---- Raw access (for advanced use) ----
    const uint8_t* mmap_base()          const noexcept;
    size_t         file_size()          const noexcept;
    size_t         tensor_data_offset() const noexcept { return tensor_data_base_; }

private:
    void parse();

    // Checked reads (throw on truncation)
    void check_bounds(size_t n) const;

    template<typename T>
    T read_pod() {
        check_bounds(sizeof(T));
        T v;
        __builtin_memcpy(&v, cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return v;
    }

    std::string read_string();
    GgufValue   read_kv_value(int32_t type_tag);
    GgufArray   read_array();

    // File / mmap state
    int    fd_{-1};
    void*  mmap_ptr_{nullptr};
    size_t file_size_{0};

    // Cursor into mmap region
    const uint8_t* begin_{nullptr};
    const uint8_t* cursor_{nullptr};
    const uint8_t* end_{nullptr};

    // Parsed metadata
    uint32_t version_{0};
    size_t   alignment_{32};
    size_t   tensor_data_base_{0};  // byte offset within file to tensor data blob

    std::unordered_map<std::string, GgufValue> kv_;
    std::vector<GgufTensorInfo>                tensors_;
    std::unordered_map<std::string, size_t>    tensor_index_;
};

} // namespace axonforge
