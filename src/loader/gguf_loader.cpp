// ============================================================
// GgufReader — mmap-based GGUF v2/v3 parser (AxonForge Phase 0).
//
// Binary layout (little-endian):
//   [4 B]  magic "GGUF"
//   [4 B]  version (uint32_t, must be 2 or 3)
//   [8 B]  n_tensors (int64_t)
//   [8 B]  n_kv      (int64_t)
//   [...]  n_kv  × KV pair
//   [...]  n_tensors × TensorInfo
//   [pad]  alignment padding
//   [...]  tensor data blob
//
// KV pair:
//   string key | int32 type | value
//   (if type==ARRAY: int32 elem_type | uint64 count | elements)
//
// TensorInfo:
//   string name | uint32 n_dims | int64 dim[n_dims] | int32 ggml_type | uint64 offset
//
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// ============================================================

#include "gguf_reader.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace axonforge {

// ---- Destructor: release mmap and close fd ----

GgufReader::~GgufReader() {
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, file_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

// ---- open() ----

void GgufReader::open(std::string_view path) {
    const std::string p(path);

    fd_ = ::open(p.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("GgufReader: cannot open '" + p + "'");
    }

    struct stat st{};
    if (fstat(fd_, &st) != 0) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("GgufReader: fstat failed for '" + p + "'");
    }
    file_size_ = static_cast<size_t>(st.st_size);

    if (file_size_ < 16) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("GgufReader: file too small (< 16 bytes)");
    }

    mmap_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("GgufReader: mmap failed for '" + p + "'");
    }

    begin_  = static_cast<const uint8_t*>(mmap_ptr_);
    cursor_ = begin_;
    end_    = begin_ + file_size_;

    parse();
}

// ---- parse() — linear cursor scan ----

void GgufReader::parse() {
    // ---- magic ----
    check_bounds(4);
    if (cursor_[0] != 'G' || cursor_[1] != 'G' ||
        cursor_[2] != 'U' || cursor_[3] != 'F') {
        throw std::runtime_error("GgufReader: invalid magic (not a GGUF file)");
    }
    cursor_ += 4;

    // ---- version (uint32_t) ----
    version_ = read_pod<uint32_t>();
    if (version_ < 2 || version_ > 3) {
        throw std::runtime_error(
            "GgufReader: unsupported GGUF version " + std::to_string(version_) +
            " (supported: 2, 3)");
    }

    // ---- counts (int64_t each) ----
    const int64_t n_tensors = read_pod<int64_t>();
    const int64_t n_kv      = read_pod<int64_t>();
    if (n_tensors < 0 || n_kv < 0) {
        throw std::runtime_error("GgufReader: negative tensor/kv counts");
    }

    // ---- KV pairs ----
    kv_.reserve(static_cast<size_t>(n_kv));
    for (int64_t i = 0; i < n_kv; ++i) {
        std::string key = read_string();
        int32_t     type_tag = read_pod<int32_t>();
        kv_.emplace(std::move(key), read_kv_value(type_tag));
    }

    // Extract alignment (may appear in KV)
    auto it = kv_.find("general.alignment");
    if (it != kv_.end()) {
        if (auto* v = std::get_if<uint32_t>(&it->second)) {
            alignment_ = *v;
        }
    }

    // ---- TensorInfo entries ----
    tensors_.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo info;
        info.name = read_string();

        const uint32_t n_dims = read_pod<uint32_t>();
        if (n_dims > 4) {
            throw std::runtime_error(
                "GgufReader: tensor '" + info.name +
                "' has more than 4 dimensions (" + std::to_string(n_dims) + ")");
        }
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.shape[d] = read_pod<int64_t>();
            if (info.shape[d] < 0) {
                throw std::runtime_error(
                    "GgufReader: tensor '" + info.name +
                    "' has negative dimension");
            }
        }

        info.ggml_type = read_pod<int32_t>();
        info.dtype     = ggml_type_to_dtype(info.ggml_type);
        info.offset    = read_pod<uint64_t>();

        // Compute element count (GGML: innermost dim is ne[0])
        int64_t ne = 1;
        for (int64_t d : info.shape) ne *= d;
        info.nbytes = gguf_tensor_nbytes(info.dtype, ne);

        tensor_index_[info.name] = tensors_.size();
        tensors_.push_back(std::move(info));
    }

    // ---- Data blob start (aligned) ----
    const size_t pos = static_cast<size_t>(cursor_ - begin_);
    // Round up to alignment_
    tensor_data_base_ = (pos + alignment_ - 1) / alignment_ * alignment_;
}

// ---- Parsing helpers ----

void GgufReader::check_bounds(size_t n) const {
    if (cursor_ + n > end_) {
        throw std::runtime_error("GgufReader: unexpected end of file (truncated)");
    }
}

std::string GgufReader::read_string() {
    const uint64_t len = read_pod<uint64_t>();
    check_bounds(static_cast<size_t>(len));
    std::string s(reinterpret_cast<const char*>(cursor_),
                  static_cast<size_t>(len));
    cursor_ += len;
    return s;
}

GgufArray GgufReader::read_array() {
    GgufArray arr;
    arr.elem_type = read_pod<int32_t>();
    const uint64_t count = read_pod<uint64_t>();
    arr.count = static_cast<size_t>(count);

    if (arr.elem_type == 8) {  // STRING elements
        arr.strings.reserve(arr.count);
        for (size_t i = 0; i < arr.count; ++i) {
            arr.strings.push_back(read_string());
        }
    } else {
        const size_t esz = gguf_type_pod_size(arr.elem_type);
        if (esz == 0) {
            throw std::runtime_error(
                "GgufReader: nested ARRAY or unknown type in array");
        }
        const size_t total = arr.count * esz;
        check_bounds(total);
        arr.data.resize(total);
        std::memcpy(arr.data.data(), cursor_, total);
        cursor_ += total;
    }
    return arr;
}

GgufValue GgufReader::read_kv_value(int32_t type_tag) {
    switch (type_tag) {
        case 0:  return read_pod<uint8_t>();
        case 1:  return read_pod<int8_t>();
        case 2:  return read_pod<uint16_t>();
        case 3:  return read_pod<int16_t>();
        case 4:  return read_pod<uint32_t>();
        case 5:  return read_pod<int32_t>();
        case 6:  return read_pod<float>();
        case 7:  { auto b = read_pod<int8_t>(); return static_cast<bool>(b); }
        case 8:  return read_string();
        case 9:  return read_array();
        case 10: return read_pod<uint64_t>();
        case 11: return read_pod<int64_t>();
        case 12: return read_pod<double>();
        default:
            throw std::runtime_error(
                "GgufReader: unknown KV type tag " + std::to_string(type_tag));
    }
}

// ---- KV accessors ----

bool GgufReader::has_key(std::string_view key) const {
    return kv_.count(std::string(key)) != 0;
}

const GgufValue* GgufReader::find_value(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    return it != kv_.end() ? &it->second : nullptr;
}

std::string GgufReader::get_str(std::string_view key, std::string_view def) const {
    const GgufValue* v = find_value(key);
    if (!v) return std::string(def);
    if (auto* p = std::get_if<std::string>(v)) return *p;
    return std::string(def);
}

// Helper: promote any unsigned/signed integer KV value to uint64_t
static uint64_t coerce_u64(const GgufValue& v, uint64_t def) {
    if (auto* p = std::get_if<uint64_t>(&v)) return *p;
    if (auto* p = std::get_if<uint32_t>(&v)) return *p;
    if (auto* p = std::get_if<int64_t> (&v)) return static_cast<uint64_t>(*p);
    if (auto* p = std::get_if<int32_t> (&v)) return static_cast<uint64_t>(*p);
    if (auto* p = std::get_if<uint16_t>(&v)) return *p;
    if (auto* p = std::get_if<int16_t> (&v)) return static_cast<uint64_t>(*p);
    if (auto* p = std::get_if<uint8_t> (&v)) return *p;
    if (auto* p = std::get_if<int8_t>  (&v)) return static_cast<uint64_t>(*p);
    return def;
}

uint64_t GgufReader::get_u64(std::string_view key, uint64_t def) const {
    const GgufValue* v = find_value(key);
    return v ? coerce_u64(*v, def) : def;
}

uint32_t GgufReader::get_u32(std::string_view key, uint32_t def) const {
    return static_cast<uint32_t>(get_u64(key, def));
}

int32_t GgufReader::get_i32(std::string_view key, int32_t def) const {
    const GgufValue* v = find_value(key);
    if (!v) return def;
    if (auto* p = std::get_if<int32_t> (v)) return *p;
    if (auto* p = std::get_if<uint32_t>(v)) return static_cast<int32_t>(*p);
    if (auto* p = std::get_if<int64_t> (v)) return static_cast<int32_t>(*p);
    if (auto* p = std::get_if<uint64_t>(v)) return static_cast<int32_t>(*p);
    return def;
}

float GgufReader::get_f32(std::string_view key, float def) const {
    const GgufValue* v = find_value(key);
    if (!v) return def;
    if (auto* p = std::get_if<float> (v)) return *p;
    if (auto* p = std::get_if<double>(v)) return static_cast<float>(*p);
    return def;
}

bool GgufReader::get_bool(std::string_view key, bool def) const {
    const GgufValue* v = find_value(key);
    if (!v) return def;
    if (auto* p = std::get_if<bool>(v)) return *p;
    return def;
}

size_t GgufReader::get_array_count(std::string_view key) const {
    const GgufValue* v = find_value(key);
    if (!v) return 0;
    if (auto* arr = std::get_if<GgufArray>(v)) return arr->count;
    return 0;
}

// ---- Tensor accessors ----

size_t GgufReader::n_tensors() const noexcept { return tensors_.size(); }

const GgufTensorInfo& GgufReader::tensor_info(size_t i) const {
    return tensors_.at(i);
}

const GgufTensorInfo* GgufReader::find_tensor(std::string_view name) const {
    auto it = tensor_index_.find(std::string(name));
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

const std::vector<GgufTensorInfo>& GgufReader::tensors() const noexcept {
    return tensors_;
}

Tensor GgufReader::weight_tensor(size_t idx) const {
    const auto& info = tensors_.at(idx);
    void* ptr = const_cast<void*>(static_cast<const void*>(
        begin_ + tensor_data_base_ + info.offset));
    return Tensor::from_raw_blob(ptr, info.nbytes, info.shape, info.dtype);
}

Tensor GgufReader::weight_tensor(std::string_view name) const {
    const GgufTensorInfo* info = find_tensor(name);
    if (!info) {
        throw std::runtime_error(
            "GgufReader: tensor '" + std::string(name) + "' not found");
    }
    void* ptr = const_cast<void*>(static_cast<const void*>(
        begin_ + tensor_data_base_ + info->offset));
    return Tensor::from_raw_blob(ptr, info->nbytes, info->shape, info->dtype);
}

// ---- Raw access ----

const uint8_t* GgufReader::mmap_base() const noexcept {
    return begin_;
}

size_t GgufReader::file_size() const noexcept {
    return file_size_;
}

} // namespace axonforge

