#pragma once
#include <cstdint>
#include <cstddef>
#include <string_view>

namespace axonforge {

// ---- Data type enum ----
enum class DType : uint8_t {
    // Floating point
    F32   = 0,
    F16   = 1,
    BF16  = 2,
    // Quantized (GGUF-compatible)
    Q8_0  = 8,
    Q4_0  = 9,
    Q4_K_M = 10,
    Q3_K_M = 11,
    Q2_K  = 12,
    Q6_K  = 13,
    // Integer
    I32   = 16,
    I16   = 17,
    I8    = 18,
    // Boolean
    BOOL  = 24,
    // Sentinel
    UNKNOWN = 255,
};

// Returns the size in bytes of one element for non-quantized types.
// For quantized types, returns the average bits-per-element / 8.
[[nodiscard]] constexpr size_t dtype_element_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::F32:    return 4;
        case DType::F16:    return 2;
        case DType::BF16:   return 2;
        case DType::Q8_0:   return 1;   // approx; actual is block-based
        case DType::Q4_0:   return 1;   // approx (0.5 byte/element rounded up)
        case DType::Q4_K_M: return 1;   // approx
        case DType::Q3_K_M: return 1;   // approx
        case DType::Q2_K:   return 1;   // approx
        case DType::I32:    return 4;
        case DType::I16:    return 2;
        case DType::I8:     return 1;
        case DType::BOOL:   return 1;
        default:            return 0;
    }
}

[[nodiscard]] constexpr bool dtype_is_quantized(DType dtype) noexcept {
    return dtype >= DType::Q8_0 && dtype <= DType::Q2_K;
}

[[nodiscard]] constexpr bool dtype_is_float(DType dtype) noexcept {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16;
}

[[nodiscard]] constexpr std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::F32:    return "f32";
        case DType::F16:    return "f16";
        case DType::BF16:   return "bf16";
        case DType::Q8_0:   return "q8_0";
        case DType::Q4_0:   return "q4_0";
        case DType::Q4_K_M: return "q4_k_m";
        case DType::Q3_K_M: return "q3_k_m";
        case DType::Q2_K:   return "q2_k";
        case DType::Q6_K:   return "q6_k";
        case DType::I32:    return "i32";
        case DType::I16:    return "i16";
        case DType::I8:     return "i8";
        case DType::BOOL:   return "bool";
        default:            return "unknown";
    }
}

// ============================================================
// GGUF / GGML interop helpers
// ============================================================

// Byte size of a scalar GGUF KV value type (0 for STRING/ARRAY).
[[nodiscard]] constexpr size_t gguf_type_pod_size(int32_t gguf_type) noexcept {
    switch (gguf_type) {
        case 0:  return 1;  // UINT8
        case 1:  return 1;  // INT8
        case 2:  return 2;  // UINT16
        case 3:  return 2;  // INT16
        case 4:  return 4;  // UINT32
        case 5:  return 4;  // INT32
        case 6:  return 4;  // FLOAT32
        case 7:  return 1;  // BOOL (stored as int8_t in GGUF)
        case 8:  return 0;  // STRING (variable-length)
        case 9:  return 0;  // ARRAY  (variable-length)
        case 10: return 8;  // UINT64
        case 11: return 8;  // INT64
        case 12: return 8;  // FLOAT64
        default: return 0;
    }
}

// Map GGML tensor type (int32_t stored in file) → AxonForge DType.
[[nodiscard]] constexpr DType ggml_type_to_dtype(int32_t ggml_type) noexcept {
    switch (ggml_type) {
        case 0:  return DType::F32;
        case 1:  return DType::F16;
        case 2:  return DType::Q4_0;
        case 8:  return DType::Q8_0;
        case 10: return DType::Q2_K;
        case 11: return DType::Q3_K_M;
        case 12: return DType::Q4_K_M;
        case 14: return DType::Q6_K;
        case 24: return DType::I8;
        case 25: return DType::I16;
        case 26: return DType::I32;
        case 30: return DType::BF16;
        default: return DType::UNKNOWN;
    }
}

// Exact byte count for `numel` elements of a given DType.
// Block-quantized types are rounded up to their block boundary.
//
// Block sizes (GGML standard):
//   Q8_0 : block=32  → 34 bytes/block  (2 B f16 scale + 32 B int8)
//   Q4_0 : block=32  → 18 bytes/block  (2 B f16 delta + 16 B packed nibbles)
//   Q4_K : block=256 → 144 bytes/block
//   Q3_K : block=256 → 110 bytes/block
//   Q2_K : block=256 →  84 bytes/block
[[nodiscard]] inline size_t gguf_tensor_nbytes(DType dtype, int64_t numel) noexcept {
    if (numel <= 0) return 0;
    const auto n = static_cast<size_t>(numel);
    switch (dtype) {
        case DType::F32:    return n * 4;
        case DType::F16:
        case DType::BF16:   return n * 2;
        case DType::I8:     return n;
        case DType::I16:    return n * 2;
        case DType::I32:    return n * 4;
        case DType::Q8_0:   return ((n + 31)  / 32)  * 34;
        case DType::Q4_0:   return ((n + 31)  / 32)  * 18;
        case DType::Q4_K_M: return ((n + 255) / 256) * 144;
        case DType::Q3_K_M: return ((n + 255) / 256) * 110;
        case DType::Q2_K:   return ((n + 255) / 256) * 84;
        case DType::Q6_K:   return ((n + 255) / 256) * 210;
        default:            return 0;
    }
}

} // namespace axonforge
