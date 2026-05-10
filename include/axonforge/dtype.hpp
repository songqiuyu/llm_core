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
        case DType::I32:    return "i32";
        case DType::I16:    return "i16";
        case DType::I8:     return "i8";
        case DType::BOOL:   return "bool";
        default:            return "unknown";
    }
}

} // namespace axonforge
