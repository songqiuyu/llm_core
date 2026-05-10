#pragma once
#include "axonforge/version.hpp"
#include <cstdint>
#include <string_view>

namespace axonforge {

// ---- Hardware device types ----
enum class DeviceType : uint8_t {
    CPU    = 0,
    CUDA   = 1,
    Vulkan = 2,
    Metal  = 3,
    NPU    = 4,
};

// Identifies a specific device (type + index).
// e.g. DeviceId::cuda(0) = first CUDA GPU
struct DeviceId {
    DeviceType type{DeviceType::CPU};
    int32_t    index{0};

    [[nodiscard]] constexpr bool operator==(const DeviceId&) const noexcept = default;
    [[nodiscard]] constexpr bool is_cpu()  const noexcept { return type == DeviceType::CPU;  }
    [[nodiscard]] constexpr bool is_cuda() const noexcept { return type == DeviceType::CUDA; }

    static constexpr DeviceId cpu(int idx = 0)    { return {DeviceType::CPU,    idx}; }
    static constexpr DeviceId cuda(int idx = 0)   { return {DeviceType::CUDA,   idx}; }
    static constexpr DeviceId vulkan(int idx = 0) { return {DeviceType::Vulkan, idx}; }
    static constexpr DeviceId metal(int idx = 0)  { return {DeviceType::Metal,  idx}; }
    static constexpr DeviceId npu(int idx = 0)    { return {DeviceType::NPU,    idx}; }
};

[[nodiscard]] constexpr std::string_view device_type_name(DeviceType t) noexcept {
    switch (t) {
        case DeviceType::CPU:    return "cpu";
        case DeviceType::CUDA:   return "cuda";
        case DeviceType::Vulkan: return "vulkan";
        case DeviceType::Metal:  return "metal";
        case DeviceType::NPU:    return "npu";
    }
    return "unknown";
}

// ---- Version ----
struct Version {
    int major, minor, patch;
    [[nodiscard]] static constexpr Version current() noexcept {
        return {AXONFORGE_VERSION_MAJOR,
                AXONFORGE_VERSION_MINOR,
                AXONFORGE_VERSION_PATCH};
    }
};

} // namespace axonforge
