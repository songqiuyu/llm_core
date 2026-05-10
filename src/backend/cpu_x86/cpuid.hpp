#pragma once

namespace axonforge::cpu_x86 {

// CPU ISA feature flags detected via CPUID at runtime.
struct CpuFeatures {
    bool sse42       = false;
    bool avx2        = false;
    bool avx512f     = false;
    bool avx512vnni  = false;
    bool avx512bf16  = false;
    bool amx_int8    = false;   // Intel AMX (Sapphire Rapids+)
};

// Detect once and return a cached reference.
const CpuFeatures& cpu_features() noexcept;

} // namespace axonforge::cpu_x86
