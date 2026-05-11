#include "cpuid.hpp"
#include <cpuid.h>   // GCC/Clang built-in

namespace axonforge::cpu_x86 {

CpuFeatures detect_cpu_features() noexcept {
    CpuFeatures f{};
    unsigned int eax, ebx, ecx, edx;

    // CPUID leaf 1: basic feature flags
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        f.sse42 = (ecx >> 20) & 1;
    }

    // CPUID leaf 7, sub-leaf 0: extended feature flags
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        f.avx2      = (ebx >>  5) & 1;
        f.avx512f   = (ebx >> 16) & 1;
        f.avx512vnni = (ecx >> 11) & 1;
        f.avx512bf16 = (edx >>  5) & 1;
        f.amx_int8  = (edx >> 25) & 1;
    }

    // CPUID leaf 7, sub-leaf 1: AVX-VNNI (VEX-encoded, Alder Lake+)
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        f.avxvnni = (eax >> 4) & 1;
    }

    return f;
}

// Global singleton — detected once at startup.
const CpuFeatures& cpu_features() noexcept {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}

} // namespace axonforge::cpu_x86
