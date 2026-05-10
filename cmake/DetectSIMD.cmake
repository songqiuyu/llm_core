# DetectSIMD.cmake
# Probes the host CPU for SIMD capabilities at configure time.
# Sets AXONFORGE_HOST_HAS_AVX2 and AXONFORGE_HOST_HAS_AVX512 cache variables.

include(CheckCXXSourceRuns)

# ---- AVX2 probe ----
if(AXONFORGE_ENABLE_AVX2)
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_set1_ps(1.0f);
            __m256 b = _mm256_fmadd_ps(a, a, a);
            (void)b;
            return 0;
        }
    " AXONFORGE_HOST_HAS_AVX2)
    unset(CMAKE_REQUIRED_FLAGS)

    if(NOT AXONFORGE_HOST_HAS_AVX2)
        message(WARNING
            "[AxonForge] Host CPU does not support AVX2+FMA. "
            "Disabling AXONFORGE_ENABLE_AVX2.")
        set(AXONFORGE_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
    else()
        message(STATUS "[AxonForge] Host CPU supports AVX2 + FMA.")
    endif()
endif()

# ---- AVX-512F + VNNI probe ----
if(AXONFORGE_ENABLE_AVX512)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512vnni")
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() {
            __m512i a = _mm512_setzero_si512();
            __m512i b = _mm512_dpbusd_epi32(a, a, a);
            (void)b;
            return 0;
        }
    " AXONFORGE_HOST_HAS_AVX512)
    unset(CMAKE_REQUIRED_FLAGS)

    if(NOT AXONFORGE_HOST_HAS_AVX512)
        message(WARNING
            "[AxonForge] Host CPU does not support AVX-512 VNNI. "
            "Disabling AXONFORGE_ENABLE_AVX512.")
        set(AXONFORGE_ENABLE_AVX512 OFF CACHE BOOL "" FORCE)
    else()
        message(STATUS "[AxonForge] Host CPU supports AVX-512F + VNNI.")
    endif()
endif()
