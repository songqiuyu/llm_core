# cpu_x86/CMakeLists.txt
#
# Kernel files that need different SIMD flags are built as OBJECT libraries
# so they don't pollute the global compile flags of axonforge.

# ---- AVX2 kernel object library ----
if(AXONFORGE_ENABLE_AVX2)
    add_library(axonforge_kernels_avx2 OBJECT
        kernels/gemm_avx2.cpp
        kernels/gemv_lut_avx2.cpp
        kernels/gemv_f16_avx2.cpp
        kernels/gemv_q4k_avx2.cpp
        kernels/gemv_q6k_avx2.cpp
        kernels/elementwise_avx2.cpp
        kernels/norm_avx2.cpp
    )
    target_compile_options(axonforge_kernels_avx2 PRIVATE -mavx2 -mfma -mf16c -O3)
    target_include_directories(axonforge_kernels_avx2
        PRIVATE ${PROJECT_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR}/include      # generated version.hpp
                ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_sources(axonforge PRIVATE $<TARGET_OBJECTS:axonforge_kernels_avx2>)
endif()

# ---- AVX-512 VNNI kernel object library ----
if(AXONFORGE_ENABLE_AVX512)
    add_library(axonforge_kernels_avx512 OBJECT
        kernels/gemv_lut_avx512.cpp
        kernels/gemm_int8_vnni.cpp
    )
    target_compile_options(axonforge_kernels_avx512 PRIVATE
        -mavx512f -mavx512vnni -mavx512bw -O3)
    target_include_directories(axonforge_kernels_avx512
        PRIVATE ${PROJECT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_sources(axonforge PRIVATE $<TARGET_OBJECTS:axonforge_kernels_avx512>)
endif()

# ---- Non-SIMD backend glue files ----
target_sources(axonforge PRIVATE
    x86_backend.cpp
    cpu_arena.cpp
    cpuid.cpp
)
target_include_directories(axonforge
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
