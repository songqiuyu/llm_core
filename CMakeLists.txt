cmake_minimum_required(VERSION 3.20)
project(AxonForge
    VERSION 0.1.0
    DESCRIPTION "Modern C++20 edge LLM inference engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Build type default ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Options ----
option(AXONFORGE_BUILD_TESTS    "Build unit tests"          ON)
option(AXONFORGE_BUILD_CLI      "Build CLI inference tool"  ON)
option(AXONFORGE_ENABLE_AVX2    "Enable AVX2 + FMA"         ON)
option(AXONFORGE_ENABLE_AVX512  "Enable AVX-512F + VNNI"    OFF)
option(AXONFORGE_ENABLE_OPENMP  "Enable OpenMP threading"   ON)

# ---- Generate version header ----
configure_file(
    include/axonforge/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/axonforge/version.hpp
    @ONLY
)

# ---- SIMD detection & validation ----
include(cmake/DetectSIMD.cmake)

# ---- Global compile-options interface library ----
add_library(axonforge_compile_options INTERFACE)

target_compile_options(axonforge_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<AND:$<BOOL:${AXONFORGE_ENABLE_AVX2}>,$<CXX_COMPILER_ID:GNU,Clang>>:-mavx2 -mfma>
    $<$<AND:$<BOOL:${AXONFORGE_ENABLE_AVX512}>,$<CXX_COMPILER_ID:GNU,Clang>>:-mavx512f -mavx512vnni -mavx512bw>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)

target_compile_definitions(axonforge_compile_options INTERFACE
    $<$<BOOL:${AXONFORGE_ENABLE_AVX2}>:AXONFORGE_HAS_AVX2>
    $<$<BOOL:${AXONFORGE_ENABLE_AVX512}>:AXONFORGE_HAS_AVX512>
)

# ---- OpenMP ----
if(AXONFORGE_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(axonforge_compile_options INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(axonforge_compile_options INTERFACE AXONFORGE_HAS_OPENMP)
    endif()
endif()

# ---- Subdirectories ----
add_subdirectory(src)

if(AXONFORGE_BUILD_CLI)
    add_subdirectory(tools)
endif()

if(AXONFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
