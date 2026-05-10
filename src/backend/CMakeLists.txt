target_sources(axonforge PRIVATE
    backend_registry.cpp
)

# ---- x86 CPU Backend ----
# Built as a separate OBJECT library so its source files can receive
# distinct SIMD compile flags (-mavx2, -mavx512f, etc.) independently
# from the rest of the axonforge library.
add_subdirectory(cpu_x86)

# ---- Future backend stubs (uncomment when implementing) ----
# add_subdirectory(cpu_arm)
# add_subdirectory(cuda)
# add_subdirectory(vulkan)
# add_subdirectory(metal)
