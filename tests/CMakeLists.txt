include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ---- Helper to add test executables ----
function(axonforge_add_test name)
    add_executable(${name} ${name}.cpp)
    # --whole-archive forces all object files from libaxonforge.a to be
    # included, ensuring [[gnu::constructor]] registration functions run.
    target_link_libraries(${name} PRIVATE
        -Wl,--whole-archive $<TARGET_FILE:axonforge> -Wl,--no-whole-archive
        axonforge
        Catch2::Catch2WithMain)
    target_include_directories(${name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include)
    catch_discover_tests(${name})
endfunction()

axonforge_add_test(test_tensor)
axonforge_add_test(test_graph)
axonforge_add_test(test_backend_x86)
axonforge_add_test(test_gguf_reader)
