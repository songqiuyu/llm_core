add_executable(axonforge-cli main.cpp)
target_link_libraries(axonforge-cli PRIVATE
    -Wl,--whole-archive $<TARGET_FILE:axonforge> -Wl,--no-whole-archive
    axonforge
    axonforge_compile_options)
target_include_directories(axonforge-cli PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
