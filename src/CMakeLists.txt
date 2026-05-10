# src/CMakeLists.txt — builds the single axonforge static library

add_library(axonforge STATIC)

target_include_directories(axonforge
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include      # generated version.hpp
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(axonforge
    PRIVATE axonforge_compile_options
)

target_sources(axonforge PRIVATE
    engine.cpp
)

add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(loader)
add_subdirectory(models)
add_subdirectory(tokenizer)
