target_sources(axonforge PRIVATE
    cuda_backend.cu
    cuda_kernels.cu
)

target_include_directories(axonforge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(axonforge PRIVATE
    CUDA::cudart
    CUDA::cublas
)
if(TARGET CUDA::cublasLt)
    target_link_libraries(axonforge PRIVATE CUDA::cublasLt)
endif()

set_source_files_properties(cuda_backend.cu cuda_kernels.cu
    PROPERTIES
        LANGUAGE CUDA
)
