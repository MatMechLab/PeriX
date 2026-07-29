option(CUDA_ASSEMBLE "Build the manuscript CUDA assembly backend" OFF)
set(CUDA_DIR "" CACHE PATH "CUDA toolkit root (optional; CUDAToolkit_ROOT is also honored)")

function(perix_require_cuda_toolkit)
    if(TARGET CUDA::cudart)
        return()
    endif()

    if(CUDA_DIR AND NOT CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "${CUDA_DIR}")
    endif()
    find_package(CUDAToolkit REQUIRED)
endfunction()

function(perix_configure_cuda_assembly target)
    if(NOT CUDA_ASSEMBLE)
        return()
    endif()

    perix_require_cuda_toolkit()
    enable_language(CUDA)

    target_sources(${target} PRIVATE ${PERIX_CUDA_ASSEMBLY_SOURCES})
    target_compile_definitions(${target} PUBLIC PERIX_CUDA_ASSEMBLE)
    target_link_libraries(${target} PRIVATE CUDA::cudart)

    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
endfunction()
