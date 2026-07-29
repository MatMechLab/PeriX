option(CUDA_ASSEMBLE "Build the manuscript CUDA assembly backend" OFF)
set(CUDA_DIR "" CACHE PATH "CUDA toolkit root (optional; CUDAToolkit_ROOT is also honored)")

macro(perix_require_cuda_toolkit)
    if(NOT TARGET CUDA::cudart)
        if(CUDA_DIR AND NOT CUDAToolkit_ROOT)
            set(CUDAToolkit_ROOT "${CUDA_DIR}")
        endif()
        find_package(CUDAToolkit REQUIRED)
    endif()
endmacro()

macro(perix_configure_cuda_assembly target)
    if(CUDA_ASSEMBLE)
        perix_require_cuda_toolkit()

        if(CUDA_DIR AND NOT CMAKE_CUDA_COMPILER
           AND EXISTS "${CUDA_DIR}/bin/nvcc")
            file(REAL_PATH "${CUDA_DIR}/bin/nvcc" _perix_nvcc)
            set(CMAKE_CUDA_COMPILER "${_perix_nvcc}" CACHE FILEPATH
                "CUDA compiler selected from CUDA_DIR")

            # Conda CUDA packages prepend their own bin directory while nvcc
            # runs the host compiler.  When CMake itself uses the system C++
            # compiler, an unrelated Conda linker can otherwise be selected.
            get_filename_component(_perix_cuda_bin_dir "${_perix_nvcc}" DIRECTORY)
            get_filename_component(_perix_cuda_prefix "${_perix_cuda_bin_dir}" DIRECTORY)
            if(UNIX AND EXISTS "${_perix_cuda_prefix}/conda-meta"
               AND CMAKE_CXX_COMPILER)
                file(REAL_PATH "${CMAKE_CXX_COMPILER}" _perix_host_cxx)
                get_filename_component(_perix_host_bin_dir
                    "${_perix_host_cxx}" DIRECTORY)
                if(NOT CMAKE_CUDA_HOST_COMPILER)
                    set(CMAKE_CUDA_HOST_COMPILER "${_perix_host_cxx}"
                        CACHE FILEPATH "Host compiler selected for Conda CUDA")
                endif()
                string(APPEND CMAKE_CUDA_FLAGS_INIT
                    " -Xcompiler=-B${_perix_host_bin_dir}")
            endif()
        endif()
        enable_language(CUDA)

        target_sources(${target} PRIVATE ${PERIX_CUDA_ASSEMBLY_SOURCES})
        target_compile_definitions(${target} PUBLIC PERIX_CUDA_ASSEMBLE)
        target_link_libraries(${target} PRIVATE CUDA::cudart)

        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
        )
    endif()
endmacro()
