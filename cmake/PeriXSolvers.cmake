option(USE_ONEAPI "Build the manuscript Intel oneAPI MKL PARDISO backend" OFF)
set(ONEAPI_DIR "" CACHE PATH "Intel oneAPI root or MKL root (optional; MKLROOT is also honored)")

option(USE_CUDSS "Build the manuscript NVIDIA cuDSS backend" OFF)
set(CUDSS_DIR "" CACHE PATH "NVIDIA cuDSS installation root")

function(perix_configure_solvers target)
    if(USE_ONEAPI)
        find_path(PERIX_MKL_INCLUDE_DIR
            NAMES mkl_pardiso.h
            HINTS "${ONEAPI_DIR}" "$ENV{MKLROOT}"
            PATH_SUFFIXES mkl/latest/include include
        )
        find_library(PERIX_MKL_RT_LIBRARY
            NAMES mkl_rt
            HINTS "${ONEAPI_DIR}" "$ENV{MKLROOT}"
            PATH_SUFFIXES mkl/latest/lib mkl/latest/lib/intel64 lib lib/intel64
        )
        mark_as_advanced(PERIX_MKL_INCLUDE_DIR PERIX_MKL_RT_LIBRARY)
        if(NOT PERIX_MKL_INCLUDE_DIR OR NOT PERIX_MKL_RT_LIBRARY)
            message(FATAL_ERROR
                "USE_ONEAPI requires mkl_pardiso.h and libmkl_rt. "
                "Set ONEAPI_DIR or MKLROOT to the oneAPI/MKL installation."
            )
        endif()

        find_package(Threads REQUIRED)
        target_sources(${target} PRIVATE ${PERIX_PARDISO_SOURCE})
        target_include_directories(${target} PUBLIC "${PERIX_MKL_INCLUDE_DIR}")
        target_compile_definitions(${target} PUBLIC PERIX_HAS_PARDISO)
        set(_perix_mkl_system_libraries Threads::Threads ${CMAKE_DL_LIBS})
        if(UNIX)
            list(APPEND _perix_mkl_system_libraries m)
        endif()
        target_link_libraries(${target} PRIVATE
            "${PERIX_MKL_RT_LIBRARY}"
            ${_perix_mkl_system_libraries}
        )
    endif()

    if(USE_CUDSS)
        perix_require_cuda_toolkit()
        find_path(PERIX_CUDSS_INCLUDE_DIR
            NAMES cudss.h
            HINTS "${CUDSS_DIR}" "$ENV{CUDSS_DIR}"
            PATH_SUFFIXES include
        )
        find_library(PERIX_CUDSS_LIBRARY
            NAMES cudss
            HINTS "${CUDSS_DIR}" "$ENV{CUDSS_DIR}"
            PATH_SUFFIXES lib lib64
        )
        mark_as_advanced(PERIX_CUDSS_INCLUDE_DIR PERIX_CUDSS_LIBRARY)
        if(NOT PERIX_CUDSS_INCLUDE_DIR OR NOT PERIX_CUDSS_LIBRARY)
            message(FATAL_ERROR
                "USE_CUDSS requires cudss.h and libcudss. "
                "Set CUDSS_DIR to the cuDSS installation."
            )
        endif()

        target_sources(${target} PRIVATE ${PERIX_CUDSS_SOURCE})
        target_include_directories(${target} PUBLIC "${PERIX_CUDSS_INCLUDE_DIR}")
        target_compile_definitions(${target} PUBLIC PERIX_HAS_CUDSS)
        target_link_libraries(${target}
            PUBLIC CUDA::cudart
            PRIVATE "${PERIX_CUDSS_LIBRARY}"
        )
    endif()
endfunction()
