cmake_minimum_required(VERSION 3.21)

project(PeriX VERSION 0.10 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(PeriXSources)
include(PeriXCuda)
include(PeriXSolvers)

option(PARALLEL_ASSEMBLE "Build the manuscript OpenMP assembly backend" ON)

add_library(perix_core STATIC ${PERIX_CORE_SOURCES})
add_library(PeriX::core ALIAS perix_core)

target_compile_features(perix_core PUBLIC cxx_std_20)
set_target_properties(perix_core PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(perix_core PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/eigen>"
)

if(PARALLEL_ASSEMBLE)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_compile_definitions(perix_core PUBLIC PERIX_PARALLEL_ASSEMBLE)
    target_link_libraries(perix_core PUBLIC OpenMP::OpenMP_CXX)
endif()

perix_configure_cuda_assembly(perix_core)
perix_configure_solvers(perix_core)

add_executable(perix src/main.cpp)
target_link_libraries(perix PRIVATE PeriX::core)

include(CTest)
if(BUILD_TESTING)
    include(PeriXTests)
    perix_register_tests(PeriX::core perix)
endif()
