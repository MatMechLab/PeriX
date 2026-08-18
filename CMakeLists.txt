cmake_minimum_required(VERSION 3.21)

project(PeriX VERSION 0.1.2 LANGUAGES CXX)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(PeriXSources)
include(PeriXCuda)
include(PeriXSolvers)

option(PARALLEL_ASSEMBLE "Build the manuscript OpenMP assembly backend" ON)

find_package(Eigen3 5.0.1 CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 CONFIG REQUIRED)

add_library(perix_core STATIC ${PERIX_CORE_SOURCES})
add_library(PeriX::core ALIAS perix_core)

target_compile_features(perix_core PUBLIC cxx_std_20)
set_target_properties(perix_core PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(perix_core PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_link_libraries(perix_core PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
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

install(TARGETS perix
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY examples/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/perix/examples"
)
install(FILES documents/PeriXManual.pdf
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
)
install(FILES
    LICENSE
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/licenses"
)

include(CTest)
if(BUILD_TESTING)
    include(PeriXTests)
    perix_register_tests(PeriX::core perix)
endif()
