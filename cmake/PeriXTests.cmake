function(perix_register_tests core_target executable_target)
    set(_perix_unit_tests
        BCSystemTests
        CahnHilliardElementTests
        DiffusionElementTests
        ExodusWriterTests
        ExplicitPDDOFracElementTests
        FracStressCahnHilliardElementTests
        ICSystemTests
        InitialVelocityTests
        JobSystemTests
        MeshModifyTests
        OutputSystemTests
        PDDODynamicFracElementTests
        PoissonElementTests
    )

    foreach(_test IN LISTS _perix_unit_tests)
        add_executable(${_test} "tests/${_test}.cpp")
        target_link_libraries(${_test} PRIVATE ${core_target})
        add_test(NAME ${_test} COMMAND ${_test})
        set_tests_properties(${_test} PROPERTIES
            LABELS "unit"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endforeach()

    set(_perix_public_decks
        diffusion.json
        kalthoff_winkler.json
        kalthoff_winkler_gpu.json
        poisson.json
        poisson_gpu.json
        spinodal.json
        tensile_plate.json
    )

    set(_perix_test_example_dir "${CMAKE_CURRENT_BINARY_DIR}/examples")
    file(MAKE_DIRECTORY "${_perix_test_example_dir}")
    foreach(_deck IN LISTS _perix_public_decks)
        configure_file(
            "${PROJECT_SOURCE_DIR}/examples/${_deck}"
            "${_perix_test_example_dir}/${_deck}"
            COPYONLY
        )
    endforeach()
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/impact2D.msh"
        "${_perix_test_example_dir}/impact2D.msh"
        COPYONLY
    )

    add_executable(InputSystemSchemaTests tests/InputSystemSchemaTests.cpp)
    target_link_libraries(InputSystemSchemaTests PRIVATE ${core_target})

    set(_perix_schema_arguments)
    foreach(_deck IN LISTS _perix_public_decks)
        list(APPEND _perix_schema_arguments
            "${_perix_test_example_dir}/${_deck}"
        )
    endforeach()
    add_test(
        NAME InputSystemSchemaTests
        COMMAND InputSystemSchemaTests ${_perix_schema_arguments}
    )
    set_tests_properties(InputSystemSchemaTests PROPERTIES
        LABELS "unit;examples"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    foreach(_deck IN LISTS _perix_public_decks)
        string(REGEX REPLACE "\\.json$" "" _example_name "${_deck}")
        add_test(
            NAME "ReadOnly_${_example_name}"
            COMMAND ${executable_target}
                -i "${_perix_test_example_dir}/${_deck}"
                --read-only
        )
        set_tests_properties("ReadOnly_${_example_name}" PROPERTIES
            LABELS "examples"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endforeach()
endfunction()
