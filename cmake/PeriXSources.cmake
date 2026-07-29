# Public PeriX source manifest.
#
# Keep this list explicit: adding a file to src/ must not silently make an
# unpublished implementation part of the public executable.

set(PERIX_CORE_SOURCES
    src/BCSystem/BCSystem.cpp
    src/BCSystem/DirichletBC.cpp
    src/BCSystem/NeumannBC.cpp
    src/BCSystem/PDTractionBC.cpp
    src/BCSystem/SpeciesFluxBC.cpp

    src/ElmtSystem/CahnHilliardElement.cpp
    src/ElmtSystem/DiffusionElement.cpp
    src/ElmtSystem/ElmtSystem.cpp
    src/ElmtSystem/ExplicitPDDOFracElement.cpp
    src/ElmtSystem/FracStressCahnHilliardElement.cpp
    src/ElmtSystem/PDDODynamicFracElement.cpp
    src/ElmtSystem/PoissonElement.cpp

    src/ICSystem/ICBase.cpp
    src/ICSystem/ICSystem.cpp
    src/ICSystem/RandomIC.cpp

    src/InputSystem/InputSchema.cpp
    src/InputSystem/InputSystem.cpp
    src/InputSystem/ReadBCSystemBlock.cpp
    src/InputSystem/ReadDOFsBlock.cpp
    src/InputSystem/ReadElmtSystemBlock.cpp
    src/InputSystem/ReadICSystemBlock.cpp
    src/InputSystem/ReadInputFile.cpp
    src/InputSystem/ReadMeshBlock.cpp
    src/InputSystem/ReadMeshModifyBlock.cpp
    src/InputSystem/ReadPDMeshBlock.cpp
    src/InputSystem/ReadRuntimeBlocks.cpp

    src/JobSystem/JobSystem.cpp

    src/LinearSolver/DefaultSolver.cpp
    src/LinearSolver/LinearSolver.cpp

    src/MathUtils/MathFuns.cpp
    src/MathUtils/MatrixXd.cpp
    src/MathUtils/Rank2Tensor.cpp
    src/MathUtils/Rank4Tensor.cpp
    src/MathUtils/SparseMatrix.cpp
    src/MathUtils/Vector3d.cpp
    src/MathUtils/VectorXd.cpp

    src/Mesh/Circle2DLatticeGenerator.cpp
    src/Mesh/Lagrange2DQuad4MeshGenerator.cpp
    src/Mesh/Lagrange3DHex8MeshGenerator.cpp
    src/Mesh/Mesh.cpp
    src/Mesh/MeshGenerator.cpp
    src/Mesh/MeshImport.cpp
    src/Mesh/Nodes.cpp
    src/Mesh/PDLatticeGeneratorBase.cpp
    src/Mesh/SaveMesh.cpp

    src/MeshModify/MeshModify.cpp
    src/NonlinearSolver/NonlinearSolver.cpp

    src/OutputSystem/ExodusWriter.cpp
    src/OutputSystem/OutputSystem.cpp

    src/PDMesh/CreatePDMesh.cpp
    src/PDMesh/PDMesh.cpp
    src/PDMesh/SavePDMesh.cpp

    src/PDOperators/CalcPDOperators.cpp
    src/PDOperators/PDOperators.cpp

    src/PDProblem/PDProblem.cpp
    src/PDProblem/RunPDProblem.cpp
    src/PDProblem/SaveResults.cpp

    src/PDSystem/FormExplicitDynamicsRate.cpp
    src/PDSystem/FormExplicitRate.cpp
    src/PDSystem/FormResidualAndJacobian.cpp
    src/PDSystem/FormResidualAndJacobianParallel.cpp
    src/PDSystem/PDSystem.cpp

    src/TimeStepping/TimeStepping.cpp

    src/Utils/JsonUtils.cpp
    src/Utils/MessagePrinter.cpp
    src/Utils/ProjectBanner.cpp
    src/Utils/StringUtils.cpp
    src/Utils/Timer.cpp
)

set(PERIX_PARDISO_SOURCE
    src/LinearSolver/PardisoSolver.cpp
)

set(PERIX_CUDSS_SOURCE
    src/LinearSolver/CudssSolver.cpp
)

set(PERIX_CUDA_ASSEMBLY_SOURCES
    src/PDSystem/FormResidualAndJacobianCUDA.cpp
    src/PDSystem/CudaAssembleKernels.cu
)
