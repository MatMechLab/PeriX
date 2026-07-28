[Mesh]
  [./mymesh]
    type = GeneratedMeshGenerator
    dim = 2
    xmax = 1.0
    ymax = 1.0
    nx = 250
    ny = 250
    elem_type = QUAD4
  []
[]

[GlobalParams]
  D = 4.0
  chi = 3.0
  kappa = 0.001
[]

[Variables]
  [./c]
    family = LAGRANGE
    order = FIRST
  [../]
  [./mu]
    family = LAGRANGE
    order = FIRST
  [../]
[]

# give a random inital distribution
[ICs]
  [./c_ic]
    type = GridDataIC
    variable = c
    data_file = ic_mesh.txt
  [../]
[]
# [ICs]
#   [./c_ic]
#     type = RandomIC
#     variable = c
#     min = 0.49
#     max = 0.51
#   [../]
# []

[Kernels]
  [./dcdt]
    type = TimeDerivative
    variable = c
  [../]
  [./c]
    type = CahnHilliardCKernel
    variable = c
    CMax = 1.0
    Mu = mu
  [../]
  [./mu]
    type = CahnHilliardMuKernel
    variable = mu
    C = c
  [../]
[]

[Preconditioning]
  [./smp]
    type = SMP
    full = true
  [../]
[]

[Postprocessors]
  [./c]
    type = ElementAverageValue
    variable = c
  [../]
  [./mu]
    type = ElementAverageValue
    variable = mu
  [../]
[]

[Executioner]
  type = Transient
  solve_type = NEWTON    
  petsc_options_iname = ' -pc_type -ksp_gmres_restart -pc_factor_mat_solver_type'
  petsc_options_value = '  lu       3601               mumps'
[./TimeStepper]
  type = IterationAdaptiveDT
  dt = 0.001
  optimal_iterations = 4
  growth_factor = 1.2
  cutback_factor = 0.5
  [../]
  dtmin = 1e-4
  dtmax = 1
  end_time = 100

  nl_rel_tol = 1e-08
  nl_abs_tol = 1e-09
  nl_max_its = 30

  steady_state_detection = true
  steady_state_tolerance = 1.0e-8
  steady_state_start_time = 310

  auto_preconditioning = false
[]

[Outputs]
  csv = true
  console = true
  print_linear_residuals = false
  print_linear_converged_reason = false
  print_nonlinear_converged_reason = false
  execute_on = 'INITIAL TIMESTEP_END FINAL'
  [./out]
    type = Exodus
    time_step_interval  = 1
    execute_on = 'INITIAL TIMESTEP_END FINAL'
  [../]
[]
