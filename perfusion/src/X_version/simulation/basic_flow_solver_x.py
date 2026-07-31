# TODO: clear code after testing
"""
Multi-compartment Darcy flow model with mixed Dirichlet and Neumann boundary conditions

System of equations (no summation notation)
Div ( Ki Grad(pi) ) - Sum_j=1^3 beta_ij (pi-pj) = sigma_i

Ki - permeability tensor [mm^3 s g^-1]
pi & pj - Darcy pressures in the ith & jth comparments [Pa]
beta_ij - coupling coefficient between the ith & jth compartments [Pa^-1 s^-1]
sigma_i - source term in the ith compartment [s^-1]

@author: Tamas Istvan Jozsa
"""

# Python imports
import argparse
import os
import sys
import time

# FEniCS-X imports
import dolfinx
from dolfinx import log
import numpy as np
from mpi4py import MPI

# Local imports
from src.X_version.io import IO_functions, basic_flow_solver_IO, permeability_initialiser_IO as perm_IO
from src.X_version.utils import suppl_fcts, config_utils
from src.X_version.utils import finite_element_functions as fe_module

# Global settings
np.set_printoptions(linewidth=200)
log.set_log_level(log.LogLevel.WARNING)  # Equivalent to set_log_level(50) in Legacy FEniCS


def create_parser_basic_flow_solver():
    """
    Create and return an argument parser for the basic flow solver script.

    Returns:
        argparse.ArgumentParser: Configured parser with the following arguments:
            --config_file (str): Path to the YAML configuration file.
                                 Default is './configs/config_basic_flow_solver.yaml'.
            --res_fldr (str): Path to the folder where results will be saved.
                              Should end with a forward slash (/). Optional.
    """
    parser = argparse.ArgumentParser(
        description="perfusion computation based on multi-compartment Darcy flow model"
    )
    parser.add_argument(
        "--config_file",
        help="path to configuration file",
        type=str,
        default='./configs/config_basic_flow_solver.yaml'
    )
    parser.add_argument(
        "--res_fldr",
        help="path to results folder (string ended with /)",
        type=str,
        default=None
    )
    return parser


def is_non_zero_file(path_file):
    """
    Check if a file exists and has data.

    Args:
        path_file (str): path to file

    Returns:
        bool: True if the file exists and has data
    """
    return os.path.isfile(path_file) and os.path.getsize(path_file) > 0


def scale_parameters_with_dead_tissue(comm, result_folder, K2, beta12, beta23, K2_space, simulation_configs):
    """
    Scales permeability and coupling coefficients based on infarcted tissue regions.

    Reads the infarct map from an XDMF file and scales K2, beta12, beta23 down
    in infarcted regions according to feedback_limit in simulation_configs.

    The scaled fields are saved to disk as K2_scaled.xdmf, beta12_scaled.xdmf,
    beta23_scaled.xdmf.

    Args:
        comm (MPI.Comm): MPI communicator.
        result_folder (str): Path to result folder.
        K2 (dolfinx.fem.Function): Permeability field for compartment 2 (modified in place).
        beta12 (dolfinx.fem.Function): Coupling coefficient between compartments 1 and 2 (modified in place).
        beta23 (dolfinx.fem.Function): Coupling coefficient between compartments 2 and 3 (modified in place).
        K2_space (dolfinx.fem.FunctionSpace): Function space for dead tissue and parameters.
        simulation_configs (dict): Simulation parameters, must include 'feedback_limit'.

    Notes:
        Scaling formula: param *= ((1 - dead_tissue) * (1 - limit) + limit)
    """
    tissue_health_file = result_folder + '../feedback/infarct.xdmf'

    if is_non_zero_file(tissue_health_file):
        dead_tissue = dolfinx.fem.Function(K2_space)

        # FEniCS-X: read_checkpoint replaced by read_mesh + read_function
        with dolfinx.io.XDMFFile(comm, tissue_health_file, "r") as f_in:
            f_in.read_function(dead_tissue, 'dead')

        lower_limit = simulation_configs.get('feedback_limit')

        # Scale arrays in place
        K2.x.array[:] *= ((1 - dead_tissue.x.array) * (1 - lower_limit) + lower_limit)
        beta12.x.array[:] *= ((1 - dead_tissue.x.array) * (1 - lower_limit) + lower_limit)
        beta23.x.array[:] *= ((1 - dead_tissue.x.array) * (1 - lower_limit) + lower_limit)

        # Propagate ghost values after in-place modification
        K2.x.scatter_forward()
        beta12.x.scatter_forward()
        beta23.x.scatter_forward()

        # Save scaled fields — FEniCS-X: write_mesh + write_function instead of write_checkpoint
        mesh = K2_space.mesh
        with dolfinx.io.XDMFFile(comm, result_folder + 'K2_scaled.xdmf', "w") as myfile:
            myfile.write_mesh(mesh)
            myfile.write_function(K2)
        with dolfinx.io.XDMFFile(comm, result_folder + 'beta12_scaled.xdmf', "w") as myfile:
            myfile.write_mesh(mesh)
            myfile.write_function(beta12)
        with dolfinx.io.XDMFFile(comm, result_folder + 'beta23_scaled.xdmf', "w") as myfile:
            myfile.write_mesh(mesh)
            myfile.write_function(beta23)


def report_execution_time_blood_flow_solver(
        results_folder, step1_end, step1_start,
        step2_end, step2_start, step3_end, step3_start,
        total_end, total_start):
    """
    Logs and prints the execution time for each phase of the blood flow solver.

    Args:
        results_folder (str): Path to folder where timing log will be saved.
        step1_end, step1_start (float): Timestamps for step 1 (input reading).
        step2_end, step2_start (float): Timestamps for step 2 (solver).
        step3_end, step3_start (float): Timestamps for step 3 (output writing).
        total_end, total_start (float): Timestamps for the full run.
    """
    total_time = total_end - total_start
    time_step1 = step1_end - step1_start
    time_step2 = step2_end - step2_start
    time_step3 = step3_end - step3_start

    oldstdout = sys.stdout
    logfile = open(results_folder + "time_info.log", 'w')
    sys.stdout = logfile
    print('Total execution time [s]; \t\t\t', total_time)
    print('Step 1: Reading input files [s]; \t\t', time_step1)
    print('Step 2: Solving governing equations [s]; \t\t', time_step2)
    print('Step 3: Preparing and saving output [s]; \t\t', time_step3)
    logfile.close()
    sys.stdout = oldstdout

    print('Execution time: \t', total_time, '[s]')
    print('Step 1: \t\t', time_step1, '[s]')
    print('Step 2: \t\t', time_step2, '[s]')
    print('Step 3: \t\t', time_step3, '[s]')


def main():
    # Define MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ####################################################################################
    # STEP 1: Read input and initialisation
    ####################################################################################
    start0 = time.time()
    if rank == 0:
        print('Step 1: Reading input files, initialising functions and parameters')
    start1 = time.time()

    # Read input and extract configurations
    parser = create_parser_basic_flow_solver()
    config_file = parser.parse_args().config_file
    configs = basic_flow_solver_IO.basic_flow_config_reader_yaml(config_file, parser)
    result_folder = configs['output']['res_fldr']

    # Define physical parameters
    physical_configs = configs.get('physical', {})
    p_arterial = physical_configs.get('p_arterial')
    p_venous = physical_configs.get('p_venous')
    K1gm_ref = physical_configs.get('K1gm_ref')
    K2gm_ref = physical_configs.get('K2gm_ref')
    K3gm_ref = physical_configs.get('K3gm_ref')
    gmowm_perm_rat = physical_configs.get('gmowm_perm_rat')
    gmowm_beta_rat = physical_configs.get('gmowm_beta_rat')
    beta12gm = physical_configs.get('beta12gm')
    beta23gm = physical_configs.get('beta23gm')

    # Read simulation parameters
    simulation_configs = configs.get('simulation', {})
    compartmental_model, velocity_order = config_utils.prepare_simulation_parameters(simulation_configs)

    # Read mesh — FEniCS-X: comm is now explicitly passed
    mesh, subdomains, boundaries = IO_functions.mesh_reader(comm, configs['input']['mesh_file'])
    
    # Determine function spaces
    # NOTE: function renamed alloc_fct_spaces in X_version (was allocation_functions_space in Legacy)
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_module.allocation_functions_space(
            mesh,
            simulation_configs.get('fe_degr'),
            model_type=compartmental_model,
            vel_order=velocity_order
        )
    
    #TODO: debug perm tens ini, compare with Legacy version maybe errors introduced by Claude
    # Initialise permeability tensors
    K1, K2, K3 = perm_IO.initialise_permeabilities(
        K1_space, K2_space,
        configs['input']['permeability_folder'],
        model_type=compartmental_model
    )
    
    # Scale permeability tensors and coupling coefficients
    if rank == 0:
        print('\t Scaling coupling coefficients and permeability tensors')
    K1, K2, K3 = suppl_fcts.scale_permeabilities(
        subdomains, K1, K2, K3,
        K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat,
        result_folder,
        model_type=compartmental_model
    )
    beta12, beta23 = suppl_fcts.scale_coupling_coefficients(
        subdomains, beta12gm, beta23gm, gmowm_beta_rat,
        K2_space, result_folder,
        model_type=compartmental_model
    )

    end1 = time.time()
    
    # If an infarct file exists, scale parameters according to it (tissue feedback)
    scale_parameters_with_dead_tissue(
        comm, result_folder, K2, beta12, beta23, K2_space, simulation_configs
    )
    
    ####################################################################################
    # STEP 2: Defining and solving governing equations
    ####################################################################################
    if rank == 0:
        print('Step 2: Defining and solving governing equations')
    start2 = time.time()

    # Set up finite element variational problem
    # NOTE: function renamed set_up_fe_solver2 in X_version
    LHS, RHS, sigma1, sigma2, sigma3, BCs = fe_module.set_up_fe_solver(
        mesh, boundaries,
        Vp, v_1, v_2, v_3,
        p, p1, p2, p3,
        K1, K2, K3, beta12, beta23,
        p_arterial, p_venous,
        configs['input']['read_inlet_boundary'],
        configs['input']['inlet_boundary_file'],
        configs['input']['inlet_BC_type'],
        model_type=compartmental_model
    )
    
    # Solver settings — bicgstab + petsc_amg matches the Legacy configuration
    # FEniCS-X: petsc_amg corresponds to 'hypre' in PETSc terminology
    lin_solver = 'bcgs'       # PETSc name for bicgstab
    precond = 'hypre'         # PETSc name for petsc_amg
    rtol = False
    mon_conv = False
    init_sol = False
    
    if rank == 0:
        print('\t pressure computation')
    start_solve = time.time()

    p_sol = fe_module.solve_lin_sys(
        Vp, LHS, RHS, BCs,
        lin_solver, precond, rtol, mon_conv, init_sol,
        model_type=compartmental_model
    )

    end_solve = time.time()
    if rank == 0:
        print('\t\t pressure computation took', end_solve - start_solve, '[s]')

    end2 = time.time()
    #CHECKPOINT
    ####################################################################################
    # STEP 3: Computing velocity fields, saving results, extracting field variables
    ####################################################################################
    if rank == 0:
        print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
    start3 = time.time()

    # Create output directory if it does not exist (rank 0 only to avoid race conditions)
    if rank == 0:
        os.makedirs(result_folder, exist_ok=True)

    results = {}
    suppl_fcts.compute_my_variables(
        p_sol, K1, K2, K3, beta12, beta23, p_venous,
        Vp, Vvel, K2_space,
        configs, results, compartmental_model, rank, comm
    )

    # Compute surface and volume integrals
    integrated_variables = {}
    surf_int_values, surf_int_header, volu_int_values, volu_int_header = \
        suppl_fcts.compute_integral_quantities(
            configs, results, integrated_variables,
            mesh, subdomains, boundaries, rank
        )

    end3 = time.time()
    end0 = time.time()
   
    
    ####################################################################################
    # STEP 4: Reporting execution time
    ####################################################################################
    if rank == 0:
        report_execution_time_blood_flow_solver(
            result_folder,
            end1, start1,
            end2, start2,
            end3, start3,
            end0, start0
        )
    print("Finito pipo")

if __name__ == "__main__":
    main()
