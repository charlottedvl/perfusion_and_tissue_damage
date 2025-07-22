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
from dolfin import *
import time
import sys
import os
import argparse
import numpy

# Local imports
from ..io import IO_fcts
from ..utils import suppl_fcts, config_utils, finite_element_fcts as fe_module

# Global settings
parameters['ghost_mode'] = 'none' # ghost mode options: 'none', 'shared_facet', 'shared_vertex'
set_log_level(50) # solver runs is "silent" mode
numpy.set_printoptions(linewidth=200)

def create_parser_basic_flow_solver():
    """
    Create and return an argument parser for the basic flow solver script.

    This parser handles command-line arguments for configuring and running
    the multi-compartment Darcy flow model used for perfusion computation.

    Returns:
        argparse.ArgumentParser: Configured parser with the following arguments:
            --config_file (str): Path to the YAML configuration file.
                                 Default is './configs/config_basic_flow_solver.yaml'.
            --res_fldr (str): Path to the folder where results will be saved.
                              Should end with a forward slash (/). Optional.
    """
    parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
    parser.add_argument("--config_file", help="path to configuration file",
                        type=str, default='./configs/config_basic_flow_solver.yaml')
    parser.add_argument("--res_fldr", help="path to results folder (string ended with /)", type=str, default=None)
    return parser


def is_non_zero_file(path_file):
    """
    Check if a file exists and has data
    Return 1 if file exists and has data.
    Args:
        path_file (str): path to file
    Returns:
        (boolean): True if the file exists and has data
    """
    return os.path.isfile(path_file) and os.path.getsize(path_file) > 0


def scale_parameters_with_dead_tissue(result_folder, K2, beta12, beta23, K2_space, simulation_configs):
    """
    Scales permeability and coupling coefficients based on infarcted tissue regions.

    This function modifies the permeability tensor `K2` and the coupling coefficients
    `beta12` and `beta23` by applying a feedback-based scaling derived from the
    infarcted (dead) tissue map. The infarcted region is read from an XDMF file
    located in the feedback directory. Each parameter is scaled down in infarcted
    regions according to the `feedback_limit` specified in the simulation config.

    The scaled fields are saved to disk as:
        - K2_scaled.xdmf
        - beta12_scaled.xdmf
        - beta23_scaled.xdmf

    Args:
        result_folder (str): Path to the result folder where the scaled output will be saved.
        K2 (Function): Permeability field for compartment 2 to be modified in place.
        beta12 (Function): Coupling coefficient between compartments 1 and 2 (modified in place).
        beta23 (Function): Coupling coefficient between compartments 2 and 3 (modified in place).
        K2_space (FunctionSpace): Function space for dead tissue and parameters.
        simulation_configs (dict): Dictionary containing simulation parameters.
                                   Must include 'feedback_limit' key.

    Note:
        This function assumes that the infarct map file is named `infarct.xdmf` and located at:
            result_folder + '../feedback/infarct.xdmf'

        Scaling is performed as:
            param *= ((1 - dead_tissue) * (1 - limit) + limit) where `limit` is `feedback_limit` in config.
    """
    tissue_health_file = result_folder + '../feedback/infarct.xdmf'

    if is_non_zero_file(tissue_health_file):
        dead_tissue = Function(K2_space)
        f_in = XDMFFile(tissue_health_file)
        f_in.read_checkpoint(dead_tissue, 'dead', 0)
        f_in.close()

        lower_limit = simulation_configs.get('feedback_limit')
        # TODO: change to k1 for a-model?
        K2.vector()[:] *= ((1 - dead_tissue.vector()) * (1 - lower_limit) + lower_limit)
        beta12.vector()[:] *= ((1 - dead_tissue.vector()) * (1 - lower_limit) + lower_limit)
        beta23.vector()[:] *= ((1 - dead_tissue.vector()) * (1 - lower_limit) + lower_limit)

        with XDMFFile(result_folder + 'K2_scaled.xdmf') as myfile:
            myfile.write_checkpoint(K2, "K2_scaled", 0, XDMFFile.Encoding.HDF5, False)
        with XDMFFile(result_folder + 'beta12_scaled.xdmf') as myfile:
            myfile.write_checkpoint(beta12, "K2_scaled", 0, XDMFFile.Encoding.HDF5, False)
        with XDMFFile(result_folder + 'beta23_scaled.xdmf') as myfile:
            myfile.write_checkpoint(beta23, "K2_scaled", 0, XDMFFile.Encoding.HDF5, False)


def report_execution_time_blood_flow_solver(results_folder, step1_end, step1_start, step2_end, step2_start, step3_end, step3_start, total_end, total_start):
    """
    Logs and prints the execution time for each phase of the blood flow solver.

    This function calculates the duration of each main step in the solver
    (reading input, solving equations, preparing output) and the total time.
    It writes the timing information to a log file and also prints it to stdout.

    Args:
        results_folder (str): Path to the folder where the timing log will be saved.
        step1_end (float): Timestamp at the end of step 1 (input reading).
        step1_start (float): Timestamp at the start of step 1.
        step2_end (float): Timestamp at the end of step 2 (solver).
        step2_start (float): Timestamp at the start of step 2.
        step3_end (float): Timestamp at the end of step 3 (output writing).
        step3_start (float): Timestamp at the start of step 3.
        total_end (float): Timestamp at the end of the entire run.
        total_start (float): Timestamp at the beginning of the entire run.
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
    return


def main():
    # Define MPI variables
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()

    #########################################################################################
    # STEP 1: Read input and initialisation
    #########################################################################################
    # Step initialisation: timer initialisation, initial print
    start0 = time.time()  # start timer for total running time
    if rank == 0:
        print('Step 1: Reading input files, initialising functions and parameters')
    start1 = time.time() # start timer for first step

    # Read input and extract configurations
    parser = create_parser_basic_flow_solver()
    config_file = parser.parse_args().config_file
    configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
    result_folder = configs['output']['res_fldr']

    # Define physical parameters
    physical_configs = configs.get('physical', {})
    p_arterial, p_venous = physical_configs.get('p_arterial'), physical_configs.get('p_venous')
    K1gm_ref, K2gm_ref, K3gm_ref = physical_configs.get('K1gm_ref'), physical_configs.get('K2gm_ref'), \
        physical_configs.get('K3gm_ref')
    gmowm_perm_rat, gmowm_beta_rat = physical_configs.get('gmowm_perm_rat'), physical_configs.get('gmowm_beta_rat')
    beta12gm, beta23gm = physical_configs.get('beta12gm'), physical_configs.get('beta23gm')

    # Read simulation parameters
    simulation_configs = configs.get('simulation', {})
    compartmental_model, velocity_order = config_utils.prepare_simulation_parameters(simulation_configs)

    # Read mesh
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

    # Determine functions spaces
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_module.alloc_fct_spaces(mesh, simulation_configs.get('fe_degr'),
                                model_type=compartmental_model, vel_order=velocity_order)

    # Initialise permeability tensors
    K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space, K2_space, mesh,
                                                   configs['input']['permeability_folder'],
                                                   model_type=compartmental_model)

    # Scaling permeability tensors and coupling coefficients
    if rank == 0:
        print('\t Scaling coupling coefficients and permeability tensors')
    K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3,
                                                 K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat,
                                                 result_folder, model_type=compartmental_model)
    beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, beta12gm, beta23gm, gmowm_beta_rat,
                                                            K2_space, result_folder,
                                                            model_type=compartmental_model)

    end1 = time.time() # End timer for first step

    # If an infarct file exists, scale parameters according to it
    scale_parameters_with_dead_tissue(result_folder, K2, beta12, beta23, K2_space, simulation_configs)

    #########################################################################################
    # STEP 2: Defining and solving governing equations
    #########################################################################################
    # Step initialisation: Starting timer and initial print
    if rank == 0:
        print('Step 2: Defining and solving governing equations')
    start2 = time.time() # Start timer for second step

    # Set up finite element solver
    LHS, RHS, sigma1, sigma2, sigma3, BCs = fe_module.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3,
                                                                     p, p1, p2, p3, K1, K2, K3, beta12, beta23,
                                                                     p_arterial, p_venous,
                                                                     configs['input']['read_inlet_boundary'],
                                                                     configs['input']['inlet_boundary_file'],
                                                                     configs['input']['inlet_BC_type'],
                                                                     model_type=compartmental_model)

    lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'petsc_amg', False, False, False

    # TODO: what are those comments? Can they be suppressed?
    # tested iterative solvers for first order elements: gmres, cg, bicgstab
    # linear_solver_methods()
    # krylov_solver_preconditioners()

    if rank == 0:
        print('\t pressure computation')
    p = fe_module.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol,
                             model_type=compartmental_model)

    end2 = time.time() # End timer for second step

    #########################################################################################
    # STEP 3: Computing velocity fields, saving results, and extracting some field variables
    #########################################################################################
    # Step initialisation: Starting timer and initial print
    if rank == 0:
        print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
    start3 = time.time() # Start timer for third step

    # Compute velocity fields and save the results
    results = {}
    suppl_fcts.compute_my_variables(p, K1, K2, K3, beta12, beta23, p_venous, Vp, Vvel, K2_space,
                                    configs, results, compartmental_model, rank)

    # TODO: determine if they are necessary as they are not used
    # Compute integrated surfaces and volumes
    integrated_variables = {}
    surf_int_values, surf_int_header, volu_int_values, volu_int_header = \
        suppl_fcts.compute_integral_quantities(configs, results, integrated_variables, mesh, subdomains, boundaries, rank)

    end3 = time.time() # Stop timer for step 3
    end0 = time.time() # Stop timer for all process

    #########################################################################################
    # STEP 4: Reporting execution time
    #########################################################################################
    if rank == 0:
        report_execution_time_blood_flow_solver(result_folder, end1, start1, end2, start2, end3, start3, end0, start0)


if __name__ == "__main__":
    main()
