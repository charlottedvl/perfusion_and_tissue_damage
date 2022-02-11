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

# IMPORT MODULES
# installed python3 modules
from dolfin import *
import time
import sys
import os
import argparse
import numpy
numpy.set_printoptions(linewidth=200)

# ghost mode options: 'none', 'shared_facet', 'shared_vertex'
parameters['ghost_mode'] = 'none'

# added module
import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

start0 = time.time()

# READ INPUT
if rank == 0:
    print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time()

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file",
                    type=str, default='./config_basic_flow_solver.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)", type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
# physical parameters
p_arterial, p_venous = configs['physical']['p_arterial'], configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], \
    configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat']
beta12gm, beta23gm, gmowm_beta_rat = \
    configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']

try:
    compartmental_model = configs['simulation']['model_type'].lower().strip()
except KeyError:
    compartmental_model = 'acv'

try:
    velocity_order = configs['simulation']['vel_order']
except KeyError:
    velocity_order = configs['simulation']['fe_degr'] - 1

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'],
                            model_type=compartmental_model, vel_order=velocity_order)

# initialise permeability tensors
K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space, K2_space, mesh,
                                               configs['input']['permeability_folder'], model_type=compartmental_model)

if rank == 0:
    print('\t Scaling coupling coefficients and permeability tensors')

# set coupling coefficients
beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, beta12gm, beta23gm, gmowm_beta_rat,
                                                        K2_space, configs['output']['res_fldr'],
                                                        model_type=compartmental_model)

K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3,
                                             K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat,
                                             configs['output']['res_fldr'], model_type=compartmental_model)
end1 = time.time()


tissue_health_file = configs['output']['res_fldr'] + '../feedback/infarct.xdmf'


def is_non_zero_file(fpath):
    """
    Return 1 if file exists and has data.
    :param fpath: path to file
    :return: boolean
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


if is_non_zero_file(tissue_health_file):
    dead_tissue = Function(K2_space)
    f_in = XDMFFile(tissue_health_file)
    f_in.read_checkpoint(dead_tissue, 'dead', 0)
    f_in.close()

    lower_limit = configs['simulation']['feedback_limit']
    # todo change to k1 for a-model?
    K2.vector()[:] *= ((1-dead_tissue.vector())*(1-lower_limit)+lower_limit)
    beta12.vector()[:] *= ((1-dead_tissue.vector())*(1-lower_limit)+lower_limit)
    beta23.vector()[:] *= ((1-dead_tissue.vector())*(1-lower_limit)+lower_limit)

    with XDMFFile(configs['output']['res_fldr'] + 'K2_scaled.xdmf') as myfile:
        myfile.write_checkpoint(K2, "K2_scaled", 0, XDMFFile.Encoding.HDF5, False)
    with XDMFFile(configs['output']['res_fldr'] + 'beta12_scaled.xdmf') as myfile:
        myfile.write_checkpoint(beta12, "K2_scaled", 0, XDMFFile.Encoding.HDF5, False)
    with XDMFFile(configs['output']['res_fldr'] + 'beta23_scaled.xdmf') as myfile:
        myfile.write_checkpoint(beta23, "K2_scaled", 0, XDMFFile.Encoding.HDF5, False)


# SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
if rank == 0:
    print('Step 2: Defining and solving governing equations')
start2 = time.time()

# set up finite element solver
LHS, RHS, sigma1, sigma2, sigma3, BCs = fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3,
                                                                 p, p1, p2, p3, K1, K2, K3, beta12, beta23,
                                                                 p_arterial, p_venous,
                                                                 configs['input']['read_inlet_boundary'],
                                                                 configs['input']['inlet_boundary_file'],
                                                                 configs['input']['inlet_BC_type'],
                                                                 model_type=compartmental_model)

lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

# tested iterative solvers for first order elements: gmres, cg, bicgstab
# linear_solver_methods()
# krylov_solver_preconditioners()
if rank == 0:
    print('\t pressure computation')
p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol,
                         model_type=compartmental_model)
end2 = time.time()

#  COMPUTE VELOCITY FIELDS, SAVE SOLUTION, EXTRACT FIELD VARIABLES
if rank == 0:
    print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
start3 = time.time()

myResults = {}
suppl_fcts.compute_my_variables(p, K1, K2, K3, beta12, beta23, p_venous, Vp, Vvel, K2_space,
                                configs, myResults, compartmental_model, rank)

my_integr_vars = {}
surf_int_values, surf_int_header, volu_int_values, volu_int_header = \
    suppl_fcts.compute_integral_quantities(configs, myResults, my_integr_vars, mesh, subdomains, boundaries, rank)

end3 = time.time()
end0 = time.time()

# REPORT EXECUTION TIME
if rank == 0:
    oldstdout = sys.stdout
    logfile = open(configs['output']['res_fldr']+"time_info.log", 'w')
    sys.stdout = logfile
    print('Total execution time [s]; \t\t\t', end0 - start0)
    print('Step 1: Reading input files [s]; \t\t', end1 - start1)
    print('Step 2: Solving governing equations [s]; \t\t', end2 - start2)
    print('Step 3: Preparing and saving output [s]; \t\t', end3 - start3)
    logfile.close()
    sys.stdout = oldstdout
    print('Execution time: \t', end0 - start0, '[s]')
    print('Step 1: \t\t', end1 - start1, '[s]')
    print('Step 2: \t\t', end2 - start2, '[s]')
    print('Step 3: \t\t', end3 - start3, '[s]')
