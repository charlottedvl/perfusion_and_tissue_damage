"""
Multi-compartment Darcy flow model with mixed Dirichlet and Neumann
boundary conditions

System of equations (no summation notation)
Div ( Ki Grad(pi) ) - Sum_j=1^3 beta_ij (pi-pj) = sigma_i

Ki - permeability tensor [mm^3 s / g]
pi & pj - volume averaged pressure in the ith & jth comparments [Pa]
beta_ij - coupling coefficient between the ith & jth compartments [Pa / s]
sigma_i - source term in the ith compartment [1 / s]

@author: Tamas Istvan Jozsa
"""

# %% IMPORT MODULES
# installed python3 modules
from dolfin import *
import time
import sys
import argparse
import numpy as np

np.set_printoptions(linewidth=200)

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

# %% READ INPUT
if rank == 0: print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time() 

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_coupled_flow_solver.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                    type=str, default=None)
parser.add_argument("--mesh_file", help="path to mesh_file",
                    type=str, default=None)
parser.add_argument("--inlet_boundary_file", help="path to inlet_boundary_file",
                    type=str, default=None)
parser.add_argument("--baseline", help="path to perfusion output of baseline scenario",
        type=str, default=None)
parser.add_argument("--occluded", help="path to perfusion output of stroke scenario",
        type=str, default=None)
parser.add_argument("--thresholds", help="number of thresholds to evaluate",
        type=int, default=21)

args = parser.parse_args()
config_file = args.config_file

configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
# physical parameters
p_arterial, p_venous = configs['physical']['p_arterial'], configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], \
    configs['physical']['gmowm_perm_rat']
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
    fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'], model_type=compartmental_model,
                            vel_order=velocity_order)

# get paths to healthy and stroke scenario outputs
healthyfile = getattr(args, 'baseline', configs['output']['res_fldr']+'perfusion.xdmf')
strokefile = getattr(args, 'occluded', configs['output']['res_fldr']+'perfusion_stroke.xdmf')
# this will never work since None also counts as an attribute
# script will therefore only work if the optional arguments are used
# getattr does not overwrite the default
# solution is to set default in parser
healthyfile = configs['output']['res_fldr'] + 'perfusion.xdmf'
strokefile = configs['output']['res_fldr'] + 'perfusion_stroke.xdmf'

if rank == 0: print('Step 2: Reading perfusion files')
# Load previous results
perfusion = Function(K2_space)
f_in = XDMFFile(healthyfile)
f_in.read_checkpoint(perfusion, 'perfusion', 0)
f_in.close()

perfusion_stroke = Function(K2_space)
f_in = XDMFFile(strokefile)
f_in.read_checkpoint(perfusion_stroke, 'perfusion', 0)
f_in.close()

if rank == 0: print('Step 3: Calculating change in perfusion and infarct volume')
# calculate change in perfusion and infarct
perfusion_change = project(((perfusion - perfusion_stroke) / perfusion) * -100, K2_space, solver_type='bicgstab',
                           preconditioner_type='amg')

# thresholds = [-10, -20, -30, -40, -50, -60, -70, -80, -90, -100]
thresholds = np.linspace(0, -100, args.thresholds)

# For now a value of `-70%` is assumed as a desired threshold value to determine
# infarct volume from perfusion data. Thus, we ensure that `-70%` is present
# within the considered threshold values
target = -70
if target not in thresholds:
    # [::-1] to reverse sort direction, maintain descending order
    thresholds = np.sort(np.append(thresholds, target))[::-1]

vol_infarct_values_thresholds = np.empty((0, 4), float)

for threshold in thresholds:
    infarct = project(conditional(gt(perfusion_change, Constant(threshold)), Constant(0.0), Constant(1.0)), K2_space,
                      solver_type='bicgstab', preconditioner_type='amg')
    infarctvolume = suppl_fcts.infarct_vol(mesh, subdomains, infarct)
    vol_infarct_values = np.concatenate((np.array([threshold,threshold,threshold])[:, np.newaxis], infarctvolume), axis=1)
    vol_infarct_values_thresholds = np.append(vol_infarct_values_thresholds, vol_infarct_values, axis=0)

if rank == 0:
    fheader = 'threshold [%],volume ID,Volume [mm^3],infarct volume [mL]'
    np.savetxt(configs['output']['res_fldr'] + 'vol_infarct_values_thresholds.csv', vol_infarct_values_thresholds, "%e,%d,%e,%e", header=fheader)

