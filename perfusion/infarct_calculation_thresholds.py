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

import argparse
import time

import numpy as np
import yaml
from dolfin import *

from src.Legacy_version.io import IO_fcts
import finite_element_fcts as fe_mod
import suppl_fcts

np.set_printoptions(linewidth=200)

# ghost mode options: 'none', 'shared_facet', 'shared_vertex'
parameters['ghost_mode'] = 'none'
# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

start0 = time.time()

# %% READ INPUT
if rank == 0:
    print('Step 1: Reading input files, initialising functions and parameters')
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

# extract the healthy file
if args.baseline is None:
    healthyfile = configs['output']['res_fldr'] + 'perfusion.xdmf'
else:
    healthyfile = args.baseline

# extract the occluded scenario
if args.occluded is None:
    strokefile = getattr(args, 'occluded', configs['output']['res_fldr']+'perfusion_stroke.xdmf')
else:
    strokefile = args.occluded

if rank == 0:
    print('Step 2: Reading perfusion files')
# Load previous results
perfusion = Function(K2_space)
f_in = XDMFFile(healthyfile)
f_in.read_checkpoint(perfusion, 'perfusion', 0)
f_in.close()

perfusion_stroke = Function(K2_space)
f_in = XDMFFile(strokefile)
f_in.read_checkpoint(perfusion_stroke, 'perfusion', 0)
f_in.close()

if rank == 0:
    print('Step 3: Calculating change in perfusion and infarct volume')
# calculate change in perfusion and infarct
perfusion_change = project(((perfusion - perfusion_stroke) / perfusion) * -100, K2_space, solver_type='bicgstab',
                           preconditioner_type='petsc_amg')

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
                      solver_type='bicgstab', preconditioner_type='petsc_amg')
    infarctvolume = suppl_fcts.infarct_vol(mesh, subdomains, infarct)
    vol_infarct_values = np.concatenate((np.array([threshold, threshold, threshold])[:, np.newaxis], infarctvolume), axis=1)
    vol_infarct_values_thresholds = np.append(vol_infarct_values_thresholds, vol_infarct_values, axis=0)

if rank == 0:
    fheader = 'threshold [%],volume ID,Volume [mm^3],infarct volume [mL]'
    np.savetxt(configs['output']['res_fldr'] + 'vol_infarct_values_thresholds.csv',
               vol_infarct_values_thresholds, "%e,%d,%e,%e", header=fheader)

    with open(configs['output']['res_fldr'] + "perfusion_outcome.yml", 'a') as outfile:
        # select the row where the perfusion drop is -70%, and the total volume (GM+WM, labelled 23)
        selected_row = np.where((vol_infarct_values_thresholds[:, 0] == -70) &
                                (vol_infarct_values_thresholds[:, 1] == 23))
        volume = vol_infarct_values_thresholds[selected_row, 3]
        yaml.safe_dump(
            {'core-volume_30%_rCBF_mL': float(volume)},
            outfile
        )
