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

# %% READ INPUT
if rank == 0: print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time()

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--healthy_config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_healthy00.yml')
parser.add_argument("--occluded_config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_RMCA_occl00.yml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                    type=str, default=None)
healthy_config_file  = parser.parse_args().healthy_config_file
occluded_config_file = parser.parse_args().occluded_config_file

healthy_configs  = IO_fcts.basic_flow_config_reader2(healthy_config_file,  parser)
occluded_configs = IO_fcts.basic_flow_config_reader2(occluded_config_file, parser)

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(healthy_configs.input.mesh_file)

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, healthy_configs.simulation.fe_degr)

healthyfile = healthy_configs.output.res_fldr + 'perfusion.xdmf'
strokefile = occluded_configs.output.res_fldr + 'perfusion.xdmf'

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
                           preconditioner_type='petsc_amg')
infarct = project(conditional(gt(perfusion_change, Constant(-70)), Constant(0.0), Constant(1.0)), K2_space,
                  solver_type='bicgstab', preconditioner_type='petsc_amg')

with XDMFFile(occluded_configs.output.res_fldr + 'perfusion_change.xdmf') as myfile:
    myfile.write_checkpoint(perfusion_change, 'perfusion_change', 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(occluded_configs.output.res_fldr + 'infarct.xdmf') as myfile:
    myfile.write_checkpoint(infarct, 'infarct', 0, XDMFFile.Encoding.HDF5, False)


if occluded_configs.output.comp_ave == True:
    vol_infarct_values = suppl_fcts.infarct_vol(mesh, subdomains, infarct)

    fheader = 'volume ID,Volume [mm^3],infarct volume [mL]'
    numpy.savetxt(occluded_configs.output.res_fldr + 'vol_infarct_values.csv', vol_infarct_values, "%d,%e,%e",
                      header=fheader)
