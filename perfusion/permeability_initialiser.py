"""
This script generates an anisotropic permeability field based on a predefined
form of the permeability tensor given in a reference coordinate system.

K1_form is the forms of the permeability tensor defined in a rerence
coordinate system, in which e_ref [0,0,1] unit vector is the coordinate
direction perpendicular to the cortical surface.

Then K1_loc is computed as
K1_loc = T*K1_form*T'

T is the transformation matrix handling rotation based on e_ref and e_loc,
where e_loc is the unit vector showing the direction normal to the cortical
surface locally.

e_loc = - grad(pe)/|grad(pe)| where Laplacian(pe) = 0 with the following BCs:
pe = 1 @ cortical surface
pe = 0 @ ventricular surface
d pe / d n = 0 @ brain stem cut plane

@author: Tamas Istvan Jozsa
"""

#%% IMPORT MODULES
from dolfin import *
import time

import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

#%% READ INPUT
if rank == 0: print('Step 1: Reading input files')

config_file = 'config_permeability_initialiser.xml'
mesh_file, e_ref, K1_form, res_fldr, save_subres = \
    IO_fcts.perm_init_config_reader(config_file)

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(mesh_file)


#%% COMPUTE PERMEABILITIES
if rank == 0: print('Step 2: Computing permeability tensor')

K_space = TensorFunctionSpace(mesh, "DG", 0)

e_loc, main_direction = suppl_fcts.comp_vessel_orientation(subdomains,boundaries,mesh,res_fldr,save_subres)

start1 = time.time()
# compute permeability tensor
K1 = suppl_fcts.perm_tens_comp(K_space,subdomains,mesh,e_ref,e_loc,K1_form)
end1 = time.time()
if rank == 0: print ("\t permeability tensor computation on processor 0 took ", '{:.2f}'.format(end1 - start1), '[s]\n')

#%% SAVE OUTPUT
"""TODO: compress output and add postprocessing option!!!"""
if rank == 0: print('Step 3: Saving output files')

with XDMFFile(res_fldr+'K1_form.xdmf') as myfile:
    myfile.write_checkpoint(K1,"K1_form", 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(res_fldr+'e_loc.xdmf') as myfile:
    myfile.write_checkpoint(e_loc,"e_loc", 0, XDMFFile.Encoding.HDF5, False)
# main_direction is non-essential output
with XDMFFile(res_fldr+'main_direction.xdmf') as myfile:
    myfile.write(main_direction)