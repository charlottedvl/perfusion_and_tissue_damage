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
import argparse

from ..io import IO_fcts
from ..utils import suppl_fcts

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

#%% READ INPUT
if rank == 0: print('Step 1: Reading input files')

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file",
                    type=str, default='./configs/config_permeability_initialiser.yaml')
config_file = parser.parse_args().config_file

configs = IO_fcts.perm_init_config_reader_yml(config_file)


# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])


#%% COMPUTE PERMEABILITIES
if rank == 0: print('Step 2: Computing permeability tensor')

K_space = TensorFunctionSpace(mesh, "DG", 0)

e_loc, main_direction = suppl_fcts.comp_vessel_orientation(subdomains,boundaries,mesh,configs['output']['res_fldr'],configs['output']['save_subres'])

start1 = time.time()
# compute permeability tensor
K1 = suppl_fcts.perm_tens_comp(K_space,subdomains,mesh,configs['physical']['e_ref'],e_loc,configs['physical']['K1_form'])
end1 = time.time()
if rank == 0: print ("\t permeability tensor computation on processor 0 took ", '{:.2f}'.format(end1 - start1), '[s]\n')

#%% SAVE OUTPUT
"""TODO: compress output and add postprocessing option!!!"""
if rank == 0: print('Step 3: Saving output files')

with XDMFFile(configs['output']['res_fldr']+'K1_form.xdmf') as myfile:
    myfile.write_checkpoint(K1,"K1_form", 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs['output']['res_fldr']+'e_loc.xdmf') as myfile:
    myfile.write_checkpoint(e_loc,"e_loc", 0, XDMFFile.Encoding.HDF5, False)
# main_direction is non-essential output
with XDMFFile(configs['output']['res_fldr']+'main_direction.xdmf') as myfile:
    myfile.write(main_direction)

myResults={}
out_vars = configs['output']['res_vars']
if len(out_vars)>0:
    myResults['K1_form'] = K1
    myResults['e_loc'] = e_loc
    myResults['main_direction'] = main_direction
else:
    if rank==0: print('No variables have been defined for saving!')

# save variables
res_keys = set(myResults.keys())
for myvar in out_vars:
    if myvar in res_keys:
        with XDMFFile(configs['output']['res_fldr']+myvar+'.xdmf') as myfile:
            myfile.write_checkpoint(myResults[myvar], myvar, 0, XDMFFile.Encoding.HDF5, False)
    else:
        if rank==0: print('warning: '+myvar+' variable cannot be saved - variable undefined!')