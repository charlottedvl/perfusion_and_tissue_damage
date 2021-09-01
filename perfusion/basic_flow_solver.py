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

#%% IMPORT MODULES
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


#%% READ INPUT
if rank == 0: print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time()

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file",
                    type=str, default='./config_basic_flow_solver.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader_yml(config_file,parser)
# physical parameters
p_arterial, p_venous = configs['physical']['p_arterial'], configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat']
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
    fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'], \
                            model_type = compartmental_model, vel_order = velocity_order)

# initialise permeability tensors
K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space,K2_space,mesh,configs['input']['permeability_folder'], model_type = compartmental_model)


if rank == 0: print('\t Scaling coupling coefficients and permeability tensors')

# set coupling coefficients
beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
                                beta12gm, beta23gm, gmowm_beta_rat, \
                                K2_space, configs['output']['res_fldr'], configs['output']['save_pvd'], model_type = compartmental_model)

K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3, \
                                  K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
                                  configs['output']['res_fldr'],configs['output']['save_pvd'], model_type = compartmental_model)
end1 = time.time()


#%% SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
if rank == 0: print('Step 2: Defining and solving governing equations')
start2 = time.time()

# set up finite element solver
LHS, RHS, sigma1, sigma2, sigma3, BCs = \
fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, \
                         p, p1, p2, p3, K1, K2, K3, beta12, beta23, \
                         p_arterial, p_venous, \
                         configs['input']['read_inlet_boundary'], configs['input']['inlet_boundary_file'], \
                         configs['input']['inlet_BC_type'], model_type = compartmental_model)

lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

# tested iterative solvers for first order elements: gmres, cg, bicgstab
#linear_solver_methods()
#krylov_solver_preconditioners()
if rank == 0: print('\t pressure computation')
p = fe_mod.solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol,model_type = compartmental_model)
end2 = time.time()


#%% COMPUTE VELOCITY FIELDS, SAVE SOLUTION, EXTRACT FIELD VARIABLES
if rank == 0: print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
start3 = time.time()

myResults={}
out_vars = configs['output']['res_vars']

if len(out_vars)>0:
    if compartmental_model == 'acv':
        p1, p2, p3 = p.split()
        myResults['press1'], myResults['press2'], myResults['press3'] = p1, p2, p3
        myResults['K1'], myResults['K2'], myResults['K3'] = K1, K2, K3
        myResults['beta12'], myResults['beta23'] = beta12, beta23
        # compute velocities and perfusion
        if 'vel1' in out_vars: myResults['vel1'] = project(-K1*grad(p1),Vvel, solver_type='bicgstab', preconditioner_type='amg')
        if 'vel2' in out_vars: myResults['vel2'] = project(-K2*grad(p2),Vvel, solver_type='bicgstab', preconditioner_type='amg')
        if 'vel3' in out_vars: myResults['vel3'] = project(-K3*grad(p3),Vvel, solver_type='bicgstab', preconditioner_type='amg')
        if 'perfusion' in out_vars: myResults['perfusion'] = project(beta12 * (p1-p2)*6000,K2_space, solver_type='bicgstab', preconditioner_type='amg')
    elif compartmental_model == 'a':
        myResults['press1'] = p
        myResults['K1'] = K1
        myResults['beta12'] = beta12
        # compute velocities and perfusion
        if 'v1' in out_vars: myResults['vel1'] = project(-K1*grad(p),Vvel, solver_type='bicgstab', preconditioner_type='amg')
        if 'perfusion' in out_vars: myResults['perfusion'] = project(beta12 * (p-Constant(p_venous))*6000,K2_space, solver_type='bicgstab', preconditioner_type='amg')
    else:
        raise Exception("unknown model type: " + model_type)
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

#%%

# TODO: implement flexible averaging and solution for model type 'a'
if configs['output']['comp_ave'] == True:
    # obtain fluxes (ID, surface area, flux1, flux2, flux3)
    fluxes, surf_p_values = suppl_fcts.surface_ave(mesh,boundaries,vels,ps)
    
    # obtain some characteristic values within the domain (ID, volume, average, min, max)
    vol_p_values, vol_vel_values = suppl_fcts.vol_ave(mesh,subdomains,ps,vels)
    
    if rank ==0:
        # print(fluxes,'\n')
        # print(surf_p_values,'\n')
        # print(vol_p_values,'\n')
        # print(vol_vel_values,'\n')
        
        fheader = 'surface ID, Area [mm^2], Qa [mm^3/s], Qc [mm^3/s], Qv [mm^3/s]'
        numpy.savetxt(configs['output']['res_fldr']+'fluxes.csv', fluxes,"%d,%e,%e,%e,%e",header=fheader)
        
        fheader = 'surface ID, Area [mm^2], pa [Pa], pc [Pa], pv [Pa]'
        numpy.savetxt(configs['output']['res_fldr']+'surf_p_values.csv', surf_p_values,"%d,%e,%e,%e,%e",header=fheader)
        
        fheader = 'volume ID, Volume [mm^3], pa [Pa], pc [Pa], pv [Pa]'
        numpy.savetxt(configs['output']['res_fldr']+'vol_p_values.csv', vol_p_values,"%e,%e,%e,%e,%e",header=fheader)
        
        fheader = 'volume ID, Volume [mm^3], ua [m/s], uc [m/s], uv [m/s]'
        numpy.savetxt(configs['output']['res_fldr']+'vol_vel_values.csv', vol_vel_values,"%d,%e,%e,%e,%e",header=fheader)

end3 = time.time()
end0 = time.time()

#%% REPORT EXECUTION TIME
if rank == 0:
    oldstdout = sys.stdout
    logfile = open(configs['output']['res_fldr']+"time_info.log", 'w')
    sys.stdout = logfile
    print ('Total execution time [s]; \t\t\t', end0 - start0)
    print ('Step 1: Reading input files [s]; \t\t', end1 - start1)
    print ('Step 2: Solving governing equations [s]; \t\t', end2 - start2)
    print ('Step 3: Preparing and saving output [s]; \t\t', end3 - start3)
    logfile.close()
    sys.stdout = oldstdout
    print ('Execution time: \t', end0 - start0, '[s]')
    print ('Step 1: \t\t', end1 - start1, '[s]')
    print ('Step 2: \t\t', end2 - start2, '[s]')
    print ('Step 3: \t\t', end3 - start3, '[s]')