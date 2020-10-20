# TODO: define cost function
def cost_function(param_values,configs):
    for i in range(len(configs['optimisation']['parameters'])):
        configs['physical'][ configs['optimisation']['parameters'][i] ] = pow(10,param_values[i])
    return configs
    
    

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
import yaml
from scipy.optimize import minimize

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
                    type=str, default='./config_basic_flow_solver.yml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader_yml(config_file,parser)

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'])
    
# initialise permeability tensors
K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space,K2_space,mesh,configs['input']['permeability_folder'])

# physical parameters
p_arterial, p_venous = configs['physical']['p_arterial'], configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat']
beta12gm, beta23gm, gmowm_beta_rat = \
    configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']


param_values = []
for i in range(len(configs['optimisation']['parameters'])):
    param_values.append( configs['physical'][ configs['optimisation']['parameters'][i] ] )
param_values = numpy.log10( numpy.array(param_values) )

# TODO: add random initialisation option

iter_info = []
#    init_cost1 = cost_function(x0,Vp,K2_ref,K1in,K2in,K3in,v_1,v_2,v_3,BCs,integrals_N,subdomains,iter_info,inlet_BC_type, beta12, beta23, True)

# TODO: organise function call and output
# res = minimize(cost_function, x0, \
#     args=(Vp,K2_ref,K1in,K2in,K3in,v_1,v_2,v_3,BCs,integrals_N,subdomains, \
#           iter_info,inlet_BC_type, beta12, beta23, False), \
#           method=opt_method, bounds=[(3,4),(0,1)], \
#     options={'disp':True, 'maxfev':1000, 'maxiter':len(x0)*800})
# print('\n\n',res.x,'\n')

# fheader = 'K1oK2, gmowm_beta_rat, P_min, P_max, P_wm, P_gm, J'

# np.savetxt('opt_res_v10_'+inlet_BC_type + '_' + opt_method + '_' + str(i) + '.csv', np.array(iter_info),"%e,%e,%e,%e,%e,%e,%e",header=fheader)






# if rank == 0: print('\t Scaling coupling coefficients and permeability tensors')

# # set coupling coefficients
# beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
#                                 beta12gm, beta23gm, gmowm_beta_rat, \
#                                 K2_space, configs.output.res_fldr, configs.output.save_pvd)

# K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3, \
#                                   K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
#                                   configs.output.res_fldr,configs.output.save_pvd)
# end1 = time.time()


# #%% SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
# if rank == 0: print('Step 2: Defining and solving governing equations')
# start2 = time.time()

# # set up finite element solver
# # TODO: handle Neuman/dirichlet boundary conditions
# LHS, RHS, sigma1, sigma2, sigma3, BCs = \
# fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, \
#                          p, p1, p2, p3, K1, K2, K3, beta12, beta23, \
#                          p_arterial, p_venous, \
#                          configs.input.read_inlet_boundary, configs.input.inlet_boundary_file, configs.input.inlet_BC_type)

# lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

# # tested iterative solvers for first order elements: gmres, cg, bicgstab
# #linear_solver_methods()
# #krylov_solver_preconditioners()
# if rank == 0: print('\t pressure computation')
# p = fe_mod.solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol)
# end2 = time.time()


# #%% COMPUTE VELOCITY FIELDS, SAVE SOLUTION, EXTRACT FIELD VARIABLES
# if rank == 0: print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
# start3 = time.time()

# p1, p2, p3=p.split()

# perfusion = project(beta12 * (p1-p2)*6000,K2_space, solver_type='bicgstab', preconditioner_type='amg')

# # compute velocities
# vel1 = project(-K1*grad(p1),Vvel, solver_type='bicgstab', preconditioner_type='amg')
# vel2 = project(-K2*grad(p2),Vvel, solver_type='bicgstab', preconditioner_type='amg')
# vel3 = project(-K3*grad(p3),Vvel, solver_type='bicgstab', preconditioner_type='amg')

# ps = [p1, p2, p3]
# vels = [vel1, vel2, vel3]
# Ks = [K1, K2, K3]

# vars2save = [ps, vels, Ks]
# fnames = ['press','vel','K']
# for idx, fname in enumerate(fnames):
#     for i in range(3):
#         with XDMFFile(configs.output.res_fldr+fname+str(i+1)+'.xdmf') as myfile:
#             myfile.write_checkpoint(vars2save[idx][i],fname+str(i+1), 0, XDMFFile.Encoding.HDF5, False)

# with XDMFFile(configs.output.res_fldr+'beta12.xdmf') as myfile:
#     myfile.write_checkpoint(beta12,"beta12", 0, XDMFFile.Encoding.HDF5, False)
# with XDMFFile(configs.output.res_fldr+'beta23.xdmf') as myfile:
#     myfile.write_checkpoint(beta23,"beta23", 0, XDMFFile.Encoding.HDF5, False)
# with XDMFFile(configs.output.res_fldr+'perfusion.xdmf') as myfile:
#     myfile.write_checkpoint(perfusion,'perfusion', 0, XDMFFile.Encoding.HDF5, False)

# fheader = 'FE degree, K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, beta12gm, beta23gm, gmowm_beta_rat'
# dom_props = numpy.array([configs.simulation.fe_degr,K1gm_ref,K2gm_ref,K3gm_ref,gmowm_perm_rat,beta12gm,beta23gm,gmowm_beta_rat])
# numpy.savetxt(configs.output.res_fldr+'dom_props.csv', [dom_props],"%d,%e,%e,%e,%e,%e,%e,%e",header=fheader)

# #%%

# if configs.output.comp_ave == True:
#     # obtain fluxes (ID, surface area, flux1, flux2, flux3)
#     fluxes, surf_p_values = suppl_fcts.surface_ave(mesh,boundaries,vels,ps)
    
#     # obtain some characteristic values within the domain (ID, volume, average, min, max)
#     vol_p_values, vol_vel_values = suppl_fcts.vol_ave(mesh,subdomains,ps,vels)
    
#     if rank ==0:
#         # print(fluxes,'\n')
#         # print(surf_p_values,'\n')
#         # print(vol_p_values,'\n')
#         # print(vol_vel_values,'\n')
        
#         fheader = 'surface ID, Area [mm^2], Qa [mm^3/s], Qc [mm^3/s], Qv [mm^3/s]'
#         numpy.savetxt(configs.output.res_fldr+'fluxes.csv', fluxes,"%d,%e,%e,%e,%e",header=fheader)
        
#         fheader = 'surface ID, Area [mm^2], pa [Pa], pc [Pa], pv [Pa]'
#         numpy.savetxt(configs.output.res_fldr+'surf_p_values.csv', surf_p_values,"%d,%e,%e,%e,%e",header=fheader)
        
#         fheader = 'volume ID, Volume [mm^3], pa [Pa], pc [Pa], pv [Pa]'
#         numpy.savetxt(configs.output.res_fldr+'vol_p_values.csv', vol_p_values,"%e,%e,%e,%e,%e",header=fheader)
        
#         fheader = 'volume ID, Volume [mm^3], ua [m/s], uc [m/s], uv [m/s]'
#         numpy.savetxt(configs.output.res_fldr+'vol_vel_values.csv', vol_vel_values,"%d,%e,%e,%e,%e",header=fheader)

# end3 = time.time()
# end0 = time.time()

# #%% REPORT EXECUTION TIME
# if rank == 0:
#     oldstdout = sys.stdout
#     logfile = open(configs.output.res_fldr+"time_info.log", 'w')
#     sys.stdout = logfile
#     print ('Total execution time [s]; \t\t\t', end0 - start0)
#     print ('Step 1: Reading input files [s]; \t\t', end1 - start1)
#     print ('Step 2: Solving governing equations [s]; \t\t', end2 - start2)
#     print ('Step 3: Preparing and saving output [s]; \t\t', end3 - start3)
#     logfile.close()
#     sys.stdout = oldstdout
#     print ('Execution time: \t', end0 - start0, '[s]')
#     print ('Step 1: \t\t', end1 - start1, '[s]')
#     print ('Step 2: \t\t', end2 - start2, '[s]')
#     print ('Step 3: \t\t', end3 - start3, '[s]')