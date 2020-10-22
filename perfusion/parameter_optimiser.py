# TODO: define cost function
def cost_function(param_values,configs,mesh,subdomains,boundaries,K2_space,K1form,K2form,K3form,p,p1,p2,p3,iter_info,save_fields):
    for i in range(len(configs['optimisation']['parameters'])):
        configs['physical'][ configs['optimisation']['parameters'][i] ] = pow(10,param_values[i])
    
    # set coupling coefficients
    beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
                                    configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat'], \
                                    K2_space, configs['output']['res_fldr'], configs['output']['save_pvd'])
    # set permeabilities
    K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1form.copy(deepcopy=True), K2form.copy(deepcopy=True), K3form.copy(deepcopy=True), \
                                      configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat'], \
                                      configs['output']['res_fldr'],configs['output']['save_pvd'])
    
    # set up finite element solver
    LHS, RHS, sigma1, sigma2, sigma3, BCs = \
        fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, \
             p, p1, p2, p3, K1, K2, K3, beta12, beta23, \
             configs['physical']['p_arterial'], configs['physical']['p_venous'], \
             configs['input']['read_inlet_boundary'], configs['input']['inlet_boundary_file'], configs['input']['inlet_BC_type'])
    
    lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False
    
    try:
        psol = fe_mod.solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol,timer=False)
        
        p1sol, p2sol, p3sol=psol.split()
        
        perfusion = project(beta12 * (p1sol-p2sol)*6000,K2_space, solver_type='bicgstab', preconditioner_type='amg')
        
        FW = assemble( perfusion*dV(11) )/V_wm
        FG = assemble( perfusion*dV(12) )/V_gm
        #P_brain = assemble( perfusion*dx )/V_brain
        
# TODO: use MPI to compute global minimum and maximum
        Fmin = min(perfusion.vector()[:])
        Fmax = max(perfusion.vector()[:])
        
        J = 0
        J = int(Fmin<configs['optimisation']['Fmintarget'])*pow(Fmin-configs['optimisation']['Fmintarget'],2)   \
            + int(Fmax>configs['optimisation']['Fmaxtarget'])*pow(Fmax-configs['optimisation']['Fmaxtarget'],2) \
            + pow(FW-configs['optimisation']['FWtarget'],2) + pow(FG-configs['optimisation']['FGtarget'],2)
        
    except RuntimeError:
        J = 1e15
    
    if save_fields == True:
        wdir = "opt_field_res/"
        vtkfile = File(wdir+"p1.pvd")
        vtkfile << p1
        vtkfile = File(wdir+"p2.pvd")
        vtkfile << p2
        vtkfile = File(wdir+"p3.pvd")
        vtkfile << p3
        
        vtkfile = File(wdir+"K1.pvd")
        vtkfile << K1c
        vtkfile = File(wdir+"K2.pvd")
        vtkfile << K2c
        vtkfile = File(wdir+"K3.pvd")
        vtkfile << K3c
        
        vtkfile = File(wdir+"beta12.pvd")
        vtkfile << beta12
        vtkfile = File(wdir+"beta23.pvd")
        vtkfile << beta23
    
    info = list(pow(10,param_values))
    info.append(Fmin)
    info.append(Fmax)
    info.append(FW)
    info.append(FG)
    info.append(J)
    iter_info.append(info)
    if (len(iter_info)-2) % 5 == 0:
        print(len(iter_info)-2,info)
    print(Fmin,Fmax,FG,FW,J)
    return J
    
    

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
# determine regional volumes
dV = dx(subdomain_data=subdomains)
V_wm = assemble( Constant(1.0)*dV(11,domain=mesh) )
V_gm = assemble( Constant(1.0)*dV(12,domain=mesh) )
V_brain = assemble( Constant(1.0)*dx(domain=mesh)  )


# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'])
    
# initialise permeability tensors
K1form, K2form, K3form = IO_fcts.initialise_permeabilities(K1_space,K2_space,mesh,configs['input']['permeability_folder'])

param_values = []
for i in range(len(configs['optimisation']['parameters'])):
    param_values.append( configs['physical'][ configs['optimisation']['parameters'][i] ] )
param_values = numpy.log10( numpy.array(param_values) )

#%% OPTIMISATION
iter_info = []
save_fields = False

# # TODO: add random initialisation option


# #    init_cost1 = cost_function(x0,Vp,K2_ref,K1in,K2in,K3in,v_1,v_2,v_3,BCs,integrals_N,subdomains,iter_info,inlet_BC_type, beta12, beta23, True)

# # TODO: organise function call and output
# res = minimize(cost_function, param_values, \
#       args=(configs,mesh,subdomains,boundaries,K2_space,K1form,K2form,K3form,p,p1,p2,p3,iter_info,False),
#       method=configs['optimisation']['method'], bounds=[(3,4),(0,1)],
#       options={'disp':True, 'maxfev':1000, 'maxiter':len(param_values)*800})
# # #     args=(Vp,K2_ref,K1in,K2in,K3in,v_1,v_2,v_3,BCs,integrals_N,subdomains, \
# # #           iter_info,inlet_BC_type, beta12, beta23, False), \
# # #           method=opt_method, bounds=[(3,4),(0,1)], \
# # #     options={'disp':True, 'maxfev':1000, 'maxiter':len(x0)*800})
# print('\n\n',res.x,'\n')

# fheader =''
# data_format = ''
# for i in range(len(configs['optimisation']['parameters'])):
#     fheader += configs['optimisation']['parameters'][i] + ', '
#     data_format += '%e,'
# fheader +='Fmin, Fmax, FW, FG, J'
# data_format += '%e,%e,%e,%e,%e'

# np.savetxt('opt_res_' + configs['optimisation']['method'] + '.csv', np.array(iter_info),data_format,header=fheader)


#%% DEBUG

# for i in range(len(configs['optimisation']['parameters'])):
#     configs['physical'][ configs['optimisation']['parameters'][i] ] = pow(10,param_values[i])

# # set coupling coefficients
# beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
#                                 configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat'], \
#                                 K2_space, configs['output']['res_fldr'], configs['output']['save_pvd'])
# # set permeabilities
# K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1form.copy(deepcopy=True), K2form.copy(deepcopy=True), K3form.copy(deepcopy=True), \
#                                   configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat'], \
#                                   configs['output']['res_fldr'],configs['output']['save_pvd'])

# # set up finite element solver
# LHS, RHS, sigma1, sigma2, sigma3, BCs = \
#     fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, \
#          p, p1, p2, p3, K1, K2, K3, beta12, beta23, \
#          configs['physical']['p_arterial'], configs['physical']['p_venous'], \
#          configs['input']['read_inlet_boundary'], configs['input']['inlet_boundary_file'], configs['input']['inlet_BC_type'])

# lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

# try:
#     psol = fe_mod.solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol)
    
#     p1sol, p2sol, p3sol=psol.split()
    
#     perfusion = project(beta12 * (p1sol-p2sol)*6000,K2_space, solver_type='bicgstab', preconditioner_type='amg')
    
#     FW = assemble( perfusion*dV(11) )/V_wm
#     FG = assemble( perfusion*dV(12) )/V_gm
#     #P_brain = assemble( perfusion*dx )/V_brain
    
#     Fmin = min(perfusion.vector()[:])
#     Fmax = max(perfusion.vector()[:])
    
#     J = 0
#     J = int(Fmin<configs['optimisation']['Fmintarget'])*pow(Fmin-configs['optimisation']['Fmintarget'],2)   \
#         + int(Fmax>configs['optimisation']['Fmaxtarget'])*pow(Fmax-configs['optimisation']['Fmaxtarget'],2) \
#         + pow(FW-configs['optimisation']['FWtarget'],2) + pow(FG-configs['optimisation']['FGtarget'],2)
    
# except RuntimeError:
#     J = 1e15

# print(Fmin,Fmax,FG,FW,J)

start = time.time()
cost_function(param_values,configs,mesh,subdomains,boundaries,K2_space,K1form,K2form,K3form,p,p1,p2,p3,iter_info,save_fields)
end = time.time()
print ('\t\t a single iteration took', end - start, '[s]')