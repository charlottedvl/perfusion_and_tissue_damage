"""
Multi-compartment advection-reaction-diffusion problem describing oxygen
ransport in the brain.

System of equations (steady state)
Dot(u_a Grad(C_a)) = phi_a D_a Div(Grad(C_a)) - beta_ac (p_a-p_c) C_a 
                     - gamma_a SaVa phi_a (tau C_a - C_t)
Dot(u_c Grad(C_c)) = phi_c D_c Div(Grad(C_c)) + beta_ac (p_a-p_c) C_a - beta_cv (p_c-p_v) C_c 
                     - gamma_c ScVc phi_c (tau C_c - C_t)
Nonlinear metabolism:
0 = phi_t D_t Div(Grad(C_t)) + gamma_a SaVa phi_a (tau C_a - C_t) 
    + gamma_c ScVc phi_ c(tau C_c - C_t) - phi_t G C_t / (C50-C_t)
Linear metabolism:
0 = phi_t D_t Div(Grad(C_t)) + gamma_a SaVa phi_a (tau C_a - C_t) 
    + gamma_c ScVc phi_ c(tau C_c - C_t) - phi_t M C_t


C - Oxygen concentration [ml O2 / mm^3 tissue]

u - Darcy velocity [mm/s]
p - volume averaged pressure [Pa]
beta - flow coupling coefficient [1 / Pa s]

D - Effective diffusion coefficient [mm^2 / s]
M - Linear metabolism
G - Michaelis-Menten kinetics: maximum consumption rate
C50 - the oxygen concentration at the half-maximum consumption rate
SV - vessel surface area to vessel volume ratio [mm^2 / mm^3]
tau - plasma oxygen to blood oxygen ratio [-]
phi - vessel volume fraction [-]
gamma - vessel wall transport coefficient [mm^3 / mm^2 s]


@author: Yun Bing (Ice)
"""

#%% Import modules
# python3 modules
from dolfin import *
import numpy as np
import time
import sys
import argparse
# user modules
import FE_solver, IO_funcs

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm=MPI.comm_world
rank=comm.Get_rank()
size=comm.Get_size()

start0=time.time()

#%% Inputs
if rank==0: print('----Reading input files, initialising functions and parameters----')
start1=time.time()

parser=argparse.ArgumentParser()
parser.add_argument("--config_file", help="path of configuration file", type=str, default='./config_oxygen_solver.yaml')
parser.add_argument("--rslt", help="path fo results folder (string ended with /)", type=str, default=None)
config_file=parser.parse_args().config_file

configs=IO_funcs.oxygen_config_reader(config_file, parser)
# paramters
phiA, phiC, phiT=configs.parameter.phiA, configs.parameter.phiC, configs.parameter.phiT
D_a, D_c, D_t=configs.parameter.D_a, configs.parameter.D_c, configs.parameter.D_t
SaVa, ScVc=configs.parameter.SaVa, configs.parameter.ScVc
gammaA, gammaC=configs.parameter.gammaA, configs.parameter.gammaC
tau=configs.parameter.tau
M, G, C50=configs.parameter.M, configs.parameter.G, configs.parameter.C50

# read in meah
mesh, subdomains, boundaries=IO_funcs.mesh_reader_xdmf(configs.input.mesh_file)
# mesh, subdomains, boundaries=IO_funcs.mesh_reader_h5(configs.input.mesh_file)

# assign function space and read in pressure, velocity and beta
Vc, DGSpace, CGSpace, uSpace, beta_ac, beta_cv,\
pa, pc, pv, ua, uc, depth=FE_solver.func_space(mesh, configs.simulation.eleD, configs)

end1=time.time()

#%%
if rank==0: print('----Computing artificial diffusion and boundary conditions----')
start2=time.time()

# compute artificial deffusion
dalta=FE_solver.art_diff(mesh, ua, D_a, DGSpace, depth, configs)
# assign DBC
BCa=FE_solver.BC(boundaries, Vc, configs)

end2=time.time()

#%% Solve for concentration
if rank==0: print('----Solving variational equations----')
start3=time.time()

if configs.simulation.nonLinear==False:
    Ca, Cc, Ct=FE_solver.O2_Linear(beta_ac,beta_cv,mesh,Vc,pa,pc,pv,ua,uc,\
                                   phiA,phiC,phiT,dalta,D_c,D_t,\
                                   SaVa,ScVc,gammaA,gammaC,tau,M,BCa)
else:
    Ca, Cc, Ct=FE_solver.O2_nonLinear(beta_ac,beta_cv,mesh,Vc,pa,pc,pv,ua,uc,\
                                      phiA,phiC,phiT,dalta,D_c,D_t,\
                                      SaVa,ScVc,gammaA,gammaC,tau,G,C50,BCa)
        
end3=time.time()
        
#%% Save solution
if rank==0: print('----Saving solutions----')
start4=time.time()

C=[Ca,Cc,Ct]
for i in range(3):
    IO_funcs.xdmf_h5_saver(C[i], 'C'+str(i+1), configs.output.rslt)

end4=time.time()
end0=time.time()

#%% Execution time
if rank == 0:
    oldstdout = sys.stdout
    logfile = open(configs.output.rslt+"time_info.log", 'w')
    sys.stdout = logfile
    print ('Total execution time [s]; \t\t\t', end0 - start0)
    print ('Step 1: Reading input files, initialising functions and parameters [s]; \t\t', end1 - start1)
    print ('Step 2: Computing artificial diffusion and boundary conditions [s]; \t\t', end2 - start2)
    print ('Step 3: Solving variational equations [s]; \t\t', end3 - start3)
    print ('Step 4: Saving solutions [s]; \t\t', end4 - start4)
    logfile.close()
    sys.stdout = oldstdout
    print ('Execution time: \t', end0 - start0, '[s]')
    print ('Step 1: \t\t\t', end1 - start1, '[s]')
    print ('Step 2: \t\t\t', end2 - start2, '[s]')
    print ('Step 3: \t\t\t', end3 - start3, '[s]')
    print ('Step 4: \t\t\t', end4 - start4, '[s]')