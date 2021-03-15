"""
Estimate the tissue health (infarct fraction) based on perfusion in each element 
and the Green's function simulation results

This code considers the treatment (perfusion recovery) and its outcome

p - perfusion [mL/100mL/min]
t - time [hour]

Yidan Xue - 2021/03
"""
from dolfin import *
import scipy.interpolate
from scipy.integrate import odeint
import numpy as np
import time

# added module
import IO_fcts
import finite_element_fcts as fe_mod

# input value

arrival_time = 3 # time between onset and imaging [hours]
treatment_time = 120 # time after treatment [hours]
treatment_effect = 0.8 # how much percentage the perfusion will be restored

# define the ODE for cell death

# ODE parameters - no extravasation
kf, kt, kb = 0.00044719, 0.00039441, 0.00215699

# kf - forward rate constants
# kt - toxic production constant
# kb - toxic recycle constant

# h[0] - dead, h[1] - toxic, h[2] - hypoxic fraction
def cell_death(h, t):
    a = 1 - h[0]
    hypo = h[2]
    return [a*kf*h[1], a*kt*hypo-kb*h[1]*(1-hypo)*a, 0]

# define the relationship between hypoxic fraction and perfusion - based on Green's function simulations
def hypoxia_estimate(perfusion):
    return 1-1/(1+np.exp(-(0.175*perfusion+(-3.304))))

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()

# healthy_perfusion_wm_gm

perfusion_gm = 56.04585125316684
perfusion_wm = 20.759413218846582
perfusion_scale = perfusion_gm/perfusion_wm

start0 = time.time()

# %% READ INPUT
if rank == 0: print('Step 1: Reading and interpolating the cell death simulation results')

# fit the 2D plane - this might be optimised in the future
# perfusion,time,dead = np.loadtxt('./dead_fraction.csv', unpack=True)
# dead_estimate = scipy.interpolate.interp2d(perfusion,time,dead,kind='quintic')

# read the mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader('../brain_meshes/b0000/clustered.xdmf')
fe_order = 1

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, fe_order)

# read simulation results - the folder name should match the case
healthyfile = '../VP_results/p0000/healthy/perfusion.xdmf'
strokefile = '../VP_results/p0000/mild_perfusion_occlusion/perfusion.xdmf'

# distinguish white and grey matter
dV = dx(subdomain_data=subdomains)
wm_idx = subdomains.where_equal(11)# white matter cell indices
gm_idx = subdomains.where_equal(12)# gray matter cell indices
num_wm_idx = len(wm_idx)
num_gm_idx = len(gm_idx)
num = len(wm_idx)+len(gm_idx) # total number of elements

if rank == 0:
    print('Step 2: Reading perfusion files')
# load previous results

# read the healthy perfusion map
perfusion_healthy = Function(K2_space)
f_in = XDMFFile(healthyfile)
f_in.read_checkpoint(perfusion_healthy,'perfusion', 0)
f_in.close()

# read the perfusion map after stroke before treatment
perfusion_stroke = Function(K2_space)
f_in = XDMFFile(strokefile)
f_in.read_checkpoint(perfusion_stroke,'perfusion', 0)
f_in.close()

if rank == 0:
    print('Step 3: Calculating the infarct fraction')

# calculate the treatment perfuison

perfusion_healthy.vector()
perfusion_healthy.vector()[:]
perfusion_healthy_vec = perfusion_healthy.vector().get_local()
perfusion_stroke.vector()
perfusion_stroke.vector()[:]
perfusion_stroke_vec = perfusion_stroke.vector().get_local()
perfusion_treatment = Function(K2_space)
perfusion_treatment.vector()
perfusion_treatment.vector()[:]
perfusion_treatment_vec = perfusion_treatment.vector().get_local()
for i in range(num):
    perfusion_treatment_vec[i] = perfusion_stroke_vec[i]+treatment_effect*(perfusion_healthy_vec[i]-perfusion_stroke_vec[i])

# define the dead fraction and toxic state in each element

dead = Function(K2_space)
dead.vector()
dead.vector()[:]
dead_vec = dead.vector().get_local()
toxic = Function(K2_space)
toxic.vector()
toxic.vector()[:]
toxic_vec = toxic.vector().get_local()

# initialise the ODEs
# before treatment - time in seconds now
t_b = np.linspace(0,arrival_time*3600,arrival_time*120+1)
t_a = np.linspace(0,treatment_time*3600,treatment_time*12+1)

# for grey matter
for i in range(num_gm_idx):
    # if the change in perfuison is smaller than 5%
    if (perfusion_healthy_vec[gm_idx[i]]-perfusion_stroke_vec[gm_idx[i]])/perfusion_healthy_vec[gm_idx[i]] < 0.05:
        dead_vec[gm_idx[i]] = 0
    # if the change is larger
    else:
        hi1 = [0,0,hypoxia_estimate(perfusion_stroke_vec[gm_idx[i]])] # first input: after onset
        hs = odeint(cell_death, hi1, t_b)
        Dead = hs[-1,0]
        Toxic = hs[-1,1]
        hi2 = [Dead,Toxic,hypoxia_estimate(perfusion_treatment_vec[gm_idx[i]])] # second input: after treatment
        hs = odeint(cell_death, hi2, t_a)
        dead_vec[gm_idx[i]] = hs[-1,0]

# for white matter
for i in range(num_wm_idx):
    # if the change in perfuison is smaller than 5%
    if (perfusion_healthy_vec[wm_idx[i]]-perfusion_stroke_vec[wm_idx[i]])/perfusion_healthy_vec[wm_idx[i]] < 0.05:
        dead_vec[wm_idx[i]] = 0
    # if the change is larger
    else:
        hi1 = [0,0,hypoxia_estimate(perfusion_stroke_vec[wm_idx[i]]*perfusion_scale)] # first input: after onset
        hs = odeint(cell_death, hi1, t_b)
        Dead = hs[-1,0]
        Toxic = hs[-1,1]
        hi2 = [Dead,Toxic,hypoxia_estimate(perfusion_treatment_vec[wm_idx[i]])*perfusion_scale] # second input: after treatment
        hs = odeint(cell_death, hi2, t_a)
        dead_vec[wm_idx[i]] = hs[-1,0]	
    
dead.vector().set_local(dead_vec)

vtkfile = File('/Users/xueyidan/Desktop/infarct_'+str(arrival_time)+'_'+str(treatment_time)+'.pvd')
vtkfile << dead

end0 = time.time()

print('Simulation finished - Total execution time [s]; \t\t\t', end0 - start0)
